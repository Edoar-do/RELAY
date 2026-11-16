import torch
from dataset import ECGDataset
from config import *
from consts import LEADS
from modelling_graph_ecg import MODEL_NAME_2_FINETUNING_MODEL
import argparse
from loguru import logger
import torch.cuda.amp as amp
from tqdm import tqdm
import numpy as np
import os
import random
from utils import get_metrics, get_criterions
import pandas as pd
from torch.utils.data import DataLoader
import threadpoolctl

def worker_init_fn(worker_id):
    torch.set_num_threads(1)
    threadpoolctl.threadpool_limits(limits=1)
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    
def random_crop(batch, crop_size=500, nodes_per_lead=20): # Make sure to pass nodes_per_lead or access it from config
    
    # We need to find which graph instance each row of x belongs to.
    # Since x[i] produces nodes [i*S ... (i+1)*S], we can check the batch index of the first node of each lead.
    # Note: This assumes fixed nodes_per_lead.
    
    # Extract the instance ID for each lead (row in x)
    # We take every S-th element from the node-level batch vector
    lead_to_instance_id = batch.batch[::nodes_per_lead] 

    unique_instance_ids = lead_to_instance_id.unique()
    new_batch_x = torch.zeros((batch.x.shape[0], crop_size), dtype=batch.x.dtype)

    for instance_id in unique_instance_ids:
        # Create a mask for leads (not nodes) belonging to this instance
        instance_mask = (lead_to_instance_id == instance_id)
        
        instance_data = batch.x[instance_mask]
        
        if instance_data.shape[1] > crop_size:
            start = torch.randint(0, instance_data.shape[1] - crop_size + 1, (1,)).item()
            end = start + crop_size
            new_batch_x[instance_mask] = instance_data[:, start:end]
        else:
            # Handle cases where signal is shorter than crop_size (optional padding or copy)
            new_batch_x[instance_mask] = instance_data[:, :crop_size] 

    batch.x = new_batch_x
    return batch

def get_all_crops(batch, crop_size=500, stride=250, nodes_per_lead=20):
    """
    Generate a list of Pytorch Geometric batches, each containing a different crop x of the original batch.x.
    The edge index, y, node_lead_ids, node_temporal_pos, and batch vectors remain the same across all batches,
    since we are only cropping in the temporal dimension of x.

    Since every lead row is sliced with the same temporal window, this is just a contiguous slice of
    batch.x along the time axis — no per-instance masking needed.
    """
    with torch.no_grad():
        signal_length = batch.x.shape[1]
        crops = []
        for start in range(0, signal_length - crop_size + 1, stride):
            end = start + crop_size
            new_batch = batch.clone()
            new_batch.x = batch.x[:, start:end].contiguous()
            crops.append(new_batch)
        return crops


def _run_batches(args, model, device, test_loader, test_criterion, metrics):
    """Runs the full test loop once with the given dataloader, updating `metrics` in-place."""
    for metric in metrics.values():
        metric.reset()

    for batch_id, batch in enumerate(tqdm(test_loader, total=len(test_loader))):

        if args.tta:
            augs = get_all_crops(batch, crop_size=500, stride=250, nodes_per_lead=model.config.nodes_per_lead)
        else:
            augs = [batch]

        probs = []

        for aug in augs:
            x = aug.x.to(device)
            edge_index = aug.edge_index.to(device)
            labels = aug.y.to(device)
            node_lead_ids = aug.node_lead_ids.to(device)
            node_temporal_pos = aug.node_temporal_pos.to(device)
            b = aug.batch.to(device)

            with torch.no_grad():
                output = model(
                    x=x,
                    edge_index=edge_index,
                    criterion=test_criterion,
                    labels=labels,
                    node_lead_ids=node_lead_ids,
                    node_temporal_pos=node_temporal_pos,
                    batch=b,
                    batch_size=args.batch_size,
                    output_hidden_states=False,
                    return_dict=True,
                    metrics={}
                )

            logits = output['logits']

            if args.task == "multi_label":
                probs.append(torch.sigmoid(logits))
            elif args.task == "multi_class":
                probs.append(torch.softmax(logits, dim=-1))
            else:
                probs.append(logits)

        probs = torch.stack(probs)
        probs = probs.mean(dim=0) if args.tta_aggregation == "mean" else probs.max(dim=0).values

        if args.save_probs:
            dest_path = f"./probs/{os.path.basename(args.path_to_test_dataset_csv).replace('.csv', '')}"
            os.makedirs(dest_path, exist_ok=True)
            np.save(os.path.join(dest_path, f"probs_{batch_id}_bs{args.batch_size}"), probs.cpu().numpy())

        labels = labels.long()

        if labels.ndim == 1:
            if labels.shape[0] == logits.shape[0] and logits.shape[1] > 1:
                pass
            else:
                labels = labels.reshape(logits.size(0), -1)

        for metric in metrics.values():
            metric.update(probs, labels)
    

def test(args, model):

    # --- fixing seeds ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- dataset and data loader creation ---
    
    if args.use_benchmark_mode:

        dataset = ECGDataset(
            use_benchmarking_format=args.use_benchmark_mode,
            benchmark_folder=args.benchmark_folder,
            label_column=args.label_column,
            path_to_dataset_csv=args.path_to_test_dataset_csv,
            ecg_dir_path=args.ecg_test_dir_path, 
            downsampling_factor=args.downsampling_factor,
            label_start_index=args.label_start_index,
            pretrain=False,
            random_crop=False,
            return_full_length=args.tta,
            create_edge_index=args.create_edge_index, 
            edge_strategy=args.edge_strategy,
            sampling=args.sampling,
            how_many_leads=args.how_many_leads,
            nodes_per_lead=model.config.nodes_per_lead,
            use_temporal_encoding=model.config.use_temporal_encoding,
            use_lead_encoding=model.config.use_lead_encoding
        )
        
        _, _, test_idx = dataset.get_splits()
        
        test_set = torch.utils.data.Subset(dataset, test_idx)

        test_loader = DataLoader(
            dataset=test_set, 
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            prefetch_factor=4 if args.num_workers > 0 else None,
            collate_fn=test_set.dataset.collate_fn,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            persistent_workers=True if args.num_workers > 0 else False
            )
    else:
        test_set = ECGDataset(
            path_to_dataset_csv=args.path_to_test_dataset_csv,
            ecg_dir_path=args.ecg_test_dir_path, 
            downsampling_factor=args.downsampling_factor,
            label_start_index=args.label_start_index,
            pretrain=False,
            random_crop=False,
            return_full_length=args.tta,
            create_edge_index=args.create_edge_index, 
            edge_strategy=args.edge_strategy,
            sampling=args.sampling,
            how_many_leads=args.how_many_leads,
            nodes_per_lead=model.config.nodes_per_lead,
            use_temporal_encoding=model.config.use_temporal_encoding,
            use_lead_encoding=model.config.use_lead_encoding
        )

        test_loader = DataLoader(
            dataset=test_set, 
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            prefetch_factor=4 if args.num_workers > 0 else None,
            collate_fn=test_set.collate_fn,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            persistent_workers=True if args.num_workers > 0 else False
            )

    model.to(device)
    model.eval()

    # --- metrics and criterion creation ---

    metrics = get_metrics(task=args.task, num_classes=model.config.num_classes)
    _, test_criterion = get_criterions(task=args.task)
    
    # --- mapping metrics and criterion to device ---

    for _, metric in metrics.items():
        metric.to(device)
        
    test_criterion.to(device)

    logger.info("Start testing")

    # Resolve the underlying ECGDataset (Subset wraps it in benchmark mode)
    underlying = test_set.dataset if isinstance(test_set, torch.utils.data.Subset) else test_set

    # --- lead-importance mode: one single-lead pass per lead in LEADS ---
    if args.lead_importance:
        per_lead_scores = {name: {} for name in metrics.keys()}

        for lead in LEADS:
            logger.info(f"=== Lead-importance pass: single lead = {lead} ===")
            underlying.sampling = [lead]
            underlying.how_many_leads = 1

            _run_batches(args, model, device, test_loader, test_criterion, metrics)

            for name, metric in metrics.items():
                score = metric.compute()
                logger.info(f"[lead={lead}] {name} (macro) = {score.mean()}")
                per_lead_scores[name][lead] = (
                    score.cpu().numpy() if args.task == "multi_label"
                    else np.array([float(score.mean().cpu())])
                )

        out_dir = f"./performance/{args.dataset_name}/{args.protocol_name}/lead_importance"
        os.makedirs(out_dir, exist_ok=True)
        diagnoses_cols = underlying.diagnoses_cols if hasattr(underlying, 'diagnoses_cols') else None

        for name, lead_dict in per_lead_scores.items():
            columns = diagnoses_cols if (diagnoses_cols is not None and args.task == "multi_label") else None
            heatmap_df = pd.DataFrame(
                data=np.stack([lead_dict[l] for l in LEADS], axis=0),
                index=LEADS,
                columns=columns,
            )
            suffix = f"_{args.save_id}" if args.save_id is not None else ""
            out_path = os.path.join(out_dir, f"heatmap_{name}_{args.tta_aggregation}{suffix}.csv")
            heatmap_df.to_csv(out_path)
            logger.info(f"Lead importance heatmap ({name}) saved at {out_path}")

        logger.info("End of testing")
        return

    # --- standard single-pass testing ---
    _run_batches(args, model, device, test_loader, test_criterion, metrics)

    performance = pd.DataFrame(columns=underlying.diagnoses_cols, index=list(metrics.keys()))

    for name, metric in metrics.items():
        score = metric.compute()
        logger.info(f"\n{name} = {score}, macro = {score.mean()}")
        performance.loc[name] = score.cpu().numpy() if args.task == "multi_label" else score.mean().cpu().numpy()

    os.makedirs(f"./performance/{args.dataset_name}/{args.protocol_name}", exist_ok=True)
    if args.save_id is not None:
        performance.to_csv(f"./performance/{args.dataset_name}/{args.protocol_name}/performance_{args.save_id}_{args.tta_aggregation}.csv")
        logger.info(f"Performance metrics saved at ./performance/{args.dataset_name}/{args.protocol_name}/performance_{args.save_id}_{args.tta_aggregation}.csv")

    logger.info("End of testing")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--use_benchmark_mode",
        default=False,
        action="store_true",
        help="Whether to use benchmarking mode"
    )

    parser.add_argument(
        "--benchmark_folder",
        type=str,
        help="Path to folder with benchmarking csv files. Required if --use_benchmark_mode is set"
    )
    
    parser.add_argument(
        "--label_column",
        help="Name of the column containing labels in the benchmarking dataset. Required if use_benchmarking_format is set",
        type=str,
        choices=['label_filtered', 'label_primary_filtered', 'label_all_filtered', 'label_diag_superclass_filtered', 'label_diag_subclass_filtered','dummy'],
        default='label_filtered'
    )
    
    parser.add_argument(
        "--path_to_test_dataset_csv",
        type=str,
        help="Path to dataset csv file is not in bechmarking mode"
    )
    
    parser.add_argument(
        "--ecg_test_dir_path",
        type=str,
        help="Path to directory with ecg files if not in benchmarking mode"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to a model checkpoint that is to be tested"
    )

    parser.add_argument(
        "--sampling", 
        help="Either 'random' or a list of strings with lead names",
        nargs='+'
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size. Default 64",
        default=64
    )

    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=6,
        help="Num workers. Default 6"
    )
    
    parser.add_argument(
        "--downsampling_factor",
        type=int,
        help="Downsampling factor. Default 5",
        default=5
    )

    parser.add_argument(
        "--label_start_index",
        default=3,
        type=int,
        help="Label start index in dataframe if not in benchmarking mode. Default 3"
    )

    parser.add_argument(
        "--save_id",
        type=str,
        default=None,
        help="The id to use for saving the csv file of performance metrics. Default None"
    )
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        default='dataset_name',
        help="Name of the dataset to better index output performance files. Default 'dataset_name'"
    )
    
    parser.add_argument(
        "--protocol_name",
        type=str,
        default='protocol_name',
        help="Name of the evaluation protocol to better index output performance files. Default 'protocol_name'"
    )

    parser.add_argument(
        "--tta",
        default=False,
        action="store_true",
        help="Whether to use test time augmentation"
    )

    parser.add_argument(
        "--tta_aggregation",
        type=str,
        default='max',
        choices=['mean', 'max'],
        help="Aggregation method for tta. Default 'max'"
    )

    parser.add_argument(
        "--n_augs",
        type=int,
        default=3,
        help="Number of augmentations if not in benchmarking mode. Default 3."
    )

    parser.add_argument(
        "--task",
        type=str,
        default='multi_label',
        choices=['multi_label', 'multi_class', 'regression'],
        help="Task type. Default 'multi_label'"
    )

    parser.add_argument(
        "--save_probs",
        default=False,
        action="store_true",
        help="Whether to save probs"
    )

    parser.add_argument(
        "--create_edge_index", 
        help="Whether to create the edge index. Default False",
        action="store_true"
    )

    parser.add_argument(
        "--edge_strategy",
        type=str,
        choices=['spatiotemporal', 'fully_connected'],
        help="Edge strategy. Default 'spatiotemporal'",
        default='spatiotemporal'
    )
    
    parser.add_argument(
        "--how_many_leads",
        type=int,
        help="Number of leads to randomly select, if sampling='random'. Default None",
        default=None
    )
    
    parser.add_argument(
        "--seed", 
        type=int,
        default=42,
        help="Random seed. Default 42"
    )
    
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--lead_importance",
        default=False,
        action="store_true",
        help="If set, run one pass per lead in LEADS (I, II, III, aVR, aVL, aVF, V1..V6) using "
             "that single lead, and save a lead x class heatmap CSV (per metric)."
    )

    args = parser.parse_args()

    logger.info(f"Loading model from {args.model_path}...")

    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False) 
    model_config = checkpoint['model_config']
    encoder_name = os.path.basename(args.model_path).split('_')[0]
    model_config = MODEL_NAME_2_CONFIG[encoder_name](**model_config.to_dict())
    model = MODEL_NAME_2_FINETUNING_MODEL[encoder_name](model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if args.lead_importance:
        # sampling is overridden per-lead inside test(); keep a harmless placeholder here
        args.sampling = [LEADS[0]]
    elif len(args.sampling) == 1:
        if args.sampling[0] != 'random':
            raise ValueError(f"Argument sampling must be either 'random' or a list of valid lead names. Got {args.sampling}")
        else:
            args.sampling = args.sampling[0] # now it's a string
    else:
        args.sampling = list(set(args.sampling)) # to remove potential duplicates
        
    if args.use_benchmark_mode:
        if args.benchmark_folder is None:
            raise ValueError("Argument --benchmark_folder is required if --use_benchmark_mode is set")
    else:
        if args.path_to_test_dataset_csv is None or os.path.exists(args.path_to_test_dataset_csv) == False:
            raise ValueError("Argument --path_to_test_dataset_csv is required if not in benchmarking mode and must be a valid path")
        if args.ecg_test_dir_path is None or os.path.exists(args.ecg_test_dir_path) == False:
            raise ValueError("Argument --ecg_test_dir_path is required if not in benchmarking mode and must be a valid path")
        
    if not os.path.exists(args.model_path):
        raise ValueError(f"Model path {args.model_path} does not exist")
    
        
    if args.verbose:
        
        ### print args ###
        logger.info("Arguments:")
        for arg in vars(args):
            logger.info(f"{arg}: {getattr(args, arg)}")
        print()
        ##################
        
        ### print model config ###
        logger.info("Model configuration:")
        print(model.config)
        print()
        
        ### print model archictecture ###
        logger.info("Model architecture:")
        print(model)
        print()
    
    logger.info("Model loaded and ready for testing.")

    test(args, model)
    