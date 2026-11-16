
import torch
import json
from dataset import ECGDataset
from config import *
from modelling_graph_ecg import *
import argparse
from loguru import logger
import wandb
import copy 
import torch.cuda.amp as amp
from tqdm import tqdm
import numpy as np
import os
import random
import torch.optim as optim
from consts import WEIGHT_DECAY_DYNAMIC_REG_FACTOR, DROPOUT_DYNAMIC_REG_FACTOR, MINIMAL_IMPROVEMENT_FINETUNING
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from utils import ResumableRandomSampler, get_criterions, get_metrics
import math
import warnings
import threadpoolctl

def worker_init_fn(worker_id):
    torch.set_num_threads(1)
    threadpoolctl.threadpool_limits(limits=1)
    np.random.seed(np.random.get_state()[1][0] + worker_id)

warnings.filterwarnings("ignore")

def validate(args, device, model, val_loader, val_criterion, val_metrics, val_losses):
    with torch.no_grad():
        for batch in tqdm(val_loader, total=len(val_loader)):
                        # --- unpack ---
            x = batch.x
            edge_index = batch.edge_index
            labels = batch.y
            node_lead_ids = batch.node_lead_ids
            node_temporal_pos = batch.node_temporal_pos
            b = batch.batch

                        # --- move to cuda ---
            x = x.to(device)
            edge_index = edge_index.to(device)
            labels = labels.to(device)
            node_lead_ids = node_lead_ids.to(device)
            node_temporal_pos = node_temporal_pos.to(device)
            b = b.to(device)

            # with torch.autocast(
            #     "cuda",
            #     enabled=True,
            #     dtype=torch.bfloat16
            #     ):
            output = model(
                        x=x,
                        edge_index=edge_index,
                        criterion=val_criterion,
                        labels=labels,
                        node_lead_ids=node_lead_ids,
                        node_temporal_pos=node_temporal_pos,
                        batch=b,
                        batch_size=args.batch_size,
                        output_hidden_states=False,
                        return_dict=True,
                        metrics=val_metrics
                    )
                                
            loss = output['loss']
            val_losses.append(loss.item())
            
    return val_losses

def dynamic_regularizer(optimizer, model, penalty, wd_dynamic_reg_factor=WEIGHT_DECAY_DYNAMIC_REG_FACTOR, dropout_dynamic_reg_factor=DROPOUT_DYNAMIC_REG_FACTOR):
    if penalty:
        # penalizing model with regularization but not too much
        optimizer.param_groups[0]['weight_decay'] *= wd_dynamic_reg_factor
        for name, module in model.named_modules():
            if 'dropout' in name.lower():
                module.p += dropout_dynamic_reg_factor
    else:
        # unburdening model from regularization
        # minimum attainable weight decay is 0.01, dropout is 0.1
        optimizer.param_groups[0]['weight_decay'] = max(0.01, optimizer.param_groups[0]['weight_decay'] / wd_dynamic_reg_factor)
        for name, module in model.named_modules():
            if 'dropout' in name.lower():
                module.p = max(0.1, module.p - dropout_dynamic_reg_factor)

        
def finetune(args):

    # --- setting device ----
    device = torch.device("cuda")

    # --- setting logging ---
    # os.makedirs("finetuning_logs", exist_ok=True)    

    if args.wandb:
        wandb.init(entity=args.wandb_entity, project=args.wandb_project, group=args.wandb_group)
        if args.wandb_run_name is not None:
            wandb.run.name = args.wandb_run_name
            # logger.add("finetuning_logs/" + args.wandb_run_name + ".log", level="INFO")
        else:
            # logger.add("finetuning_logs/" + datetime.now().strftime("%d-%m-%Y_%H.%M.%S") + ".log", level="INFO")
            pass

    # --- fixing seeds ---
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed) 

    # --- resume or start fine-tuning from scratch ---
    model_name = os.path.basename(args.checkpoint_path)
    logger.info(f"Loading checkpoint of {model_name} for {'resuming' if args.resume else ''} finetuning")

    if args.resume:

        checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False) # partially fine-tuned model checkpoint
        model_config = checkpoint['model_config']
        model = MODEL_NAME_2_FINETUNING_MODEL[args.encoder_name](model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        global_step = checkpoint['global_step']
        best_val_loss = checkpoint['best_val_loss']
        patience_count = checkpoint['patience_count']
        best_target_score = checkpoint['best_target_score']

        if args.lin_eval:
            # redundant stuff
            model.config.layerdrop = 0.0
            model.config.edge_drop_prob = 0.0
            # model.config.dropout = 0.0 # ! add in the end
            model.embedder.requires_grad_(False)
            if hasattr(model, 'lead_encoding'):
                model.lead_encoding.requires_grad_(False)
            if hasattr(model, 'temporal_encoding'):
                model.temporal_encoding.requires_grad_(False)
            model.encoder.requires_grad_(False)
            # model.norm_post_encoder.requires_grad_(False)

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), # we don't plan to unfreeze parameters in the middle of training
            lr=args.lr, 
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=args.weight_decay
        )
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=math.ceil(args.warm_up_perc * args.training_steps),
            num_training_steps=args.training_steps,
        )        

        lr_scheduler.last_epoch = checkpoint['lr_scheduler_state_dict']['last_epoch']-1
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

        logger.info("Ready to resume fine-tuning")

    else:

        with open(args.model_config_path, 'rb') as f:
            model_config = json.load(f)
        model_config = MODEL_NAME_2_CONFIG[args.encoder_name](**model_config)

        checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False) # pretrained model checkpoint

        model = MODEL_NAME_2_FINETUNING_MODEL[args.encoder_name](model_config)
        # model.load_state_dict(checkpoint['model_state_dict'], strict=True) # strict=False allows to ignore masked-node-modelling blocks and parameters and to leave intact output classifier modules (desirable)
        
        # Get the state dict from the checkpoint
        checkpoint_state_dict = checkpoint['model_state_dict']

        # Define keys to explicitly ignore when loading
        ignore_keys = ["mask_embed", "classifiers.0.weight", "classifiers.0.bias"]

        # Filter the state dict to exclude ignored keys
        filtered_state_dict = {k: v for k, v in checkpoint_state_dict.items() 
                              if not any(ignore_key in k for ignore_key in ignore_keys)}

        # Load the filtered state dict
        result = model.load_state_dict(filtered_state_dict, strict=False)
        logger.info(f"Missing keys in state_dict: {result.missing_keys}")
        logger.info(f"Unexpected keys in state_dict: {result.unexpected_keys}")
        model.to(device)

        if args.lin_eval:
            # model.config.dropout = 0.0 # ! add in the end
            model.config.layerdrop = 0.0
            model.config.edge_drop_prob = 0.0
            
            model.embedder.requires_grad_(False)
            if hasattr(model, 'lead_encoding'):
                model.lead_encoding.requires_grad_(False)
            if hasattr(model, 'temporal_encoding'):
                model.temporal_encoding.requires_grad_(False)
            model.encoder.requires_grad_(False)
            # model.norm_post_encoder.requires_grad_(False) # ! add in the end
            
        # reset dropout layers
        for name, module in model.named_modules():
            if 'dropout' in name.lower():
                module.p = model_config.dropout
        
        # reset layerdrop
        model.encoder.layerdrop = model_config.layerdrop

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), # we don't plan to unfreeze parameters in the middle of training
            lr=args.lr, 
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=args.weight_decay
        )

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=math.ceil(args.warm_up_perc * args.training_steps),
            num_training_steps=args.training_steps,
        )

        global_step = 0
        best_val_loss = float('inf')
        patience_count = 0
        best_target_score = 0

        logger.info("Ready to start fine-tuning")

    # --- ready to resume/start fine-tuning ---
    
    # --- log model config ---
    logger.info(f"Model configuration: {model.config}")

    # ---- log number of params ---
    logger.info(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # --- creation of datasets and dataloaders ---
    
    if args.use_benchmarking_format:
        common_args = {
            "use_benchmarking_format" : True,
            "benchmarking_folder" : args.benchmarking_folder,
            "label_column" : args.label_column,
            "downsampling_factor" : args.downsampling_factor,
            "nodes_per_lead" : model_config.nodes_per_lead,
            "pretrain" : False,
            "create_edge_index" : args.create_edge_index,
            "edge_strategy" : args.edge_strategy,
            "sampling" : args.sampling,
            "how_many_leads" : args.how_many_leads,
            "use_temporal_encoding" : model_config.use_temporal_encoding,
            "use_lead_encoding" : model_config.use_lead_encoding,
            "pad_unsampled_leads" : args.pad_unsampled_leads
        }
        
        full_train_set = ECGDataset(
            **common_args, 
            random_crop=True
        )
        
        full_val_set = ECGDataset(
            **common_args, 
            random_crop=False,
            fixed_lead_seed=args.val_lead_seed
        )
        
        train_idx, val_idx, _ = full_train_set.get_splits()
        
        train_set = torch.utils.data.Subset(full_train_set, train_idx)
        val_set = torch.utils.data.Subset(full_val_set, val_idx)
        
        # train_loader = ...
        
        val_loader = DataLoader(
            dataset=val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            prefetch_factor=4,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=val_set.dataset.collate_fn,
            worker_init_fn=worker_init_fn
        )
        
        train_weights = train_set.dataset.weights
        val_weights = val_set.dataset.weights
    else:

        train_set = ECGDataset(
            path_to_dataset_csv=args.path_to_train_dataset_csv,
            ecg_dir_path=args.ecg_train_dir_path, 
            features_path=None,
            downsampling_factor=args.downsampling_factor,
            label_start_index=args.label_start_index,
            nodes_per_lead=model_config.nodes_per_lead,
            pretrain=False,
            random_crop=True,
            return_full_length=False,
            create_edge_index=args.create_edge_index,
            edge_strategy=args.edge_strategy,
            sampling=args.sampling,
            how_many_leads=args.how_many_leads,
            use_temporal_encoding=model_config.use_temporal_encoding,
            use_lead_encoding=model_config.use_lead_encoding,
            pad_unsampled_leads=args.pad_unsampled_leads
        )

        val_set = ECGDataset(
            path_to_dataset_csv=args.path_to_val_dataset_csv,
            ecg_dir_path=args.ecg_val_dir_path, 
            features_path=None,
            downsampling_factor=args.downsampling_factor,
            label_start_index=args.label_start_index,
            nodes_per_lead=model_config.nodes_per_lead,
            pretrain=False,
            random_crop=False,
            return_full_length=False,
            create_edge_index=args.create_edge_index,
            edge_strategy=args.edge_strategy,
            sampling=args.sampling,
            how_many_leads=args.how_many_leads,
            use_temporal_encoding=model_config.use_temporal_encoding,
            use_lead_encoding=model_config.use_lead_encoding,
            fixed_lead_seed=args.val_lead_seed,
            pad_unsampled_leads=args.pad_unsampled_leads
        )

        # train_loader = DataLoader(
        #     dataset=train_set,
        #     batch_size=args.batch_size,
        #     shuffle=True,
        #     num_workers=args.num_workers,
        #     prefetch_factor=4,
        #     persistent_workers=True,
        #     pin_memory=True,
        #     collate_fn=train_set.collate_fn
        # )

        val_loader = DataLoader(
            dataset=val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            prefetch_factor=4,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=val_set.collate_fn,
            worker_init_fn=worker_init_fn
        )
        
        train_weights = train_set.weights
        val_weights = val_set.weights

    # ---- starting training loop ---

    epochs = math.ceil(args.training_steps / math.ceil(len(train_set) / (args.batch_size * args.accumulation_steps)))

    rng_generator = torch.Generator() 
    
    start_epoch = global_step // (len(train_set) // args.batch_size)

    patience = args.patience if args.patience is not None else args.training_steps // args.val_frequency

    scaler = amp.GradScaler()

    train_criterion, val_criterion = get_criterions(args.task, train_weigths=train_weights, val_weights=val_weights, label_smoothing=args.label_smoothing)
    train_criterion.to(device)
    val_criterion.to(device)

    train_metrics = get_metrics(args.task, num_classes=model_config.num_classes)
    val_metrics = get_metrics(args.task, num_classes=model_config.num_classes)

    for train_metric, val_metric in zip(train_metrics.values(), val_metrics.values()):
        train_metric.to(device)
        val_metric.to(device)
        
    train_sampler = ResumableRandomSampler(
        train_set,
        batch_size=args.batch_size,
        start_idx=0,
        generator=rng_generator
    )

    collate = train_set.dataset.collate_fn if args.use_benchmarking_format else train_set.collate_fn

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False,
        pin_memory=True,
        collate_fn=collate,
        worker_init_fn=worker_init_fn
    )
            
    for epoch in range(start_epoch, epochs):
        
        if global_step >= args.training_steps:
            break

        model.train()
        if args.lin_eval:
            # redundant stuff
            model.embedder.requires_grad_(False)
            model.encoder.requires_grad_(False)
            if hasattr(model, 'lead_encoding'):
                model.lead_encoding.requires_grad_(False)
            if hasattr(model, 'temporal_encoding'):
                model.temporal_encoding.requires_grad_(False)
            # model.norm_post_encoder.requires_grad_(False)
            
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        rng_generator.manual_seed(epoch + args.seed)
        
        if epoch == start_epoch:
            batches_in_epoch = len(train_set) // args.batch_size 
            current_epoch_step = global_step % batches_in_epoch
            samples_to_skip = current_epoch_step * args.batch_size
        else:
            samples_to_skip = 0
            
        train_sampler.set_epoch(epoch, start_idx=samples_to_skip)
        
        # train_sampler = ResumableRandomSampler(
        #     train_set, 
        #     batch_size=args.batch_size,
        #     start_idx=samples_to_skip,
        #     generator=rng_generator
        # )
        
        # if args.use_benchmarking_format:
        #     train_loader = DataLoader(
        #         dataset=train_set,
        #         batch_size=args.batch_size,
        #         sampler=train_sampler,
        #         shuffle=False, # shuffle is managed by the sampler
        #         num_workers=args.num_workers,
        #         prefetch_factor=4 if args.num_workers > 0 else None,
        #         persistent_workers=True if args.num_workers > 0 else False,
        #         pin_memory=True,
        #         collate_fn=train_set.dataset.collate_fn,
        #         worker_init_fn=worker_init_fn
        #     )
        # else:
        #     train_loader = DataLoader(
        #         dataset=train_set,
        #         batch_size=args.batch_size,
        #         sampler=train_sampler,
        #         shuffle=False, # shuffle is managed by the sampler
        #         num_workers=args.num_workers,
        #         prefetch_factor=4 if args.num_workers > 0 else None,
        #         persistent_workers=True if args.num_workers > 0 else False,
        #         pin_memory=True,
        #         collate_fn=train_set.collate_fn,
        #         worker_init_fn=worker_init_fn
        #     )

        train_losses = []
        for metric in train_metrics.values():
            metric.reset()
            
        # start_batch = global_step % len(train_loader)

        # for batch in tqdm(
        #                 islice(train_loader, start_batch, len(train_loader)),
        #                 initial=global_step,
        #                 total=len(train_loader) - start_batch
        #                 ):
        
        for batch in tqdm(train_loader, initial=global_step, total=len(train_loader)):
            
            global_step += 1

            # --- unpack ---

            x = batch.x
            edge_index = batch.edge_index
            labels = batch.y 
            node_lead_ids = batch.node_lead_ids
            node_temporal_pos = batch.node_temporal_pos
            b = batch.batch

            # --- move to cuda ---
            x = x.to(device)
            edge_index = edge_index.to(device)
            labels = labels.to(device)
            node_lead_ids = node_lead_ids.to(device)
            node_temporal_pos = node_temporal_pos.to(device)
            b = b.to(device)

            # with torch.autocast(
            #     'cuda',
            #     enabled=True,
            #     dtype=torch.bfloat16
            #     ):
                # NOTE: this generic forward surely works for gat- and gin-based GNN.
            output = model(
                x=x,
                edge_index=edge_index,
                criterion=train_criterion,
                labels=labels,
                node_lead_ids=node_lead_ids,
                node_temporal_pos=node_temporal_pos,
                batch=b,
                batch_size=args.batch_size,
                output_hidden_states=False,
                return_dict=True,
                metrics=train_metrics
            )
            
            loss = output['loss'] / args.accumulation_steps
            scaler.scale(loss).backward()
            train_losses.append(loss.item())

            # --- gradient accumulation ---

            if global_step % args.accumulation_steps == 0 or global_step == args.training_steps:
                scaler.step(optimizer)
                lr_scheduler.step()
                scaler.update()
                optimizer.zero_grad()

            # --- validation and logging ---

            if global_step % args.val_frequency == 0 or global_step == args.training_steps:

                model.eval()

                # --- clear metrics containers ---

                val_losses = []
                target_score = 0
                for _, metric in val_metrics.items():
                    metric.reset()

                logger.info(f"Validating the model at step {global_step}")

                # --- validation loop ---
                
                val_losses = validate(args, device, model, val_loader, val_criterion, val_metrics, val_losses)

                # --- end of validation loop ---

                val_loss = np.mean(val_losses)
                train_loss = np.mean(train_losses)
                train_losses.clear() # to keep train and val logs aligned

                # --- logging ---
                logger.info(f"Training loss = {train_loss}")
                logger.info(f"Validation loss = {val_loss}")
                to_log = {
                    "Training_loss" : train_loss,
                    "Validation_loss" : val_loss
                }

                for name, metric in train_metrics.items():
                    score = metric.compute()
                    macro = score.mean()
                    logger.info(f"Training {name} (macro) = {macro}")
                    to_log.update({f"Training_{name}" : macro})
                    metric.reset() # to keep train and loss metrics aligned

                for name, metric in val_metrics.items():
                    score = metric.compute()
                    macro = score.mean()
                    logger.info(f"Validation {name} (macro) = {macro}")
                    to_log.update({f"Validation_{name}" : macro})
                    if name == args.target_metric:
                        target_score = macro

                wandb.log(to_log)
                # --- end of logging ---

                # --- checkpoint if either loss or target_metric improved by a minimum delta ---

                if val_loss <= best_val_loss - MINIMAL_IMPROVEMENT_FINETUNING:

                    logger.info(f"Validation loss improved from {best_val_loss} to {val_loss}")

                    best_val_loss = val_loss
                    best_target_score = target_score if target_score > best_target_score + MINIMAL_IMPROVEMENT_FINETUNING else best_target_score 

                    logger.info(f"Current best {args.target_metric} = {best_target_score}")

                    patience_count = 0

                    checkpoint = {
                        "global_step" : global_step,
                        "best_val_loss" : best_val_loss,
                        "model_config" : model.config,
                        "model_state_dict" : copy.deepcopy(model.state_dict()),
                        "optimizer_state_dict" : copy.deepcopy(optimizer.state_dict()),
                        "lr_scheduler_state_dict" : copy.deepcopy(lr_scheduler.state_dict()),
                        "patience_count" : patience_count,
                        "best_target_score" : best_target_score
                    }
                    # checkpoint_name = f"{args.encoder_name}_{global_step}_{wandb.run.name}.pt"
                    checkpoint_name = f"{args.encoder_name}_best_val_loss_{wandb.run.name}.pt"
                    torch.save(checkpoint, os.path.join(args.finetuned_models_path, checkpoint_name))

                    dynamic_regularizer(
                        optimizer,
                        model,
                        penalty=False,
                        wd_dynamic_reg_factor=args.wd_dynamic_reg_factor,
                        dropout_dynamic_reg_factor=args.dropout_dynamic_reg_factor
                    ) if args.dynamic_reg and not args.lin_eval else None

                elif target_score >= best_target_score + MINIMAL_IMPROVEMENT_FINETUNING:

                    logger.info(f"Validation {args.target_metric} improved from {best_target_score} to {target_score}")
                    logger.info(f"Current best validation loss = {best_val_loss}")

                    best_target_score = target_score

                    checkpoint = {
                        "global_step" : global_step,
                        "best_val_loss" : best_val_loss,
                        "model_config" : model.config,
                        "model_state_dict" : copy.deepcopy(model.state_dict()),
                        "optimizer_state_dict" : copy.deepcopy(optimizer.state_dict()),
                        "lr_scheduler_state_dict" : copy.deepcopy(lr_scheduler.state_dict()),
                        "patience_count" : patience_count,
                        "best_target_score" : best_target_score
                    }

                    # checkpoint_name = f"{args.encoder_name}_{global_step}_{wandb.run.name}.pt"
                    checkpoint_name = f"{args.encoder_name}_best_val_{args.target_metric}_{wandb.run.name}.pt" 
                    torch.save(checkpoint, os.path.join(args.finetuned_models_path, checkpoint_name))

                    dynamic_regularizer(
                        optimizer,
                        model,
                        penalty=False,
                        wd_dynamic_reg_factor=args.wd_dynamic_reg_factor,
                        dropout_dynamic_reg_factor=args.dropout_dynamic_reg_factor
                    ) if args.dynamic_reg and not args.lin_eval else None

                else:
                    
                    logger.info(f"No improvement in both validation loss and {args.target_metric}")

                    patience_count += 1

                    if args.dynamic_reg and not args.lin_eval and patience_count % (patience // args.validations_for_penalty) == 0 and patience_count != patience:
                        dynamic_regularizer(
                            optimizer,
                            model,
                            penalty=True,
                            wd_dynamic_reg_factor=args.wd_dynamic_reg_factor,
                            dropout_dynamic_reg_factor=args.dropout_dynamic_reg_factor
                            )

                    if patience_count == patience:
                        logger.info(f"Early stopping reached at {global_step}")
                        wandb.log({"patience_count" : patience_count})
                        return
                    
                # --- end of checkpointing and early stopping ---

                model.train()
                
            if global_step >= args.training_steps:
                break

    # --- end of training loop ---

    logger.info(f"End of training")
    logger.info(f"Total steps = {global_step}")
    logger.info(f"Best validation loss = {best_val_loss}")
    logger.info(f"Best Validation {args.target_metric} = {best_target_score}")
    wandb.finish()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Finetuning script")

    parser.add_argument(
        "--wandb",
        help="Whether or not use wandb. Default False",
        action="store_true"
    )

    parser.add_argument(
        "--wandb_entity",
        help="Name of wandb entity. Default team name",
        type=str,
        default=None
    )

    parser.add_argument(
        "--wandb_project",
        help="Name of wandb project. Default your project name",
        type=str,
        default="RELAY"
    )

    parser.add_argument(
        "--wandb_group",
        help="Name of wandb group. Defatult finetuning",
        type=str,
        default="finetuning"
    )

    parser.add_argument(
        "--wandb_run_name",
        help="Name of wandb run name. If None, wandb sets it automatically",
        type=str,
        default=None
    )
    
    parser.add_argument(
        "--label_start_index",
        help="Index of the first column in the .csv file containing labels. Default 3",
        type=int,
        default=3
    )

    parser.add_argument(
        "--resume",
        help="Whether to resume fine-tuning loading a checkpoint. Default False",
        action="store_true"
    )

    parser.add_argument(
        "--checkpoint_path",
        help="Path to a pretrained or partially finetuned model checkpoint",
        type=str,
    )

    parser.add_argument(
        "--encoder_name",
        help="Name of the backbone to use. Default gat",
        choices=['gin', 'gat', 'mlp', 'graph_transformer'],
        type=str,
        default='gat'
    )

    parser.add_argument(
        "--lin_eval", 
        action="store_true", 
        help="Whether to perform a linear evaluation of frozen representation. Default False"
    )

    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate. Default=1e-5",
        default=1e-3
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        help="Weight decay. Default 1e-2",
        default=0.01
    )

    parser.add_argument(
        "--warm_up_perc",
        type=float,
        help="Percentage of training steps for lr warm-up. Default=0.08",
        default=0.08
    )

    parser.add_argument(
        "--training_steps",
        type=int,
        help="Number of training steps to perform. Default 50K",
        default=50000
    )

    parser.add_argument(
        "--model_config_path",
        type=str,
        help="Path to JSON file containing model configuration to start pre-training. Default None",
        default=None
    )

    parser.add_argument(
        "--path_to_train_dataset_csv",
        type=str,
        help="Path to the .csv file referencing the training data if not using the benchmarking format",
        default=None
    )

    parser.add_argument(
        "--ecg_train_dir_path",
        type=str,
        help="Path to the directory containing the actual training data if not using the benchmarking format",
        default=None
    )

    parser.add_argument(
        "--downsampling_factor",
        help="Downsampling factor of the ECG signal. Defeault 5",
        type=int,
        default=5
    )

    parser.add_argument(
        "--create_edge_index",
        action="store_true",
    )

    parser.add_argument(
        "--edge_strategy", 
        type=str,
        choices=["spatiotemporal", "fully_connected"],
        help="Topology to apply to the graph, if create_edge_index is set. Default spatiotemporal",
        default="spatiotemporal"
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
        help="Seed of the head initialisation, data order and training lead draws. Default 42",
        default=42
    )

    parser.add_argument(
        "--pad_unsampled_leads",
        action="store_true",
        help="Keep the leads not selected by --sampling/--how_many_leads as all-zero signals (full 12-lead "
             "graph) instead of dropping them, in training and validation. Default False"
    )

    parser.add_argument(
        "--val_lead_seed",
        type=int,
        help="If set (and sampling='random'), validation leads are drawn once per record from this seed, " \
        "so every validation and every run sees the same lead subsets. Default None (redrawn at each validation)",
        default=None
    )

    parser.add_argument(
        "--path_to_val_dataset_csv",
        type=str,
        help="Path to the .csv file referencing the validation data if not using the benchmarking format",
        default=None
    )

    parser.add_argument(
        "--ecg_val_dir_path",
        type=str,
        help="Path to the directory containing the actual validation data if not using the benchmarking format",
        default=None
    )

    parser.add_argument(
        "--sampling", 
        help="Either 'random' or a list of strings with lead names",
        nargs='+'
    )

    parser.add_argument(
        "--batch_size",
        type=int, 
        default=64, 
        help="Batch size. Default 64"
    )

    parser.add_argument(
        "--num_workers", 
        type=int,
        help="Number of workers in the dataloaders. Default 6",
        default=6
    )

    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=1,
        help="Number of gradients to accumulate. Default 1"
    )

    parser.add_argument(
        "--patience", 
        help="Patience: number of validations with no improvements to wait before stopping the training",
        default=10,
        type=int
    )

    parser.add_argument(
        "--task", 
        type=str,
        help="The type of task to perform. Default multi_label.",
        choices=['multi_label', 'multi_class', 'regression'],
        default='multi_label'
    )

    parser.add_argument(
        "--target_metric",
        help="Target metric for checkpointing. Default auroc", 
        type=str, 
        choices=["f1-score", "recall", "specificity", "precision", "auroc", "auprc"],
        default='auroc'
    )

    parser.add_argument(
        "--val_frequency",
        help="Frequency of validation expressed in terms of steps. Default 5000",
        default=5000,
        type=int
    )

    parser.add_argument(
        "--finetuned_models_path", 
        help="Path to the directory to save the checkpoints. Default ./",
        type=str,
        default="."
    )

    parser.add_argument(
        "--dynamic_reg",
        help="Whether or not use the dynamic regularization. Default False",
        action="store_true",
        
    )

    parser.add_argument(
        "--validations_for_penalty",
        help="Number of validations with no improvement in both the val. loss and accuracy " \
        "to wait before penalising the model with the dynamic regularization. Default 4",
        type=int,
        default=1
    )
    
    parser.add_argument(
        "--wd_dynamic_reg_factor",
        help="Weight decay dynamic regularization factor. Default 5",
        type=float,
        default=WEIGHT_DECAY_DYNAMIC_REG_FACTOR
    )
    
    parser.add_argument(
        "--dropout_dynamic_reg_factor",
        help="Dropout dynamic regularization factor. Default 0.05",
        type=float,
        default=DROPOUT_DYNAMIC_REG_FACTOR
    )

    parser.add_argument(
        "--label_smoothing",
        help="Label smoothing to use in the cross entropy loss in case of task=multi_class. Default 0.0",
        type=float,
        default=0.0
    )
    
    parser.add_argument(
        "--use_benchmarking_format",
        help="Whether or not the dataset is in the benchmarking format. Default False",
        action="store_true"
    )
    
    parser.add_argument(
        "--benchmarking_folder",
        help="Path to the folder containing the benchmarking dataset. Required if use_benchmarking_format is set",
        type=str,
        default=None
    )
    
    parser.add_argument(
        "--label_column",
        help="Name of the column containing labels in the benchmarking dataset. Required if use_benchmarking_format is set",
        type=str,
        choices=['label_filtered', 'label_primary_filtered', 'label_all_filtered', 'label_diag_superclass_filtered', 'label_diag_subclass_filtered', 'dummy'],
        default='label_filtered'
    )

    args = parser.parse_args()

    # --- input check ---

    if not args.resume:
        if args.model_config_path is None or not os.path.exists(args.model_config_path):
            raise ValueError("If not resuming fine-tuning, then argument model_config_path must be passed and valid")

    if args.warm_up_perc > 1.0 or args.warm_up_perc < 0.0:
        raise ValueError(f"Argument wamr_up_perc must be in [0,1]. Got {args.warm_up_perc}")
    
    if args.patience % args.validations_for_penalty != 0:
        raise ValueError(f"Argument patience must be multiple of validations_for_penalty. Got patience={args.patience} and validations_for_penalty={args.validations_for_penalty}")

    if args.use_benchmarking_format:
        if args.benchmarking_folder is None or not os.path.exists(args.benchmarking_folder):
            raise ValueError("Argument benchmarking_folder must be passed and valid if use_benchmarking_format is set")
        if args.label_column is None:
            raise ValueError("Argument label_column must be passed if use_benchmarking_format is set")
        
    # --- check existence of all required paths ---
    required_paths = [
        ("checkpoint_path", args.checkpoint_path),
        ("path_to_train_dataset_csv", args.path_to_train_dataset_csv),
        ("ecg_train_dir_path", args.ecg_train_dir_path),
        ("path_to_val_dataset_csv", args.path_to_val_dataset_csv),
        ("ecg_val_dir_path", args.ecg_val_dir_path),
        ("finetuned_models_path", args.finetuned_models_path),
        ("benchmarking_folder", args.benchmarking_folder)
    ]

    for name, path in required_paths:
        if path is not None and not os.path.exists(path):
            raise ValueError(f"Path argument '{name}' does not exist. Got {path}")
        
    if args.sampling == ['random']:
        args.sampling = args.sampling[0] # now it's a string
    else:
        # a fixed lead set, possibly a single lead (e.g. --sampling I); unknown names would otherwise
        # fall back silently to all leads in the dataset
        from consts import LEADS
        unknown = [lead for lead in args.sampling if lead not in LEADS]
        if unknown:
            raise ValueError(f"Argument sampling must be either 'random' or a list of valid lead names {LEADS}. Got {args.sampling}")
        args.sampling = list(dict.fromkeys(args.sampling)) # to remove potential duplicates


    # --- end of input check ---

    print("Inserted arguments: ")
    for arg in vars(args):
        print(f"{arg} = {getattr(args, arg)}")

    finetune(args)






    
