import os
# Force thread limits before importing torch/numpy
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
import numpy as np

# Force PyTorch intra-op parallelism to 1
torch.set_num_threads(1)

from itertools import islice
from torchmetrics.classification import MulticlassAccuracy
from hubert_ecg.hubert_ecg import HuBERTECGConfig
import json
import torch.nn.functional as F
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
import random
import torch.optim as optim
from consts import MINIMAL_IMPROVEMENT_FINETUNING, WEIGHT_DECAY_DYNAMIC_REG_FACTOR, DROPOUT_DYNAMIC_REG_FACTOR, MINIMAL_IMPROVEMENT_PRETRAINING
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from utils import retrieve_kmeans_models, ResumableRandomSampler
import math
import threadpoolctl

def worker_init_fn(worker_id):
    torch.set_num_threads(1)
    threadpoolctl.threadpool_limits(limits=1)
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def validate(model, val_loader, device, criterion, val_metrics, val_losses, centroids):
    
    model.eval()
    
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
            
            # start = perf_counter()
            # torch.cuda.synchronize()
            
            x = x.to(device)
            edge_index = edge_index.to(device)
            labels = labels.to(device)
            node_lead_ids = node_lead_ids.to(device)
            node_temporal_pos = node_temporal_pos.to(device)
            b = b.to(device)
            
            with torch.no_grad():
                # newly added: compute equivalent of kmeans predict on labels, which are actually features
                # labels are L*S, D, centroids are list of length num_tasks, each element is tensor of shape (num_clusters, D)
                labels = torch.stack([
                    torch.argmin(
                        torch.cdist(
                            labels,
                            centroid.unsqueeze(0) # 1, num_clusters, D
                        ).squeeze(0), # num_clusters, D -> num_clusters
                        dim=1
                    ) for centroid in centroids
                ], dim=1) # num_tasks, L*S -> L*S, num_tasks
            
            # torch.cuda.synchronize()
            # elapsed = perf_counter() - start 
            # logger.info(f"Time to map data to device during validation:{perf_counter()-start}")

            # with torch.autocast(
            #     "cuda",
            #     enabled=True,
            #     dtype=torch.bfloat16
            #     # dtype=torch.float32
            #     ):
                
                # start = perf_counter()
                # torch.cuda.synchronize()

            output = model(
                        x=x,
                        edge_index=edge_index,
                        criterion=criterion,
                        labels=labels,
                        node_lead_ids=node_lead_ids,
                        node_temporal_pos=node_temporal_pos,
                        batch=b,
                        batch_size=args.batch_size,
                        output_hidden_states=False,
                        return_dict=True,
                        metrics=val_metrics,
                        real_node_mask=batch.real_node_mask.to(device) if args.pad_unsampled_leads else None
                    )
                

                # torch.cuda.synchronize()
                # elapsed_model_forward_val = perf_counter() - start
                # time_to_log["elapsed_model_forward_val"] = elapsed_model_forward_val  
                # time_running_average_model_forward_val = (time_running_average_model_forward_val + elapsed_model_forward_val) / 2 
                # time_to_log["running_avg_model_forward_val"] = time_running_average_model_forward_val
                                            
            loss = output['loss']
            val_losses.append(loss.item())
            
    return val_losses #, time_to_log


def dynamic_regularizer(optimizer, model, penalty, wd_dynamic_reg_factor=WEIGHT_DECAY_DYNAMIC_REG_FACTOR, dropout_dynamic_reg_factor=DROPOUT_DYNAMIC_REG_FACTOR):
    """
    Applies a dynamic regularization to the model.
    If penalty is True, increases weight decay and dropout. If False, decreases them.
    0.01 is the minimum attainable weight decay, 0.1 is the minimum attainable dropout.
    ASSUMPTION: model dropout layers are named with 'dropout' in their name and are instances of torch.nn.Dropout
    """
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

def pretrain(args):

    # --- setting device ----
    device = torch.device("cuda")

    # --- setting logging ---
    os.makedirs("pretraining_logs", exist_ok=True)    

    if args.wandb:
        wandb.init(entity=args.wandb_entity, project=args.wandb_project, group=args.wandb_group)
        if args.wandb_run_name is not None:
            wandb.run.name = args.wandb_run_name

    # --- fixing seeds ---
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # --- resume or start pre-training from scratch ---
    if args.resume:
        
        model_name = os.path.basename(args.checkpoint_path)
        logger.info(f"Loading checkpoint of {model_name} to resume pre-training...")
        
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)
        model_config = MODEL_NAME_2_CONFIG[args.encoder_name](**checkpoint['model_config'].to_dict())
        model = MODEL_NAME_2_PRETRAINING_MODEL[args.encoder_name](model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        global_step = checkpoint['global_step']
        best_val_loss = checkpoint['best_val_loss']
        patience_count = checkpoint['patience_count']
        best_target_score = checkpoint['best_target_score']

        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr, 
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=args.weight_decay
        )
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # THIS IS PROBABLY NOT NECESSARY
        # optimizer.param_groups[0]['weight_decay'] = max(0.01, optimizer.param_groups[0]['weight_decay'])
        # for name, module in model.named_modules():
        #     if 'dropout' in name:
        #         module.p = max(0.1, module.p)


        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=math.ceil(args.warm_up_perc * args.training_steps),
            num_training_steps=args.training_steps,
            last_epoch=checkpoint['lr_scheduler_state_dict']['last_epoch']-1
        )
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

        logger.info("Checkpoint loaded.")
        
    elif args.change_iteration:
        
        model_name = os.path.basename(args.checkpoint_path)
        logger.info(f"Loading checkpoint of {model_name} to changing pre-training iteration...")
        
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)
                
        with open(args.model_config_path, 'rb') as f:
            model_config = json.load(f)
        model_config = MODEL_NAME_2_CONFIG[args.encoder_name](**model_config)
        
        model = MODEL_NAME_2_PRETRAINING_MODEL[args.encoder_name](model_config)
        
        # filter out pre-training heads from checkpoint because they have different sizes when changing iteration        
        filtered_model_state_dict = {k : v for k, v in checkpoint['model_state_dict'].items()
                                     if not k.startswith('classifiers') and 
                                     not k.startswith("pseudo")}
        
        # add randomly initialized pre-training heads to the filtered state dict
        filtered_model_state_dict.update({k : v for k, v in model.state_dict().items()
                                            if k.startswith('classifiers') or 
                                            k.startswith("pseudo")})
        
        model.load_state_dict(filtered_model_state_dict) 
        model.to(device)

        global_step = 0
        best_val_loss = float('inf')
        patience_count = 0
        best_target_score = 0.0

        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr, 
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=args.weight_decay
        )

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=math.ceil(args.warm_up_perc * args.training_steps),
            num_training_steps=args.training_steps,
            # last_epoch=checkpoint['lr_scheduler_state_dict']['last_epoch']-1
        )

        logger.info("Checkpoint loaded.")        

    else:

        logger.info(f"Preparing to pre-training an ECG {args.encoder_name} from scratch...")

        with open(args.model_config_path, 'rb') as f:
            model_config = json.load(f)
        model_config = MODEL_NAME_2_CONFIG[args.encoder_name](**model_config)

        model = MODEL_NAME_2_PRETRAINING_MODEL[args.encoder_name](model_config)
        model.to(device)

        global_step = 0
        best_val_loss = float('inf')
        patience_count = 0
        best_target_score = 0.0

        optimizer = optim.AdamW(
            model.parameters(),
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

        logger.info("Ready to start pre-training.")

    # --- log number of params: nice thing to have ---
    logger.info(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")
    
    logger.warning("DEBUG MODE: Watching the model")
    wandb.watch(model, log="all", log_freq=args.val_frequency)
    
    # print model config
    logger.info(f"Model configuration")
    print(model.config)
    

    if args.gradient_clipping is not None:
        logger.info("Gradient clipping enabled")
        import torch.nn.utils as nn_utils
        nn_utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clipping)
        
    # --- creation of datasets and data loaders ---

    train_set = ECGDataset(
        path_to_dataset_csv=args.path_to_train_dataset_csv,
        ecg_dir_path=args.ecg_train_dir_path, 
        features_path=args.train_features_path,
        downsampling_factor=args.downsampling_factor,
        nodes_per_lead=model_config.nodes_per_lead,
        label_start_index=None,
        pretrain=True,
        random_crop=False,
        return_full_length=False,
        create_edge_index=args.create_edge_index,
        edge_strategy=args.edge_strategy,
        sampling='random',
        how_many_leads=args.train_how_many_leads, # None -> random lead sampling, L ~ U{1..12}
        use_lead_encoding=model_config.use_lead_encoding,
        use_temporal_encoding=model_config.use_temporal_encoding,
        pad_unsampled_leads=args.pad_unsampled_leads
    )

    val_set = ECGDataset(
        path_to_dataset_csv=args.path_to_val_dataset_csv,
        ecg_dir_path=args.ecg_val_dir_path, 
        features_path=args.val_features_path,
        downsampling_factor=args.downsampling_factor,
        nodes_per_lead=model_config.nodes_per_lead,
        label_start_index=None,
        pretrain=True,
        random_crop=False,
        return_full_length=False,
        create_edge_index=args.create_edge_index,
        edge_strategy=args.edge_strategy,
        sampling='random',
        how_many_leads=None,
        use_lead_encoding=model_config.use_lead_encoding,
        use_temporal_encoding=model_config.use_temporal_encoding,
        pad_unsampled_leads=args.pad_unsampled_leads
    )
    
    # newly added
    kmeans_models = retrieve_kmeans_models(args.kmeans_path)
    centroids = [torch.tensor(kmeans.cluster_centers_).float().to(device) for kmeans in kmeans_models]

    # train_loader = DataLoader(
    #     dataset=train_set,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.num_workers,
    #     prefetch_factor=4,
    #     persistent_workers=True,
    #     pin_memory=True,
    #     collate_fn=train_set.collate_fn,
    #     drop_last=True
    # )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False,
        pin_memory=True,
        collate_fn=val_set.collate_fn,
        worker_init_fn=worker_init_fn
    )
    
    # --- checks ---
    # assert len(train_set.ensemble_kmeans) == len(model_config.pseudo_vocab_sizes),  \
    #     f"Number of kmeans models must be equal to the number of pretext tasks. Got {len(train_set.ensemble_kmeans)} and {len(model_config.pseudo_vocab_sizes)} respectively"
        
    # for kmeans, vocab_size in zip(train_set.ensemble_kmeans, model_config.pseudo_vocab_sizes):
    #     assert kmeans.n_clusters == vocab_size, \
    #         f"Number of clusters of each kmeans model must be equal to the corresponding pseudo vocab size. Got {kmeans.n_clusters} and {vocab_size} respectively."

    # --- starting training loop ---

    # epochs = math.ceil(args.training_steps / (len(train_loader) // args.accumulation_steps))
    epochs = math.ceil(args.training_steps / ((len(train_set) // args.batch_size) // args.accumulation_steps))

    rng_generator = torch.Generator()
    
    # start_epoch = global_step // len(train_loader)
    start_epoch = global_step // (len(train_set) // args.batch_size)

    patience = args.patience if args.patience is not None else args.training_steps // args.val_frequency

    scaler = amp.GradScaler()

    train_criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    val_criterion = torch.nn.CrossEntropyLoss().to(device)
    
    train_metrics = {}
    for i in range(len(model_config.pseudo_vocab_sizes)):
        train_metrics[f"Train_Accuracy_task_{i}"] = MulticlassAccuracy(num_classes=model_config.pseudo_vocab_sizes[i], average=None)
        # train_metrics[f"Train_Top5_Accuracy_task_{i}"] = MulticlassAccuracy(num_classes=model_config.pseudo_vocab_sizes[i], average=None, top_k=5)
        

    val_metrics = {}
    for i in range(len(model_config.pseudo_vocab_sizes)):
        val_metrics[f"Val_Accuracy_task_{i}"] = MulticlassAccuracy(num_classes=model_config.pseudo_vocab_sizes[i], average=None)
        # val_metrics[f"Val_Top5_Accuracy_task_{i}"] = MulticlassAccuracy(num_classes=model_config.pseudo_vocab_sizes[i], average=None, top_k=5)
    
    for train_metric, val_metric in zip(train_metrics.values(), val_metrics.values()):
        train_metric.to(device)
        val_metric.to(device)
        
    # time_running_average_embed_reshape_train = 0
    # time_running_average_model_forward_train = 0
    # time_running_average_embed_reshape_val = 0
    # time_running_average_model_forward_val = 0

    for epoch in range(start_epoch, epochs):
        
        if global_step >= args.training_steps:
            break

        model.train()
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        rng_generator.manual_seed(epoch + 42) 
            
        if epoch == start_epoch:
            # calculate how many batches were finished in this partial epoch
            batches_in_epoch = len(train_set) // args.batch_size
            current_epoch_step = global_step % batches_in_epoch
            samples_to_skip = current_epoch_step * args.batch_size
        else:
            samples_to_skip = 0
            
        train_sampler = ResumableRandomSampler(
            train_set,
            batch_size=args.batch_size,
            start_idx=samples_to_skip,
            generator=rng_generator
        )
        
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            prefetch_factor=4 if args.num_workers > 0 else None,
            persistent_workers=True if args.num_workers > 0 else False,
            pin_memory=True,
            collate_fn=train_set.collate_fn,
            drop_last=True,
            shuffle=False,
            worker_init_fn=worker_init_fn
        )
        
        train_losses = []
        for metric in train_metrics.values():
            metric.reset()
            
        # start_batch = global_step % len(train_loader)

        # for batch in tqdm(
        #                     islice(
        #                         train_loader,
        #                         start_batch,
        #                         len(train_loader)),
        #                     initial=global_step,
        #                     total=len(train_loader) - start_batch
        #                 ):
        
        for batch in tqdm(train_loader, initial=global_step, total=len(train_loader)):
            
            # time_to_log = {}
            
            # start = perf_counter() # todo

            global_step += 1

            # --- unpack ---

            x = batch.x
            edge_index = batch.edge_index
            labels = batch.y
            node_lead_ids = batch.node_lead_ids
            node_temporal_pos = batch.node_temporal_pos
            b = batch.batch

            # --- move to cuda ---
            
            # start = perf_counter()
            # torch.cuda.synchronize()
            
            x = x.to(device)
            edge_index = edge_index.to(device)
            labels = labels.to(device)
            node_lead_ids = node_lead_ids.to(device)
            node_temporal_pos = node_temporal_pos.to(device)
            b = b.to(device)
            
            # newly added: compute equivalent of kmeans predict on labels, which are actually features
            # labels are L*S, D, centroids are list of length num_tasks, each element is tensor of shape (num_clusters, D)
            with torch.no_grad():
                labels = torch.stack([
                    torch.argmin(
                        torch.cdist(
                            labels,
                            centroid.unsqueeze(0) # 1, num_clusters, D
                        ).squeeze(0), # num_clusters, D -> num_clusters
                        dim=1
                    ) for centroid in centroids
                ], dim=1) # num_tasks, L*S -> L*S, num_tasks
            
            # torch.cuda.synchronize()
            # elapsed = perf_counter() - start
            # logger.info(f"Time to map data to device:{perf_counter()-start}")

            # with torch.autocast(
            #     "cuda",
            #     enabled=True,
            #     dtype=torch.bfloat16
            #     # dtype=torch.float32
            #     ): # !appereantly it can work only in bfloat16 and float32
                
                # start = perf_counter()
                # torch.cuda.synchronize()

                # NOTE: this certainly works for GIN and GAT.
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
                metrics=train_metrics,
                real_node_mask=batch.real_node_mask.to(device) if args.pad_unsampled_leads else None
            )
                
                # torch.cuda.synchronize()
                # elapsed_model_forward_train = perf_counter() - start
                # time_to_log["elapsed_model_forward_train"] = elapsed_model_forward_train
                # time_running_average_model_forward_train = (time_running_average_model_forward_train + elapsed_model_forward_train) / 2
                # time_to_log["running_avg_model_forward_train"] = time_running_average_model_forward_train

            loss = output["loss"] / args.accumulation_steps
            scaler.scale(loss).backward()
            train_losses.append(loss.item())

            # --- gradient accumulation ---

            if global_step % args.accumulation_steps == 0 or global_step == args.training_steps:
                scaler.step(optimizer)
                lr_scheduler.step()
                scaler.update()
                optimizer.zero_grad()

            # NOTE: if val_frequency equals the number of steps in an epoch, then the validation is performed at the end of every epoch

            if not args.skip_val and (global_step % args.val_frequency == 0 or global_step == args.training_steps):
                
                model.eval()

                # --- clear validation metric/loss containers ---

                val_losses = []
                target_score = 0
                for metric in val_metrics.values():
                    metric.reset()

                logger.info(f"Validating the model at step {global_step}")

                # --- validation loop ---
                
                val_losses = validate(model, val_loader, device, val_criterion, val_metrics, val_losses, centroids)
                # --- end of validation loop ---
                            
                val_loss = np.mean(val_losses)
                train_loss = np.mean(train_losses)

                # ideally one would clear this list when starting an epoch but to keep the train curve aligned with validation curve we need to clear it here
                train_losses.clear()

                # --- logging ---
                if global_step % args.log_frequency == 0 or global_step == args.training_steps:
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
                        metric.reset() # to keep train and val metrics aligned

                    
                    for name, metric in val_metrics.items():
                        score = metric.compute()
                        macro = score.mean()
                        logger.info(f"Validation {name} (macro) = {macro}")
                        to_log.update({f"Validation_{name}" : macro})
                        target_score += macro
                    target_score /= len(model_config.pseudo_vocab_sizes) # mean along task dimension
                    
                    logger.info(f"Validation Overall Accuracy = {target_score}")
                    to_log.update({f"Validation_Overall_Accuracy" : target_score})

                    if args.wandb:
                        wandb.log(to_log)
                # --- end of logging ---


                # --- checkpointing if either or Overall Accuracy improved by a minimum delta && early stopping ---

                if val_loss <= best_val_loss - MINIMAL_IMPROVEMENT_PRETRAINING:

                    logger.info(f"Validation loss improved from {best_val_loss} to {val_loss}")

                    best_val_loss = val_loss
                    best_target_score = target_score if target_score > best_target_score + MINIMAL_IMPROVEMENT_FINETUNING else best_target_score 

                    logger.info(f"Current best target score = {best_target_score}")


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
                    checkpoint_name = f"{args.encoder_name}_{global_step}_{wandb.run.name}.pt"
                    torch.save(checkpoint, os.path.join(args.pretrained_models_path, checkpoint_name))

                    dynamic_regularizer(
                        optimizer,
                        model,
                        penalty=False,
                        wd_dynamic_reg_factor=args.wd_dynamic_reg_factor,
                        dropout_dynamic_reg_factor=args.dropout_dynamic_reg_factor
                        ) if args.dynamic_reg else None

                elif target_score >= best_target_score + MINIMAL_IMPROVEMENT_PRETRAINING:

                    logger.info(f"Validation target score improved from {best_target_score} to {target_score}")
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

                    checkpoint_name = f"{args.encoder_name}_{global_step}_{wandb.run.name}.pt"
                    torch.save(checkpoint, os.path.join(args.pretrained_models_path, checkpoint_name))

                    dynamic_regularizer(
                        optimizer,
                        model,
                        penalty=False,
                        wd_dynamic_reg_factor=args.wd_dynamic_reg_factor,
                        dropout_dynamic_reg_factor=args.dropout_dynamic_reg_factor
                    ) if args.dynamic_reg else None

                else:

                    patience_count += 1

                    if args.dynamic_reg and patience_count % args.validations_for_penalty == 0 and patience_count != patience:
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

                if args.save_last and global_step == args.training_steps:
                    checkpoint_name = f"{args.encoder_name}_{global_step}_{wandb.run.name}.pt"
                    checkpoint_file = os.path.join(args.pretrained_models_path, checkpoint_name)
                    if not os.path.exists(checkpoint_file):
                        logger.info(f"Saving last checkpoint at step {global_step}")
                        torch.save({
                            "global_step" : global_step,
                            "best_val_loss" : best_val_loss,
                            "model_config" : model.config,
                            "model_state_dict" : copy.deepcopy(model.state_dict()),
                            "optimizer_state_dict" : copy.deepcopy(optimizer.state_dict()),
                            "lr_scheduler_state_dict" : copy.deepcopy(lr_scheduler.state_dict()),
                            "patience_count" : patience_count,
                            "best_target_score" : best_target_score
                        }, checkpoint_file)

                model.train()
                
            else:                
                
                if global_step % args.log_frequency == 0 or global_step == args.training_steps:

                    logger.info("Logging training metrics and checkpointing only after skipping validation")

                    train_loss = np.mean(train_losses)
                    train_losses.clear()
                    to_log = {
                        "Training_loss" : train_loss,
                    }
                    logger.info(f"Step {global_step}, training loss = {train_loss}")
                    
                    for name, metric in train_metrics.items():
                        score = metric.compute()
                        macro = score.mean()
                        logger.info(f"Training {name} (macro) = {macro}")
                        to_log.update({f"Training_{name}" : macro})
                        metric.reset() # to keep train and val metrics aligned

                    if args.wandb:
                        wandb.log(to_log)
                        
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

                    checkpoint_name = f"{args.encoder_name}_{global_step}_{wandb.run.name}.pt"
                    torch.save(checkpoint, os.path.join(args.pretrained_models_path, checkpoint_name))
                    
                model.train() # just to be sure

            # wandb.log(time_to_log)
            
            # print(f"Time for step {global_step} = {perf_counter() - start}") # todo
            if global_step >= args.training_steps:
                break


    # --- end of training loop ---

    logger.info(f"End of training")
    logger.info(f"Total steps = {global_step}")
    logger.info(f"Best validation loss = {best_val_loss}")
    logger.info(f"Best Validation Overall Accuracy = {best_target_score}")
    wandb.finish()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Pretraining script")

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
        help="Name of wandb project. Default project name",
        type=str,
        default="RELAY"
    )

    parser.add_argument(
        "--wandb_group",
        help="Name of wandb group. Defatult pre-training",
        type=str,
        default="pre-training"
    )

    parser.add_argument(
        "--wandb_run_name",
        help="Name of wandb run name. If None, wandb sets it automatically",
        type=str,
        default=None
    )

    parser.add_argument(
        "--resume",
        help="Whether to resume pre-training loading a checkpoint. Default False",
        action="store_true"
    )
    
    parser.add_argument(
        "--change_iteration",
        help="Whether to change pre-training iteration loading a checkpoint but changing the pre-training head. Default False",
        action="store_true"
    )

    parser.add_argument(
        "--checkpoint_path",
        help="Path to checkpoint to resume pre-training",
        type=str,
        default=None
    )

    parser.add_argument(
        "--encoder_name",
        help="Name of the backbone to use. Default gat",
        choices=['gin', 'gat', 'mlp', 'graph_transformer'],
        type=str,
        default='gat'
    )

    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate. Default=5e-5",
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
        "kmeans_path",
        type=str,
        help="Path to the file containing the paths to kmeans model to load.",
    )

    parser.add_argument(
        "path_to_train_dataset_csv",
        type=str,
        help="Path to the .csv file referencing the training data",
    )

    parser.add_argument(
        "ecg_train_dir_path",
        type=str,
        help="Path to the directory containing the actual training data",
    )

    parser.add_argument(
        "train_features_path",
        type=str,
        help="Path to the directory to the training features to feed the kmeans models",
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
        "path_to_val_dataset_csv",
        type=str,
        help="Path to the .csv file referencing the validation data",
    )

    parser.add_argument(
        "ecg_val_dir_path",
        type=str,
        help="Path to the directory containing the actual validation data",
    )

    parser.add_argument(
        "val_features_path",
        type=str,
        help="Path to the directory to the validation features to feed the kmeans models",
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
        help="Patience: number of validations with no improvements to wait before stopping the training. Default 10",
        default=10,
        type=int
    )

    parser.add_argument(
        "--val_frequency",
        help="Frequency of validation expressed in terms of steps. Default 5000",
        default=5000,
        type=int
    )
    
    parser.add_argument(
        "--log_frequency",
        help="Frequency of logging expressed in terms of steps. Default None, that is, equal to val_frequency",
        default=None,
        type=int
    )

    parser.add_argument(
        "--pretrained_models_path", 
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
        default=4
    )
    
    parser.add_argument(
        "--wd_dynamic_reg_factor",
        help="Factor by which to multiply/divide the weight decay when applying the dynamic regularization. Default 5",
        type=float,
        default=WEIGHT_DECAY_DYNAMIC_REG_FACTOR
    )
    
    parser.add_argument(
        "--dropout_dynamic_reg_factor",
        help="Value to add/subtract to/from the dropout when applying the dynamic regularization. Default 0.05",
        type=float,
        default=DROPOUT_DYNAMIC_REG_FACTOR
    )
    
    parser.add_argument(
        "--gradient_clipping",
        help="Whether or not use gradient clipping. Default False",
        type=float,
        default=None
    )
    
    parser.add_argument(
        "--label_smoothing",
        help="Label smoothing to use in the cross entropy loss. Default 0.0",
        type=float,
        default=0.0
    )
    
    parser.add_argument(
        "--train_how_many_leads",
        help="Number of leads randomly sampled per training ECG. Default None, i.e. random lead sampling with L ~ U{1..12}. " \
        "Set 12 to pre-train without random lead sampling. Validation always uses random lead sampling",
        type=int,
        default=None
    )

    parser.add_argument(
        "--pad_unsampled_leads",
        help="Keep the leads NOT drawn by random lead sampling as all-zero signals (full 12-lead graph, "
             "zero-padding counterpart of RLS) instead of dropping them. Only the sampled leads' nodes are "
             "masked and scored. Default False",
        action="store_true"
    )

    parser.add_argument(
        "--save_last",
        help="Always save a checkpoint at the last training step, even if validation did not improve. Default False",
        action="store_true"
    )

    parser.add_argument(
        "--skip_val",
        help="Whether or not skip validation and logging. Default False",
        action="store_true"
    )

    args = parser.parse_args()

    # --- input check ---

    if args.resume or args.change_iteration:
        if args.checkpoint_path is None or not os.path.exists(args.checkpoint_path):
            raise ValueError(f"If resuming pre-training you must provide a valid checkpoint path")
    else:
        if args.model_config_path is None or not os.path.exists(args.model_config_path):
            raise ValueError("If not resuming pre-training, then argument model_config_path must be passed and be valid")

    if args.warm_up_perc > 1.0 or args.warm_up_perc < 0.0:
        raise ValueError(f"Argument warm_up_perc must be in [0,1]. Got {args.warm_up_perc}")

    # --- check existence of all required paths ---
    required_paths = [
        ("kmeans_path", args.kmeans_path),
        ("path_to_train_dataset_csv", args.path_to_train_dataset_csv),
        ("ecg_train_dir_path", args.ecg_train_dir_path),
        ("train_features_path", args.train_features_path),
        ("path_to_val_dataset_csv", args.path_to_val_dataset_csv),
        ("ecg_val_dir_path", args.ecg_val_dir_path),
        ("val_features_path", args.val_features_path),
        ("pretrained_models_path", args.pretrained_models_path)
    ]

    for name, path in required_paths:
        if not os.path.exists(path):
            raise ValueError(f"Path argument '{name}' does not exist. Got {path}")

    # --- end of input check

    logger.info("Inserted arguments: ")
    for arg in vars(args):
        logger.info(f"{arg} = {getattr(args, arg)}")
        
    if args.log_frequency is None:
        args.log_frequency = args.val_frequency

    pretrain(args)
