from torch.utils.data import Dataset
from scipy import signal
from modelling_graph_ecg import *
import pandas as pd 
import os
import argparse 
import torch 
from torch.utils.data import DataLoader
import math
from modelling_graph_ecg import MODEL_NAME_2_PRETRAINING_MODEL
from config import MODEL_NAME_2_CONFIG
import json
from loguru import logger
from consts import *
import numpy as np
import random
import torch_geometric

class ECGEncodingDataset(Dataset):
    def __init__(
        self,
        path_to_dataset_csv : str,
        ecg_dir_path : str,
        start_perc : float = 0.0,
        end_perc : float = 1.0,
        downsampling_factor : Optional[int] = 5,
        nodes_per_lead : int = 20,
        create_edge_index : Optional[bool] = True,
        edge_strategy : Optional[str] = "spatiotemporal", 
        use_spatiotemporal_pos_enc : Optional[bool] = False,
        ):
        
        
        logger.info(f"Loading dataset from {path_to_dataset_csv}...")

        self.ecg_dataframe = pd.read_csv(path_to_dataset_csv, dtype={'filename': str})
        self.ecg_dataframe = self.ecg_dataframe.iloc[
            math.floor(start_perc * len(self.ecg_dataframe)) : math.floor(end_perc * len(self.ecg_dataframe))
        ].reset_index(drop=True)
        
        # # take only the first 64 instances for debugging
        # logger.warning("DEBUG MODE: taking only the first 64 instances of the dataset")
        # self.ecg_dataframe = self.ecg_dataframe.iloc[:64]
        
        self.ecg_dir_path = ecg_dir_path
        self.downsampling_factor = downsampling_factor
        self.create_edge_index = create_edge_index 
        self.use_spatiotemporal_pos_enc = use_spatiotemporal_pos_enc       
        self.num_subgraph_nodes = nodes_per_lead        

        if edge_strategy not in ["fully_connected", "spatiotemporal"]:
            raise ValueError(f"edge_strategy must be one of {['fully_connected', 'spatiotemporal']} but got {edge_strategy}")
        
        self.edge_strategy = edge_strategy

        self._edge_index_cache = {} # memoization for edge_index
        self.precompile_edge_index()

    def precompile_edge_index(self):
        """
        Precompile all possible edge indices for different number of leads and edge strategies.
        This is done to avoid recomputing the edge index at each __getitem__ call, which can be expensive.
        The precompiled edge indices are stored in a dictionary with keys as (number_of_leads, edge_strategy).
        Simply retrieve the edge index from the dictionary during __getitem__ call can be made because the edge index only depends on the number of leads and the edge strategy, not on the actual data.
        This method precomputes undirected edge indices for number of leads from 1 to 12 (inclusive), in both "fully_connected" and "spatiotemporal" strategies
        """

        logger.info(f"Precompiling all possible edge indices for {self.edge_strategy} strategy...")

        for l in range(1, 13):

            cache_key = (l, self.edge_strategy)

            if self.edge_strategy == "fully_connected":

                total_nodes = l*self.num_subgraph_nodes                
                
                all_indices = torch.arange(total_nodes)
                i, j = torch.combinations(all_indices, r=2).unbind(1)
                # add edges (i, j) and immediately (j, i)
                row = []
                col = []
                for src, tgt in zip(i, j):
                    row.append(src)
                    col.append(tgt)
                    row.append(tgt)
                    col.append(src)
                    
                row = [torch.tensor(row)]
                col = [torch.tensor(col)]
                    
                edge_index = torch.stack([torch.cat(row), torch.cat(col)], dim=0) # (2, E)

                self._edge_index_cache[cache_key] = edge_index

            elif self.edge_strategy == "spatiotemporal":

                row = []
                col = []

                # 1) Intra-subgraph fully connected edges
                for lead_idx in range(l):
                    start = lead_idx * self.num_subgraph_nodes
                    nodes = torch.arange(start, start + self.num_subgraph_nodes)
                    i, j = torch.combinations(nodes, r=2).unbind(1)
                    # add edges (i, j) and immediately (j, i)
                    for src, tgt in zip(i, j):
                        row.append(src)
                        col.append(tgt)
                        row.append(tgt)
                        col.append(src)

                # 2) Inter-subgraph edges: connect homologous nodes (same temporal position)
                for t in range(self.num_subgraph_nodes):
                    nodes = torch.arange(l) * self.num_subgraph_nodes + t
                    i, j = torch.combinations(nodes, r=2).unbind(1)
                    for src, tgt in zip(i, j):
                        row.append(src)
                        col.append(tgt)
                        row.append(tgt)
                        col.append(src)
                        
                row = [torch.tensor(row)]
                col = [torch.tensor(col)]

                edge_index = torch.stack([torch.cat(row), torch.cat(col)], dim=0)

                self._edge_index_cache[cache_key] = edge_index

            else:
                raise ValueError(f"edge_strategy must be one of {['fully_connected', 'spatiotemporal']} but got {self.edge_strategy}")
            
        logger.info("Done witht the edge indices precompilation.")
    
    def __len__(self):
        return len(self.ecg_dataframe)

    def __getitem__(self, idx):
        
        record = self.ecg_dataframe.iloc[idx]
        ecg_filename = record['filename'] # the mere filename

        ecg_path = ecg_filename if os.path.isfile(ecg_filename) else os.path.join(self.ecg_dir_path, ecg_filename)

        ecg_data = np.load(ecg_path, mmap_mode='r') # (12, 5000+)       
            
        ecg_data = ecg_data[:, :SAMPLES_IN_5_SECONDS_AT_500HZ]

        sampled_leads = LEADS

        sampled_indices = [LEAD_2_INDEX[sampled_lead] for sampled_lead in sampled_leads]

        ecg_data = ecg_data[sampled_indices] # L, time_steps 
        
        # --- downsampling ---
        if self.downsampling_factor is not None:
            ecg_data = signal.decimate(ecg_data, self.downsampling_factor)            

        ecg_data = torch.from_numpy(ecg_data.copy()).float() # L, time_steps

        if self.create_edge_index:
            edge_index = self._edge_index_cache[len(sampled_leads), self.edge_strategy]
        else:
            edge_index = torch.empty((2,0), dtype=torch.long)


        node_lead_ids = torch.empty((0,), dtype=torch.long)
        node_temporal_pos = torch.empty((0,), dtype=torch.long)
        if self.use_spatiotemporal_pos_enc:
            node_lead_ids = torch.tensor([[LEAD_2_INDEX[lead]] * self.num_subgraph_nodes for lead in sampled_leads], dtype=torch.long).flatten() # L, S -> flatten to L*S

            node_temporal_pos = torch.tensor([list(range(self.num_subgraph_nodes)) for _ in range(len(sampled_leads))], dtype=torch.long).flatten() # L, S -> flatten to L*S
        
        # create fake labels of shape L*S, num_tasks
        labels = torch.empty((len(sampled_leads) * self.num_subgraph_nodes, 0))
        
        return torch_geometric.data.Data(
            x = ecg_data, # L, time_steps
            y = labels,
            edge_index = edge_index, # (2, E)
            node_lead_ids = node_lead_ids, # (L*S,)
            node_temporal_pos = node_temporal_pos, # (L*S, )            
        )


    def collate_fn(self, batch : List[torch_geometric.data.Data]) -> torch_geometric.data.Data:
        """
        Custom collate function to handle batches of ECGGraphDataset items. Class method to access self.num_subgraph_nodes.
        batch is a List of `batch_size` torch_geometric.data.Data objects, each containing:
        x: (L_i, time_steps) (NOTE: x is not the matrix of node feature vectors)
        edge_index: (2, E) or (2, 0) if no edges
        node_lead_ids: (L_i*S,) or empty tensors
        node_temporal_pos: (L_i*S, ) or empty tensors
        
        Returns a batched torch_geometric Data object where:
        x are concatenated along the first dimension: (sum_{i=1}^B L_i, time_steps)
        edge_index are shifted by the cumulative sum of L_i*S to account for the new node indices in the batch. Final shape is (2, sum_{i=1}^{B} E_i)
        node_lead_ids are concatenated along the first dimension (sum_{i=1}^B L_i*S,)
        node_temporal_pos are concatenated along the first dimension (sum_{i=1}^B L_i*S, )        
        A batch.batch is also created to indicate the graph to which each node belongs (sum_{i=1}^B L_i*S,)
        """
        
        x_list = [data.x for data in batch]
        y_list = [data.y for data in batch]
        edge_index_list = [data.edge_index for data in batch]
        node_lead_ids_list = [data.node_lead_ids for data in batch]
        node_temporal_pos_list = [data.node_temporal_pos for data in batch]
        
        x = torch.cat(x_list, dim=0) # (sum_{i=1}^B L_i, time_steps)
        y = torch.cat(y_list, dim=0)
        node_lead_ids = torch.cat(node_lead_ids_list, dim=0) # (sum_{i=1}^B L_i*S,)
        node_temporal_pos = torch.cat(node_temporal_pos_list, dim=0) # (sum_{i=1}^B L_i*S, )
        
        # shift edge indices
        edge_index = []
        cumsum_nodes = 0
        for edge_index_i in edge_index_list:
            edge_index.append(edge_index_i + cumsum_nodes)
            cumsum_nodes += edge_index_i.max().item() + 1 if edge_index_i.numel() > 0 else 0 
        edge_index = torch.cat(edge_index, dim=1) if len(edge_index) > 0 else torch.empty((2,0), dtype=torch.long) # (2, sum_{i=1}^{B} E_i)
        
        batch_vector = []
        cumsum_nodes = 0
        for i, data in enumerate(batch):
            num_nodes = data.x.size(0) # L_i
            num_nodes = num_nodes * self.num_subgraph_nodes
            batch_vector.extend([i] * num_nodes)
            cumsum_nodes += num_nodes
        batch_vector = torch.LongTensor(batch_vector) # (sum_{i=1}^B L_i*S,)
        
        return torch_geometric.data.Data(
            x = x,
            y = y,
            edge_index = edge_index,
            node_lead_ids = node_lead_ids,
            node_temporal_pos = node_temporal_pos,
            batch = batch_vector,
        )


def dump_latents(args):
    
    # --- fixing seeds ---
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # --- loading checkpoint ---

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
    model_name = os.path.basename(args.checkpoint_path)

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)
    model_config = checkpoint['model_config']
    model = MODEL_NAME_2_PRETRAINING_MODEL[args.encoder_name](model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    logger.info(f"Checkpoint {model_name} loaded.")
    
    # --- creation of dataset and dataloader ---

    dataset = ECGEncodingDataset(
        path_to_dataset_csv=args.path_to_dataset_csv,
        ecg_dir_path=f"./{'train' if args.is_train else 'val'}_self_supervised/", 
        start_perc=args.start_perc,
        end_perc=args.end_perc,
        downsampling_factor=args.downsampling_factor,
        nodes_per_lead=model_config.nodes_per_lead,
        create_edge_index=True,
        edge_strategy="spatiotemporal",
        use_spatiotemporal_pos_enc=False
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=6,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )
    
    # create output dataframe where entries correspond to batched data
    output_dfs = [pd.DataFrame(columns=['filename', "idx_start", "idx_end"], index=range(len(dataloader))) for _ in range(model_config.num_layers + 1)]

    # create output dirs for latents
    if args.is_train:
        for i in range(model_config.num_layers + 1):
            os.makedirs(f"./latents_layer_{i}", exist_ok=True)
    else:
        for i in range(model_config.num_layers + 1):
            os.makedirs(f"./val_latents_layer_{i}", exist_ok=True)

    model.eval()

    criterion = torch.nn.CrossEntropyLoss().to(device)

    for i, batch in enumerate(dataloader):        

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


        with torch.no_grad():
            output = model(
                x=x,
                edge_index=edge_index,
                criterion=criterion,
                labels=labels,
                node_lead_ids=node_lead_ids,
                node_temporal_pos=node_temporal_pos,
                batch=b,
                batch_size=args.batch_size,
                output_hidden_states=True,
                return_dict=True,
                metrics={},
                output_attention_weights=False
            )
        
        assert len(output['hidden_states']) == model_config.num_layers + 1, \
            f"Expected {model_config.num_layers + 1} hidden states but got {len(output['hidden_states'])}"
            
        assert output['hidden_states'][0].shape == (args.batch_size * len(LEADS) * model_config.nodes_per_lead, model_config.hidden_channels), \
            f"Expected hidden state shape {(args.batch_size * len(LEADS) * model_config.nodes_per_lead, model_config.hidden_channels)} but got {output['hidden_states'][0].shape}"
            
        hidden_states = [
            hs.reshape(args.batch_size, len(LEADS), model_config.nodes_per_lead, model_config.hidden_channels).cpu().numpy()
            for hs in output['hidden_states']
            ] # num_layers + 1 x (B, L, S, hidden_channels)
        
        # save in the corresponding output dir and populate the corresponding output df with full path, idx_start, idx_end for mapping from the original dataset
        for layer_idx, hs in enumerate(hidden_states):
            output_path = f"./{'val' if not args.is_train else ''}_latents_layer_{layer_idx}/latents_batch_{i}.npy"
            np.save(output_path, hs)
            output_dfs[layer_idx].at[i, 'filename'] = output_path
            output_dfs[layer_idx].at[i, 'idx_start'] = i * args.batch_size
            output_dfs[layer_idx].at[i, 'idx_end'] = i * args.batch_size + hs.shape[0] - 1
            
    # save output dfs
    for layer_idx, df in enumerate(output_dfs):
        output_csv_path = f"./{'val' if not args.is_train else ''}_latents_layer_{layer_idx}.csv"
        df.to_csv(output_csv_path, index=False)
        logger.info(f"Saved latents metadata for layer {layer_idx} to {output_csv_path}")
            
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--checkpoint_path',
        type=str,
        required=True,
        help='Path to checkpoint to load the model from.'
    )

    parser.add_argument(
        '--encoder_name',
        type=str,
        required=True,
        help='Name of the encoder used (gat, gcn, gin, mlp, graph_transformer).'
    )
    
    parser.add_argument(
        '--path_to_dataset_csv',
        type=str,
        required=True,
        help='Path to the dataset CSV file.',
        default=None
    )
    
    parser.add_argument(
        '--start_perc',
        type=float,
        default=0.0,
        help='Start percentage of the dataset to use.'
    )
    
    parser.add_argument(
        '--end_perc',
        type=float,
        default=1.0,
        help='End percentage of the dataset to use.'
    )
    
    parser.add_argument(
        '--downsampling_factor',
        type=int,
        default=5,
        help='Downsampling factor to apply to the ECG signals.'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for the dataloader.'
    )
    
    parser.add_argument(
        "--is_train",
        action="store_true",
        help="Whether to use the training set or validation set."
    )
    
    

    args = parser.parse_args()

    dump_latents(args)





    







