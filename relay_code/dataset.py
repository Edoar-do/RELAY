# import os
# os.environ["OMP_NUM_THREADS"] = "1" 
# os.environ["MKL_NUM_THREADS"] = "1" 
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

# import torch
# import numpy as np

# # Force PyTorch intra-op parallelism to 1
# torch.set_num_threads(1)

# import threadpoolctl

# import numpy as np
# from loguru import logger
# import pandas as pd
# import torch

# from torch.utils.data import Dataset
# from scipy import signal
# import random
# from typing import List, Union, Optional, Any
# from consts import *
# from sklearn import preprocessing
# import torch_geometric

# class ECGDataset(Dataset):
#     def __init__(
#         self,
#         path_to_dataset_csv : str,
#         ecg_dir_path : str,
#         downsampling_factor : Optional[int] = None,
#         features_path : Optional[str] = None,
#         nodes_per_lead : int = 20,
#         label_start_index : Optional[int] = 3,
#         pretrain : Optional[bool] = True,
#         random_crop : Optional[bool] = False,
#         return_full_length : Optional[bool] = False,
#         create_edge_index : Optional[bool] = True,
#         edge_strategy : Optional[str] = "spatiotemporal", 
#         use_temporal_encoding : Optional[bool] = False,
#         use_lead_encoding : Optional[bool] = False,
#         sampling : Optional[Union[str, List[str]]] = 'random',
#         how_many_leads : Optional[int] = None,
#         ):
#         '''
#         Args:
#         - `path_to_dataset_csv` = path to dataset in csv format to use. 
#         This csv should contain as many binary columns as labels to predict in the multilabel classification task.
#         In case of multiclass classification problem, only one column is expected and the values are the integers representing the classes in the range [0, n_classes-1].
#         - `ecg_dir_path` = path to the dir where raw ecgs are
#         - `downsampling_factor` = integer value indicating the downsampling factor to apply to the ecg signal. Default None
#         - `features_path` = path to the dir where dumped features are (extracted from shards or from mid-layers of transformer). Used only when pretrain is true
#         - `nodes_per_lead` = number of nodes that will be created per lead. Each lead is represented as a subgraph of `nodes_per_lead` nodes. Default 20
#         - `label_start_index` = index of the column in the csv dataset at which labels are. Use when pretrain and encode are false
#         - `pretrain` = boolean indicating whether pretraining is in progress or not
#         - `encode` = additional boolean value used only to speed up features dumping
#         - `beat_based_attention_mask` = optional boolean indicating whether beat-based attention mask should be calculated. Default False
#         - `random_crop` = optional boolean indicating whether to randomly crop the ecg signal. Default False. To use only during finetuning and testing to avoid misalignments between signals and features.
#         - `return_full_length` = optional boolean indicating whether to return the full length of the ecg signal. Default False. 
#         - `create_edge_index` = optional boolean flag indicating whether to create an edge index. Default true. False for MLP
#         - `edge_strategy` = optional string to indicate the topology of the input graph. Options are [fully_connected, spatiotemporal]. Default spatiotemporal
#         - `use_temporal_encoding` = optional boolean indicating whether to create node temporal positions as additional torch_geometric.data.Data attributes. Default False.
#         - `use_lead_encoding` = optional boolean indicating whether to create node lead ids as additional torch_geometric.data.Data attributes. Default False.
#         This might be useful if the model uses spatiotemporal positional encodings.
#         - `sampling` = optional string or list of strings taking values in 'random' or [lead_names], respectively. 
#         If 'random', a random number of leads is selected or, if `how_many_leads` is set, that exact number of leads is randomly sampled.
#         If not random, then a list of lead names as strings
#         - `how_many_leads` = optional int, default None. When `sampling` is random and this parameter is set, exactly `how_many_leads` leads are sampled randomly
     
#         The `__getitem__` method returns:
#         - a torch_geometric Data object containing:
#             x = ecg_data, (L*N, F)
#             edge_index = edge_index,
#             y = labels, 
#             node_lead_ids = node_lead_ids, (L*N, )
#             node_temporal_pos = node_temporal_pos, (L*N, )
        
#         '''
        
#         logger.info(f"Loading dataset from {path_to_dataset_csv}...")

#         self.ecg_dataframe = pd.read_csv(path_to_dataset_csv, dtype={'filename': str})
        
#         # # take only the first 64 instances for debugging
#         # logger.warning("DEBUG MODE: taking only the first 64 instances of the dataset")
#         # self.ecg_dataframe = self.ecg_dataframe.iloc[:64]
        
#         self.ecg_dir_path = ecg_dir_path
#         self.downsampling_factor = downsampling_factor
#         self.pretrain = pretrain
#         self.random_crop = random_crop
#         self.return_full_length = return_full_length
#         self.create_edge_index = create_edge_index 
#         self.use_temporal_encoding = use_temporal_encoding
#         self.use_lead_encoding = use_lead_encoding     
#         self.num_subgraph_nodes = nodes_per_lead
        

#         if edge_strategy not in ["fully_connected", "spatiotemporal"]:
#             raise ValueError(f"edge_strategy must be one of {['fully_connected', 'spatiotemporal']} but got {edge_strategy}")
        
#         self.edge_strategy = edge_strategy
        
#         if type(sampling) != str and type(sampling) != list:
#             raise ValueError(f"type of sampling must be either str or list, got {type(sampling)}")
        
#         if type(sampling) == str:
#             if sampling != 'random':
#                 raise NotImplementedError(f"when sampling is a string, not other value than 'random' is possible. Got {sampling}")
#         else:
#             if not set(LEAD_2_INDEX.keys()).issuperset(set(sampling)):
#                 raise ValueError(f"Provided sampling list must contain valid lead names including {LEAD_2_INDEX.keys()}. Got {sampling}")

#         self.sampling = sampling
#         self.how_many_leads = how_many_leads

#         self._edge_index_cache = {} # memoization for edge_index
#         self.precompile_edge_index()

#         if pretrain:            
#             self.features_path = features_path
#             self.normalizer = preprocessing.Normalizer() # no need to fit this
#         else:
#             self.diagnoses_cols = self.ecg_dataframe.columns.values.tolist()[label_start_index:]
#             assert len(self.diagnoses_cols) > 0, "No labels found in the dataset"
#             self.weights = self.compute_weights()
            

        
#     def compute_weights(self):
#         logger.info("Computing weights...")        
#         if len(self.diagnoses_cols) > 1:
#             weights = []
#             for label in self.diagnoses_cols:
#                 count = self.ecg_dataframe[label].sum()
#                 weight = (self.ecg_dataframe.__len__() - count) / (count + 1e-9)
#                 weights.append(weight)
#             logger.info("Done with the weights.")
#         else:
#             num_labels = self.ecg_dataframe[self.diagnoses_cols[0]].max() + 1
#             weights = num_labels / self.ecg_dataframe[self.diagnoses_cols].value_counts()
#             weights = weights.values.tolist()
#         return torch.FloatTensor(weights)

#     def precompile_edge_index(self):
#         """
#         Precompile all possible edge indices for different number of leads and edge strategies.
#         This is done to avoid recomputing the edge index at each __getitem__ call, which can be expensive.
#         The precompiled edge indices are stored in a dictionary with keys as (number_of_leads, edge_strategy).
#         Simply retrieve the edge index from the dictionary during __getitem__ call can be made because the edge index only depends on the number of leads and the edge strategy, not on the actual data.
#         This method precomputes undirected edge indices for number of leads from 1 to 12 (inclusive), in both "fully_connected" and "spatiotemporal" strategies
#         """

#         logger.info(f"Precompiling all possible edge indices for {self.edge_strategy} strategy...")

#         for l in range(1, 13):

#             cache_key = (l, self.edge_strategy)

#             if self.edge_strategy == "fully_connected":

#                 total_nodes = l*self.num_subgraph_nodes                
                
#                 all_indices = torch.arange(total_nodes)
#                 i, j = torch.combinations(all_indices, r=2).unbind(1)
#                 # add edges (i, j) and immediately (j, i)
#                 row = []
#                 col = []
#                 for src, tgt in zip(i, j):
#                     row.append(src)
#                     col.append(tgt)
#                     row.append(tgt)
#                     col.append(src)
                    
#                 row = [torch.tensor(row)]
#                 col = [torch.tensor(col)]
                    
#                 edge_index = torch.stack([torch.cat(row), torch.cat(col)], dim=0) # (2, E)

#                 self._edge_index_cache[cache_key] = edge_index

#             elif self.edge_strategy == "spatiotemporal":

#                 row = []
#                 col = []

#                 # 1) Intra-subgraph fully connected edges
#                 for lead_idx in range(l):
#                     start = lead_idx * self.num_subgraph_nodes
#                     nodes = torch.arange(start, start + self.num_subgraph_nodes)
#                     i, j = torch.combinations(nodes, r=2).unbind(1)
#                     # add edges (i, j) and immediately (j, i)
#                     for src, tgt in zip(i, j):
#                         row.append(src)
#                         col.append(tgt)
#                         row.append(tgt)
#                         col.append(src)

#                 # 2) Inter-subgraph edges: connect homologous nodes (same temporal position)
#                 for t in range(self.num_subgraph_nodes):
#                     nodes = torch.arange(l) * self.num_subgraph_nodes + t
#                     i, j = torch.combinations(nodes, r=2).unbind(1)
#                     for src, tgt in zip(i, j):
#                         row.append(src)
#                         col.append(tgt)
#                         row.append(tgt)
#                         col.append(src)
                        
#                 row = [torch.tensor(row)]
#                 col = [torch.tensor(col)]

#                 edge_index = torch.stack([torch.cat(row), torch.cat(col)], dim=0)

#                 self._edge_index_cache[cache_key] = edge_index

#             else:
#                 raise ValueError(f"edge_strategy must be one of {['fully_connected', 'spatiotemporal']} but got {self.edge_strategy}")
            
#         logger.info("Done witht the edge indices precompilation.")
    
#     def __len__(self):
#         return len(self.ecg_dataframe)


#     def __getitem__(self, idx):
    
#         with threadpoolctl.threadpool_limits(limits=1, user_api='blas'):
            
#             record = self.ecg_dataframe.iloc[idx]
#             ecg_filename = record['filename'] # the mere filename

#             ecg_path = ecg_filename if os.path.isfile(ecg_filename) else os.path.join(self.ecg_dir_path, ecg_filename)

#             ecg_data = np.load(ecg_path, mmap_mode='r') # (num_leads, 5000+)
                
#             if self.pretrain:
#                 ecg_data = ecg_data[:, :SAMPLES_IN_5_SECONDS_AT_500HZ]
#             elif self.random_crop: 
#                 start = torch.randint(0, ecg_data.shape[1] - SAMPLES_IN_5_SECONDS_AT_500HZ + 1, (1,)).item()
#                 ecg_data = ecg_data[:, start:start+SAMPLES_IN_5_SECONDS_AT_500HZ]
#             elif self.return_full_length:
#                 # we take a random crop of 10 seconds because it's the largest and most commong window in the literature
#                 assert ecg_data.shape[1] >= SAMPLES_IN_10_SECONDS_AT_500HZ, f"ecg length is less than 10 seconds for file {ecg_filename}"
#                 start = torch.randint(0, ecg_data.shape[1] - SAMPLES_IN_10_SECONDS_AT_500HZ + 1, (1,)).item() 
#                 ecg_data = ecg_data[:, start:start+SAMPLES_IN_10_SECONDS_AT_500HZ]
#             else:
#                 ecg_data = ecg_data[:, :SAMPLES_IN_5_SECONDS_AT_500HZ]

#             # --- random lead sampling ----  
            
#             num_available_leads = ecg_data.shape[0]    
            
#             # Create a mapping from lead names to actual indices for this ECG to handle ECGs with arbitrary numbers of leads
#             available_lead_mapping = {INDEX_2_LEAD.get(i, f"Lead_{i}"): i for i in range(num_available_leads)}
            
#             if self.sampling == "random":
#                 if self.how_many_leads is not None:
#                     l = min(self.how_many_leads, num_available_leads)
#                 else: 
#                     l = torch.randint(1, num_available_leads + 1, (1,)).item()
                    
#                 all_lead_names = list(available_lead_mapping.keys())
#                 sampled_leads = random.sample(all_lead_names, l)
#                 sampled_indices = [available_lead_mapping[lead] for lead in sampled_leads]
#             else:
#                 # Only use lead names that actually exist in this ECG
#                 sampled_leads = [lead_name for lead_name in self.sampling if lead_name in available_lead_mapping]
#                 sampled_indices = [available_lead_mapping[lead_name] for lead_name in sampled_leads]
                
#                 # Fallback: if no specified leads exist, use all available
#                 if not sampled_leads:
#                     sampled_leads = list(available_lead_mapping.keys())
#                     sampled_indices = list(available_lead_mapping.values())
            

#             ecg_data = ecg_data[sampled_indices] # L, time_steps 
            
#             # --- downsampling ---
#             if self.downsampling_factor is not None:
#                 ecg_data = signal.decimate(ecg_data, self.downsampling_factor)
                

#             ecg_data = torch.from_numpy(ecg_data.copy()).float() # L, time_steps
            
#             assert torch.isfinite(ecg_data).all(), f"Non-finite values found in ecg data for file {ecg_filename}"

#             if self.create_edge_index:
#                 edge_index = self._edge_index_cache[len(sampled_leads), self.edge_strategy]
#             else:
#                 edge_index = torch.empty((2,0), dtype=torch.long)


#             node_lead_ids = torch.empty((0,), dtype=torch.long)
#             node_temporal_pos = torch.empty((0,), dtype=torch.long)
            
#             if self.use_lead_encoding:
#                 # --- create node lead ids: to tell which lead a node belongs to, for each node ---
#                 # Use the actual lead names from sampled_leads to create consistent IDs
#                 node_lead_ids_list = []
#                 for lead_name in sampled_leads:
#                     # Try to get standard ID from LEAD_2_INDEX, otherwise use position
#                     if lead_name in LEAD_2_INDEX:
#                         lead_id = LEAD_2_INDEX[lead_name]
#                     else:
#                         # For unknown lead names, use the index in sampled_leads
#                         lead_id = sampled_leads.index(lead_name)
#                     node_lead_ids_list.extend([lead_id] * self.num_subgraph_nodes)
                
#                 node_lead_ids = torch.tensor(node_lead_ids_list, dtype=torch.long) # L*S
                
#             if self.use_temporal_encoding:
#                 # --- create node temporal pos: to tell which temporal position nodes occupy in a lead, for each lead ---
#                 # --- example of expected output: [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6] if 2 leads are sampled and S=7 is the number of nodes per lead ---
#                 node_temporal_pos = torch.tensor([list(range(self.num_subgraph_nodes)) for _ in range(len(sampled_leads))], dtype=torch.long).flatten() # L, S -> flatten to L*S
            
#             if self.pretrain:

#                 # --- fetch and index L features ---

#                 lead_features = np.load(
#                     os.path.join(
#                         self.features_path,
#                         os.path.basename(ecg_filename)[:-4] + "_grouped.npy"),
#                     mmap_mode='r'
#                     )[sampled_indices] # loads num_leads, S, D -> slices L, S, D
                
#                 # --- compute the pseudo-labels aka cluster assignments --- 
                
#                 lead_features = lead_features.reshape(-1, lead_features.shape[-1]) # L*S, D
#                 lead_features = self.normalizer.transform(lead_features) # divide each feature vector by its L2 norm
                
#                 y = torch.from_numpy(lead_features).float() # L*S, D
                
#             else: # finetuning

#                 y = record[self.diagnoses_cols].values.astype(float if len(self.diagnoses_cols) > 1 else int)
#                 y = torch.from_numpy(y).float() if len(self.diagnoses_cols) > 1 else torch.from_numpy(y).long()
                
            
#             return torch_geometric.data.Data(
#                 x = ecg_data, # L, time_steps
#                 y = y, # L*S, num_taks if pretrain else (num_labels,) if finetuning
#                 edge_index = edge_index, # (2, E)
#                 node_lead_ids = node_lead_ids, # (L*S,)
#                 node_temporal_pos = node_temporal_pos, # (L*S, )            
#             )


#     def collate_fn(self, batch : List[torch_geometric.data.Data]) -> torch_geometric.data.Data:
#         """
#         Custom collate function to handle batches of ECGGraphDataset items. Class method to access self.num_subgraph_nodes.
#         batch is a List of `batch_size` torch_geometric.data.Data objects, each containing:
#         x: (L_i, time_steps) (NOTE: x is not the matrix of node feature vectors)
#         y: (L_i*S, num_tasks) if pretrain else (num_labels,) if finetuning
#         edge_index: (2, E) or (2, 0) if no edges
#         node_lead_ids: (L_i*S,) or empty tensors
#         node_temporal_pos: (L_i*S, ) or empty tensors
        
#         Returns a batched torch_geometric Data object where:
#         x are concatenated along the first dimension: (sum_{i=1}^B L_i, time_steps)
#         y are concatenated along the first dimension: (sum_{i=1}^B L_i*S, num_tasks) if pretrain else (B * num_labels) if finetuning
#         edge_index are shifted by the cumulative sum of L_i*S to account for the new node indices in the batch. Final shape is (2, sum_{i=1}^{B} E_i)
#         node_lead_ids are concatenated along the first dimension (sum_{i=1}^B L_i*S,)
#         node_temporal_pos are concatenated along the first dimension (sum_{i=1}^B L_i*S, )        
#         A batch.batch is also created to indicate the graph to which each node belongs (sum_{i=1}^B L_i*S,)
#         """
        
#         x_list = [data.x for data in batch]
#         y_list = [data.y for data in batch]
#         edge_index_list = [data.edge_index for data in batch]
#         node_lead_ids_list = [data.node_lead_ids for data in batch]
#         node_temporal_pos_list = [data.node_temporal_pos for data in batch]
        
#         x = torch.cat(x_list, dim=0) # (sum_{i=1}^B L_i, time_steps)
#         y = torch.cat(y_list, dim=0) # (sum_{i=1}^B L_i*S, num_tasks) if pretrain else (B, num_labels)
#         node_lead_ids = torch.cat(node_lead_ids_list, dim=0) # (sum_{i=1}^B L_i*S,)
#         node_temporal_pos = torch.cat(node_temporal_pos_list, dim=0) # (sum_{i=1}^B L_i*S, )
        
#         # shift edge indices
#         edge_index = []
#         cumsum_nodes = 0
#         for edge_index_i in edge_index_list:
#             edge_index.append(edge_index_i + cumsum_nodes)
#             cumsum_nodes += edge_index_i.max().item() + 1 if edge_index_i.numel() > 0 else 0 
#         edge_index = torch.cat(edge_index, dim=1) if len(edge_index) > 0 else torch.empty((2,0), dtype=torch.long) # (2, sum_{i=1}^{B} E_i)
        
#         batch_vector = []
#         cumsum_nodes = 0
#         for i, data in enumerate(batch):
#             num_nodes = data.x.size(0) # L_i
#             num_nodes = num_nodes * self.num_subgraph_nodes
#             batch_vector.extend([i] * num_nodes)
#             cumsum_nodes += num_nodes
#         batch_vector = torch.LongTensor(batch_vector) # (sum_{i=1}^B L_i*S,)
        
#         return torch_geometric.data.Data(
#             x = x,
#             y = y,              
#             edge_index = edge_index,
#             node_lead_ids = node_lead_ids,
#             node_temporal_pos = node_temporal_pos,
#             batch = batch_vector,
#         )


######################## new implementation for integrating with benchmarking ##########

import os
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from sqlalchemy import all_
import torch
import numpy as np

# Force PyTorch intra-op parallelism to 1
torch.set_num_threads(1)

import threadpoolctl

import numpy as np
from loguru import logger
import pandas as pd
import torch

from torch.utils.data import Dataset
from scipy import signal
import random
from typing import List, Union, Optional, Any
from consts import *
from sklearn import preprocessing
import torch_geometric
from consts import LEADS
from torch.utils.data import DataLoader
import glob

def multihot_encode(x, num_classes):
    res = np.zeros(num_classes,dtype=np.float32)
    for y in x:
        res[y]=1
    return res

class ECGDataset(Dataset):
    
    @staticmethod
    def get_benchmarking_splits(ecg_dataframe: pd.DataFrame) -> tuple:
        if 'strat_fold' not in ecg_dataframe.columns:
            raise ValueError("Dataframe must contain 'strat_fold' column for benchmarking splits.")
        
        max_fold = ecg_dataframe['strat_fold'].max()
        logger.info(f"Creating splits using strat_fold (max={max_fold})")

        folds = ecg_dataframe['strat_fold'].to_numpy()
        all_pos = np.arange(len(ecg_dataframe), dtype=np.int64)

        train_indices = all_pos[folds < (max_fold - 1)].tolist()
        val_indices = all_pos[folds == (max_fold - 1)].tolist()
        test_indices = all_pos[folds == max_fold].tolist()

        assert len(set(train_indices) & set(val_indices)) == 0, "Train and Val sets overlap!"
        assert len(set(train_indices) & set(test_indices)) == 0, "Train and Test sets overlap!"
        assert len(set(val_indices) & set(test_indices)) == 0, "Val and Test sets overlap!"

        total = len(train_indices) + len(val_indices) + len(test_indices)
        assert total == len(ecg_dataframe), f"Total indices ({total}) does not match dataframe length ({len(ecg_dataframe)})"

        return train_indices, val_indices, test_indices

    def __init__(
        self,
        # === Mode selection ===
        use_benchmarking_format: bool = False,
        
        # === Mode 1: Original format parameters ===
        path_to_dataset_csv: Optional[str] = None,
        ecg_dir_path: Optional[str] = None,
        label_start_index: int = 3,
        
        # === Mode 2: Benchmarking format parameters ===
        benchmarking_folder: Optional[str] = None,
        label_column: Optional[str] = None,
        
        # === Common parameters ===
        downsampling_factor: Optional[int] = None,
        features_path: Optional[str] = None,
        nodes_per_lead: int = 20,
        pretrain: bool = False,
        random_crop: bool = False,
        return_full_length: bool = False,
        create_edge_index: bool = True,
        edge_strategy: str = "spatiotemporal",
        use_temporal_encoding: bool = False,
        use_lead_encoding: bool = False,
        sampling: Union[str, List[str]] = 'random',
        how_many_leads: Optional[int] = None,
        fixed_lead_seed: Optional[int] = None,
        pad_unsampled_leads: bool = False,
    ):
        """
        ECGDataset for loading ECG data in graph neural network format.
        
        This dataset class operates in two modes:
        
        **Mode 1: Original Format** (use_benchmarking_format=False, default)
            - Loads data from CSV file + individual .npy files
            - Suitable for custom datasets
            - Required: path_to_dataset_csv, ecg_dir_path
        
        **Mode 2: Benchmarking Format** (use_benchmarking_format=True)
            - Loads preprocessed data from the benchmark's clinical_ts memmap format
            - Uses memory-mapped files for efficiency
            - Required: benchmarking_folder
            - Expects folder structure:
                - df_memmap.pkl or df.pkl (dataframe)
                - memmap_0_preprocessed.npy (preprocessed ECG data)
                - memmap_meta_preprocessed.npz (metadata)
                - lbl_itos.npy (label mapping) or lbl_itos.pkl
                - label_column (name of the column in the dataframe to use for labels in pkl format)
        
        Parameters
        ----------
        use_benchmarking_format : bool, default=False
            If True, use benchmarking format (Mode 2). If False, use original format (Mode 1).
        
        **Mode 1 Parameters (Original Format):**
        
        path_to_dataset_csv : str, optional
            Path to CSV file containing dataset metadata and labels.
            Required when use_benchmarking_format=False.
            For multi-label: CSV should have binary columns (one per label).
            For multi-class: CSV should have one column with class indices [0, n_classes-1].
        
        ecg_dir_path : str, optional
            Path to directory containing individual ECG .npy files.
            Required when use_benchmarking_format=False.
        
        label_start_index : int, default=3
            Column index in CSV where labels start (Mode 1 only).
            Used only when pretrain=False.
        
        **Mode 2 Parameters (Benchmarking Format):**
        
        benchmarking_folder : Path or str, optional
            Path to preprocessed benchmarking dataset folder.
            Required when use_benchmarking_format=True.
            Should contain: df_memmap.pkl, memmap_0_preprocessed.npy, 
            memmap_meta_preprocessed.npz, lbl_itos.npy
        
        **Common Parameters:**
        
        downsampling_factor : int, optional
            Downsampling factor to apply to ECG signal.
            If None, no downsampling is applied.
            Note: Not needed if using preprocessed benchmarking data at target frequency.
        
        features_path : str, optional
            Path to directory containing precomputed features for pretraining.
            Required when pretrain=True.
            Features should be stored as {filename}_grouped.npy files.
        
        nodes_per_lead : int, default=20
            Number of graph nodes created per ECG lead.
            Each lead is represented as a subgraph with this many nodes.
        
        pretrain : bool, default=False
            If True, dataset is used for self-supervised pretraining.
            If False, dataset is used for supervised fine-tuning.
        
        random_crop : bool, default=False
            If True, randomly crop ECG signals during loading.
            Useful for data augmentation during training.
        
        return_full_length : bool, default=False
            If True, return 10-second ECG segments (if available).
            If False, return 5-second segments.
        
        create_edge_index : bool, default=True
            If True, create graph edge indices based on edge_strategy.
            Set to False for MLP-based models that don't use graph structure.
        
        edge_strategy : str, default="spatiotemporal"
            Graph topology strategy. Options:
            - "spatiotemporal": Fully connected within leads + temporal connections across leads
            - "fully_connected": Fully connected graph across all nodes
        
        use_temporal_encoding : bool, default=False
            If True, add temporal position encodings to graph nodes.
            Useful for models that utilize temporal structure.
        
        use_lead_encoding : bool, default=False
            If True, add lead ID encodings to graph nodes.
            Useful for models that utilize spatial (lead) structure.
        
        sampling : str or list of str, default='random'
            Lead sampling strategy:
            - 'random': Randomly sample leads (optionally controlled by how_many_leads)
            - List of lead names: Use specific leads (e.g., ['I', 'II', 'V1', 'V2'])
            Valid lead names: 'I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1'-'V6'
        
        how_many_leads : int, optional
            Number of leads to randomly sample when sampling='random'.
            If None, samples a random number between 1 and available leads.

        fixed_lead_seed : int, optional
            If set and sampling='random', the lead subset of record `idx` is drawn from an RNG
            seeded with (fixed_lead_seed, idx), so every epoch, worker and run sees the same
            leads for the same record. Intended for validation/test sets. If None, leads are
            drawn from the global `random` state (original behaviour).

        pad_unsampled_leads : bool, optional
            Zero-padding counterpart of lead sampling (cf. random lead masking, Oh et al.): the
            leads are sampled exactly as above, but the unsampled ones are KEPT as all-zero
            signals, so every graph has all available leads (always the full 12-lead topology).
            `real_node_mask` marks the nodes of the sampled leads; masked node modelling only
            masks (and scores) those. Default False (unsampled leads are dropped from the graph).

        Returns (from __getitem__)
        ---------------------------
        torch_geometric.data.Data object with:
            x : Tensor, shape (L, time_steps)
                ECG signal data for L sampled leads
            y : Tensor
                Labels (shape depends on pretrain mode)
                - Pretraining: (L*S, D) where S=nodes_per_lead, D=feature_dim
                - Fine-tuning: (num_labels,) for multi-label or (1,) for multi-class
            edge_index : LongTensor, shape (2, num_edges)
                Graph edge indices in COO format
            node_lead_ids : LongTensor, shape (L*S,)
                Lead ID for each node (if use_lead_encoding=True)
            node_temporal_pos : LongTensor, shape (L*S,)
                Temporal position for each node (if use_temporal_encoding=True)
            batch : LongTensor, shape (L*S,)
                Batch assignment for each node (added by collate_fn)
        
        Examples
        --------
        Mode 1 - Original format:
        >>> dataset = ECGDataset(
        ...     use_benchmarking_format=False,
        ...     path_to_dataset_csv="data/train.csv",
        ...     ecg_dir_path="data/ecg_files/",
        ...     label_start_index=3,
        ...     nodes_per_lead=20,
        ...     pretrain=False
        ... )
        
        Mode 2 - Benchmarking format:
        >>> dataset = ECGDataset(
        ...     use_benchmarking_format=True,
        ...     benchmarking_folder="/data/ptb_preprocessed",
        ...     nodes_per_lead=20,
        ...     pretrain=False
        ... )
        
        With custom lead sampling:
        >>> dataset = ECGDataset(
        ...     use_benchmarking_format=True,
        ...     benchmarking_folder="/data/ptb_preprocessed",
        ...     sampling=['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
        ...     nodes_per_lead=20
        ... )
        
        Notes
        -----
        - When use_benchmarking_format=True, path_to_dataset_csv and ecg_dir_path 
          can be set to dummy values (e.g., "dummy") as they are not used.
        - Preprocessed benchmarking data should already be filtered, resampled, 
          and scaled to [-1, 1] range.
        - The dataset automatically handles different numbers of leads across samples
          when sampling='random'.
        """
        
        self.use_benchmarking_format = use_benchmarking_format
        
        # # take only the first 64 instances for debugging
        # logger.warning("DEBUG MODE: taking only the first 64 instances of the dataset")
        # self.ecg_dataframe = self.ecg_dataframe.iloc[:64]
        
        self.downsampling_factor = downsampling_factor
        self.pretrain = pretrain
        self.random_crop = random_crop
        self.return_full_length = return_full_length
        self.create_edge_index = create_edge_index 
        self.use_temporal_encoding = use_temporal_encoding
        self.use_lead_encoding = use_lead_encoding     
        self.num_subgraph_nodes = nodes_per_lead
        
        if edge_strategy not in ["fully_connected", "spatiotemporal"]:
            raise ValueError(f"edge_strategy must be one of {['fully_connected', 'spatiotemporal']} but got {edge_strategy}")
        
        self.edge_strategy = edge_strategy
        
        if type(sampling) != str and type(sampling) != list:
            raise ValueError(f"type of sampling must be either str or list, got {type(sampling)}")
        
        if type(sampling) == str:
            if sampling != 'random':
                raise NotImplementedError(f"when sampling is a string, not other value than 'random' is possible. Got {sampling}")
        else:
            if not set(LEAD_2_INDEX.keys()).issuperset(set(sampling)):
                raise ValueError(f"Provided sampling list must contain valid lead names including {LEAD_2_INDEX.keys()}. Got {sampling}")

        self.sampling = sampling
        self.how_many_leads = how_many_leads
        self.fixed_lead_seed = fixed_lead_seed
        self.pad_unsampled_leads = pad_unsampled_leads

        self._edge_index_cache = {} # memoization for edge_index
        self.precompile_edge_index()
        
        # === MODE 1 ===
        
        if not use_benchmarking_format:
            logger.info(f"Loading dataset from {path_to_dataset_csv}...")
            
            self.ecg_dataframe = pd.read_csv(path_to_dataset_csv, dtype={'filename': str})
            
            self.ecg_dir_path = ecg_dir_path

            if pretrain:            
                self.features_path = features_path
                self.normalizer = preprocessing.Normalizer() # no need to fit this
            else:
                self.diagnoses_cols = self.ecg_dataframe.columns.values.tolist()[label_start_index:]
                assert len(self.diagnoses_cols) > 0, "No labels found in the dataset"
                self.weights = self.compute_weights()
        
        else: # === MODE 2 ===
            
            logger.info(f"Loading dataset from benchmarking folder {benchmarking_folder}...")
            
            if benchmarking_folder is None:
                raise ValueError("benchmarking_folder must be provided when use_benchmarking_format is True")
            
            self.benchmarking_folder = benchmarking_folder
            self.label_column = label_column
            
            # load dataframe 
            
            if os.path.exists(os.path.join(self.benchmarking_folder, "df_memmap.pkl")):
                logger.info("Loading df_memmap.pkl...")
                with open(os.path.join(self.benchmarking_folder, "df_memmap.pkl"), "rb") as f:
                    self.ecg_dataframe = pd.read_pickle(f)
            elif os.path.exists(os.path.join(self.benchmarking_folder, "df.pkl")):
                logger.info("Loading df.pkl...")
                with open(os.path.join(self.benchmarking_folder, "df.pkl"), "rb") as f:
                    self.ecg_dataframe = pd.read_pickle(f)
            else:
                raise FileNotFoundError("Neither df_memmap.pkl nor df.pkl found in benchmarking_folder")
            
            # load label mapping
            try:
                self.lbl_itos = np.load(os.path.join(self.benchmarking_folder, "lbl_itos.npy"), allow_pickle=True)
                logger.info(f"Loaded label mapping with {len(self.lbl_itos)} classes: {list(self.lbl_itos)}")
                
                if 'label' not in self.ecg_dataframe.columns:
                    raise ValueError("Expected 'label' column in dataframe for benchmarking format, but not found.")
                self.ecg_dataframe['label'] = self.ecg_dataframe['label'].apply(lambda x: multihot_encode(x, num_classes=len(self.lbl_itos)))
                
            except Exception as e:
                self.lbl_itos = np.load(os.path.join(self.benchmarking_folder, "lbl_itos.pkl"), allow_pickle=True)
                logger.info(f"Loaded label mapping from .pkl with {len(self.lbl_itos[self.label_column])} classes: {list(self.lbl_itos[self.label_column])}")
                
                candidate_label_for_multihot = self.label_column + "_numeric"
                
                if candidate_label_for_multihot in self.ecg_dataframe.columns:
                    self.ecg_dataframe['label'] = self.ecg_dataframe[candidate_label_for_multihot].apply(lambda x: multihot_encode(x, num_classes=len(self.lbl_itos[self.label_column])))
                elif 'labels' in self.ecg_dataframe.columns:
                    self.ecg_dataframe['label'] = self.ecg_dataframe['labels'].apply(lambda x: multihot_encode(x, num_classes=len(self.lbl_itos[self.label_column])))
                else:
                    raise ValueError(f"No suitable column found for labels in dataframe. Expected 'label', '{self.label_column}_numeric', or 'labels'.")
                    
            
            
            # self.mean = np.load(os.path.join(self.benchmarking_folder, "mean.npy"))
            # self.std = np.load(os.path.join(self.benchmarking_folder, "std.npy"))
            
            memmap_meta_path = None
            if os.path.exists(os.path.join(self.benchmarking_folder, "memmap_meta_preprocessed.npz")):
                memmap_meta_path = os.path.join(self.benchmarking_folder, "memmap_meta_preprocessed.npz")
            elif os.path.exists(os.path.join(self.benchmarking_folder, "memmap_meta.npz")):
                memmap_meta_path = os.path.join(self.benchmarking_folder, "memmap_meta.npz")
            else:
                raise FileNotFoundError("Neither memmap_meta_preprocessed.npz nor memmap_meta.npz found in benchmarking_folder")
            
            logger.info(f"Loading memmap metadata from {memmap_meta_path}...")
            memmap_meta_file = np.load(memmap_meta_path, allow_pickle=True)
            
            self.memmap_starts = memmap_meta_file['start'].copy()
            self.memmap_lengths = memmap_meta_file['length'].copy()
            self.memmap_shape = memmap_meta_file['shape'].copy()
            self.memmap_dtype = memmap_meta_file['dtype'].copy()
            
            logger.info(f"Loaded memmap metadata with {len(self.memmap_starts)} entries.")
            
            memmap_meta_file.close()
            del memmap_meta_file
            
            self.memmap_files = sorted(glob.glob(os.path.join(self.benchmarking_folder, "memmap_*_preprocessed.npy")))
            if len(self.memmap_files) == 0:
                raise FileNotFoundError("No memmap_*_preprocessed.npy files found in benchmarking_folder")
            logger.info(f"Found {len(self.memmap_files)} memmap files.")
            
            
            self.memmap_data = None
            
            if pretrain:
                self.features_path = features_path
                self.normalizer = preprocessing.Normalizer() # no need to fit this
            else:
                self.weights = self.compute_weights_benchmarking()
   
    def get_splits(self) -> tuple:
        if not self.use_benchmarking_format:
            raise ValueError("get_splits() only works in benchmarking mode")
        return self.get_benchmarking_splits(self.ecg_dataframe) 
                 
    def _lazy_load_memmap(self):
        if self.memmap_data is None:
            logger.info("Loading memmap data into memory...")
            memmap_path = self.memmap_files[0]
            
            try:
                self.memmap_data = np.load(str(memmap_path), mmap_mode='r', allow_pickle=True)
                logger.info(f"Loaded memmap with shape: {self.memmap_data.shape}")
            except Exception as e:
                logger.warning(f"Direct load failed: {e}, trying with shape and dtype from extracted metadata")
                
                if self.memmap_shape is not None and self.memmap_dtype is not None:
                    shape = tuple(int(x) for x in self.memmap_shape.ravel())
                    dtype = str(self.memmap_dtype)
                    
                    # Validate shape vs file size
                    file_size = os.path.getsize(memmap_path)
                    expected_size = np.prod(shape) * np.dtype(dtype).itemsize
                    
                    if expected_size != file_size:
                        logger.warning(f"Shape mismatch: expected {expected_size} bytes, got {file_size} bytes")
                        # Infer correct shape
                        total_elements = file_size // np.dtype(dtype).itemsize
                        total_timepoints = int(self.memmap_starts[-1] + self.memmap_lengths[-1])
                        num_leads = total_elements // total_timepoints
                        shape = (total_timepoints, num_leads)
                        logger.info(f"Inferred shape: {shape}")
                    
                    self.memmap_data = np.memmap(str(memmap_path), mode='r', dtype=dtype, shape=shape)
                    logger.info(f"Loaded memmap with shape: {self.memmap_data.shape}")
                else:
                    raise ValueError("Cannot load memmap: missing shape/dtype in extracted metadata")
                
    def compute_weights_benchmarking(self):
        logger.info("Computing weights for benchmarking format...")        

        all_labels = np.stack(self.ecg_dataframe['label'].values) # (N, num_classes)
        label_counts = all_labels.sum(axis=0) # (num_classes,)
        
        total_samples = len(self.ecg_dataframe)
        weights = (total_samples - label_counts) / (label_counts + 1e-9) # (num_classes,)
        
        logger.info("Done with the weights.")
        return torch.FloatTensor(weights)
               
    def compute_weights(self):
        logger.info("Computing weights...")        
        if len(self.diagnoses_cols) > 1:
            weights = []
            for label in self.diagnoses_cols:
                count = self.ecg_dataframe[label].sum()
                weight = (self.ecg_dataframe.__len__() - count) / (count + 1e-9)
                weights.append(weight)
            logger.info("Done with the weights.")
        else:
            num_labels = self.ecg_dataframe[self.diagnoses_cols[0]].max() + 1
            weights = num_labels / self.ecg_dataframe[self.diagnoses_cols].value_counts()
            weights = weights.values.tolist()
        return torch.FloatTensor(weights)

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
    
        with threadpoolctl.threadpool_limits(limits=1, user_api='blas'):
            
            record = self.ecg_dataframe.iloc[idx]
            
            if self.use_benchmarking_format:
            
                self._lazy_load_memmap()
                
                memmap_idx = int(record['data'])
                start = int(self.memmap_starts[memmap_idx])
                length = int(self.memmap_lengths[memmap_idx])

                ecg_data = self.memmap_data[start:start + length].copy()
                ecg_data = ecg_data.T # (num_leads, time_steps)
                
            else:
                ecg_filename = record['filename'] # the mere filename

                ecg_path = ecg_filename if os.path.isfile(ecg_filename) else os.path.join(self.ecg_dir_path, ecg_filename)

                ecg_data = np.load(ecg_path, mmap_mode='r') # (num_leads, 5000+)
                
            if self.pretrain:
                ecg_data = ecg_data[:, :SAMPLES_IN_5_SECONDS_AT_500HZ]
            elif self.random_crop: 
                start = torch.randint(0, ecg_data.shape[1] - SAMPLES_IN_5_SECONDS_AT_500HZ + 1, (1,)).item()
                ecg_data = ecg_data[:, start:start+SAMPLES_IN_5_SECONDS_AT_500HZ]
            elif self.return_full_length:
                # we take a random crop of 10 seconds because it's the largest and most commong window in the literature
                assert ecg_data.shape[1] >= SAMPLES_IN_10_SECONDS_AT_500HZ, f"ecg length is less than 10 seconds for file {ecg_filename}"
                start = torch.randint(0, ecg_data.shape[1] - SAMPLES_IN_10_SECONDS_AT_500HZ + 1, (1,)).item() 
                ecg_data = ecg_data[:, start:start+SAMPLES_IN_10_SECONDS_AT_500HZ]
            else:
                ecg_data = ecg_data[:, :SAMPLES_IN_5_SECONDS_AT_500HZ]

            # --- random lead sampling ----  
            
            num_available_leads = ecg_data.shape[0]    
            
            # Create a mapping from lead names to actual indices for this ECG to handle ECGs with arbitrary numbers of leads
            available_lead_mapping = {INDEX_2_LEAD.get(i, f"Lead_{i}"): i for i in range(num_available_leads)}
            
            if self.sampling == "random":
                # per-record RNG makes the draw independent of epoch, worker and run
                rng = random.Random(self.fixed_lead_seed * 1_000_003 + idx) if self.fixed_lead_seed is not None else None

                if self.how_many_leads is not None:
                    l = min(self.how_many_leads, num_available_leads)
                elif rng is not None:
                    l = rng.randint(1, num_available_leads)
                else:
                    l = torch.randint(1, num_available_leads + 1, (1,)).item()

                all_lead_names = list(available_lead_mapping.keys())
                sampled_leads = rng.sample(all_lead_names, l) if rng is not None else random.sample(all_lead_names, l)
                sampled_indices = [available_lead_mapping[lead] for lead in sampled_leads]
            else:
                # Only use lead names that actually exist in this ECG
                sampled_leads = [lead_name for lead_name in self.sampling if lead_name in available_lead_mapping]
                sampled_indices = [available_lead_mapping[lead_name] for lead_name in sampled_leads]
                
                # Fallback: if no specified leads exist, use all available
                if not sampled_leads:
                    sampled_leads = list(available_lead_mapping.keys())
                    sampled_indices = list(available_lead_mapping.values())
            

            if self.pad_unsampled_leads:
                # keep every lead in index order, zero the unsampled ones
                real_leads = set(sampled_indices)
                sampled_indices = list(range(num_available_leads))
                sampled_leads = [INDEX_2_LEAD.get(i, f"Lead_{i}") for i in sampled_indices]
                is_real_lead = np.array([i in real_leads for i in sampled_indices])
                ecg_data = np.array(ecg_data[sampled_indices], dtype=float)
                ecg_data[~is_real_lead] = 0.0
            else:
                is_real_lead = np.ones(len(sampled_indices), dtype=bool)
                ecg_data = ecg_data[sampled_indices] # L, time_steps 
            real_node_mask = torch.from_numpy(np.repeat(is_real_lead, self.num_subgraph_nodes)) # (L*S,) bool
            
            # --- downsampling ---
            if self.downsampling_factor is not None:
                ecg_data = signal.decimate(ecg_data, self.downsampling_factor)
                

            ecg_data = torch.from_numpy(ecg_data.copy()).float() # L, time_steps
            
            assert torch.isfinite(ecg_data).all(), f"Non-finite values found in ecg data for file {ecg_filename}"

            if self.create_edge_index:
                edge_index = self._edge_index_cache[len(sampled_leads), self.edge_strategy]
            else:
                edge_index = torch.empty((2,0), dtype=torch.long)


            node_lead_ids = torch.empty((0,), dtype=torch.long)
            node_temporal_pos = torch.empty((0,), dtype=torch.long)
            
            if self.use_lead_encoding:
                # --- create node lead ids: to tell which lead a node belongs to, for each node ---
                # Use the actual lead names from sampled_leads to create consistent IDs
                node_lead_ids_list = []
                for lead_name in sampled_leads:
                    # Try to get standard ID from LEAD_2_INDEX, otherwise use position
                    if lead_name in LEAD_2_INDEX:
                        lead_id = LEAD_2_INDEX[lead_name]
                    else:
                        # For unknown lead names, use the index in sampled_leads
                        lead_id = sampled_leads.index(lead_name)
                    node_lead_ids_list.extend([lead_id] * self.num_subgraph_nodes)
                
                node_lead_ids = torch.tensor(node_lead_ids_list, dtype=torch.long) # L*S
                
            if self.use_temporal_encoding:
                # --- create node temporal pos: to tell which temporal position nodes occupy in a lead, for each lead ---
                # --- example of expected output: [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6] if 2 leads are sampled and S=7 is the number of nodes per lead ---
                node_temporal_pos = torch.tensor([list(range(self.num_subgraph_nodes)) for _ in range(len(sampled_leads))], dtype=torch.long).flatten() # L, S -> flatten to L*S
            
            if self.pretrain:

                # --- fetch and index L features ---

                lead_features = np.load(
                    os.path.join(
                        self.features_path,
                        os.path.basename(ecg_filename)[:-4] + "_grouped.npy"),
                    mmap_mode='r'
                    )[sampled_indices] # loads num_leads, S, D -> slices L, S, D
                
                # --- compute the pseudo-labels aka cluster assignments --- 
                
                lead_features = lead_features.reshape(-1, lead_features.shape[-1]) # L*S, D
                lead_features = self.normalizer.transform(lead_features) # divide each feature vector by its L2 norm
                
                y = torch.from_numpy(lead_features).float() # L*S, D
                
            else: # finetuning
                
                if self.use_benchmarking_format:
                    y = record['label'] # this should already be a multihot encoded vector if the dataset was prepared correctly
                    y = torch.from_numpy(y).float() # num_labels
                    
                else:

                    y = record[self.diagnoses_cols].values.astype(float if len(self.diagnoses_cols) > 1 else int)
                    y = torch.from_numpy(y).float() if len(self.diagnoses_cols) > 1 else torch.from_numpy(y).long()
                
            
            return torch_geometric.data.Data(
                x = ecg_data, # L, time_steps
                y = y, # L*S, num_taks if pretrain else (num_labels,) if finetuning
                edge_index = edge_index, # (2, E)
                node_lead_ids = node_lead_ids, # (L*S,)
                node_temporal_pos = node_temporal_pos, # (L*S, )            
                real_node_mask = real_node_mask, # (L*S,) True = node of a sampled (non-padded) lead
            )


    def collate_fn(self, batch : List[torch_geometric.data.Data]) -> torch_geometric.data.Data:
        """
        Custom collate function to handle batches of ECGGraphDataset items. Class method to access self.num_subgraph_nodes.
        batch is a List of `batch_size` torch_geometric.data.Data objects, each containing:
        x: (L_i, time_steps) (NOTE: x is not the matrix of node feature vectors)
        y: (L_i*S, num_tasks) if pretrain else (num_labels,) if finetuning
        edge_index: (2, E) or (2, 0) if no edges
        node_lead_ids: (L_i*S,) or empty tensors
        node_temporal_pos: (L_i*S, ) or empty tensors
        
        Returns a batched torch_geometric Data object where:
        x are concatenated along the first dimension: (sum_{i=1}^B L_i, time_steps)
        y are concatenated along the first dimension: (sum_{i=1}^B L_i*S, num_tasks) if pretrain else (B * num_labels) if finetuning
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
        y = torch.cat(y_list, dim=0) # (sum_{i=1}^B L_i*S, num_tasks) if pretrain else (B, num_labels)
        node_lead_ids = torch.cat(node_lead_ids_list, dim=0) # (sum_{i=1}^B L_i*S,)
        node_temporal_pos = torch.cat(node_temporal_pos_list, dim=0) # (sum_{i=1}^B L_i*S, )
        real_node_mask = torch.cat([data.real_node_mask for data in batch], dim=0) # (sum_{i=1}^B L_i*S,)
        
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
            real_node_mask = real_node_mask,
            batch = batch_vector,
        )