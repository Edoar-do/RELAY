from math import e
import torch
from torch import nn
from typing import List, Union, Optional, Tuple, Callable, Any, Dict
from torch_geometric import nn as nng
# from torch_geometric.typing import SparseTensor
import torch_geometric
from torchmetrics import Metric
from transformers import PreTrainedModel
from graph_modelling_output import GraphClassificationOutput, MaskedNodeModellingOuputWithAttention, MaskedNodeModellingOutput, GraphClassificationOutputWithAttention
from config import ECGGATConfig, ECGGINConfig, ECGGraphTransformerConfig, ECGMLPConfig, ECGGCNConfig
from models.gat import GAT
from models.gin import GIN
from models.mlp import MLP
from models.gcn import GCN
from models.graph_transformer import GraphTransformer
from models.modules.spatiotemporal_encoding import InterLeadSpatialEncoding, IntraLeadTemporalEncoding
from models.modules.granola import GranolaNorm
from transformers.models.hubert.modeling_hubert import HubertFeatureEncoder
import copy
# --- Base Classes ---

# --- Masked node modelling ---

class MaskedNodeModellingBase(PreTrainedModel):
    def __init__(self, config, encoder_cls):
        super().__init__(config)

        if config.mask_node_prob is None:
            raise ValueError(f"Argument mask_node_prob cannot be None for masked node modelling")
        if config.pseudo_vocab_sizes is None:
            raise ValueError(f"Argument pseudo_vocab_sizes cannot be None for masked node modelling")
        if config.pseudo_classifier_projection_size is None:
            raise ValueError(f"Argument pseudo_classifier_projection_size cannot be None for masked node modelling")
        
        assert config.in_channels == config.hidden_channels, \
            f"Currently, in_channels must be equal to hidden_channels. Found {config.in_channels} and {config.hidden_channels}"
        
        self.embedder = HubertFeatureEncoder(config)
        
        if config.conv_dim[-1] != config.in_channels:
            self.feature_projection = nn.Sequential(
                nn.Linear(config.conv_dim[-1], config.in_channels, bias=False),
                nn.LayerNorm(config.in_channels)
            )
        else:
            self.feature_projection = nn.Identity()
        
        if config.use_lead_encoding:
            self.lead_encoding = InterLeadSpatialEncoding(config)
        
        if config.use_temporal_encoding:
            self.temporal_encoding = IntraLeadTemporalEncoding(config)
        
        self.encoder = encoder_cls(config)

        self.norm_post_encoder = copy.deepcopy(self.encoder.encoder.norms[0])

        self.mask_embed = nn.Parameter(torch.zeros(config.in_channels))
        
        if config.use_pseudo_label_embeddings:
            self.pseudo_classifier_projections = nn.ModuleList(
                [nn.Linear(config.out_channels if config.out_channels is not None else config.hidden_channels, config.pseudo_classifier_projection_size, bias=False) for _ in range(len(config.pseudo_vocab_sizes))]
            )
            self.pseudo_label_embeddings = nn.ModuleList(
                [nn.Embedding(pseudo_vocab_size, config.pseudo_classifier_projection_size) for pseudo_vocab_size in config.pseudo_vocab_sizes]
            )
            self.temperature = 0.1
        else:
            self.classifiers = nn.ModuleList(
                [nn.Linear(config.out_channels if config.out_channels is not None else config.hidden_channels, pseudo_vocab_size, bias=False) for pseudo_vocab_size in config.pseudo_vocab_sizes]
            )
    
    def _mask_node_features(self, x: torch.Tensor, batch : torch.Tensor) -> Tuple[torch.Tensor, torch.BoolTensor]:
        """
        Args:
            x: [total_nodes, in_channels]
            batch: [total_nodes] which node belongs to which graph in the batch
        Returns:
            x_masked: masked node features
            mask: [total_nodes] boolean mask indicating which nodes were masked
            
        Masking is performed per graph in the batch, ensuring that each graph has at least `min_node_masked` nodes masked.
        """
        mask = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
        if self.config.mask_node_prob > 0 and self.training:
            x_masked = x.clone()
            for graph_id in batch.unique():
                node_indices = (batch == graph_id).nonzero(as_tuple=True)[0]
                num_nodes = node_indices.size(0)
                num_to_mask = max(int(round(self.config.mask_node_prob * num_nodes)), self.config.min_node_masked)
                permuted = node_indices[torch.randperm(num_nodes)[:num_to_mask]]
                mask[permuted] = True
            x_masked[mask] = self.mask_embed.to(x.dtype)
            return x_masked, mask
        else:
            return x, mask # no masking applied during eval or if mask_node_prob is 0
        
    def _vectorized_mask_node_features(self, x: torch.Tensor, batch: torch.Tensor, real_node_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.BoolTensor]:
            """
            Vectorized masking of node features handling variable graph sizes in the batch.
            The total number of nodes per graph to mask = 
            real_node_mask (optional, bool [total_nodes]): only these nodes can be masked, and the
            count is computed from them, i.e. round(p_mask * real nodes of the graph). Used with
            zero-padded unsampled leads (ECGDataset(pad_unsampled_leads=True)): padded nodes are
            visible context only, never masked or scored. None = every node is eligible (default).
            """
    
            if not self.training or self.config.mask_node_prob == 0:
                mask = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
                return x, mask

            # 2. Calculate how many nodes to mask PER GRAPH
            # Count nodes in each graph in the batch
            nodes_per_graph = batch.bincount()
            eligible_per_graph = nodes_per_graph if real_node_mask is None else \
                batch[real_node_mask].bincount(minlength=nodes_per_graph.numel())
            
            # Calculate target mask count: K = max(round(N * prob), min_masked)
            # Using float math for the percentage, then clamping
            num_to_mask = (eligible_per_graph.float() * self.config.mask_node_prob).round().long()
            num_to_mask = torch.clamp(num_to_mask, min=self.config.min_node_masked)
            num_to_mask = torch.min(num_to_mask, eligible_per_graph) # Safety cap: don't mask more than exist

            # 3. Vectorized Random Sampling via Sorting
            # We want to pick 'num_to_mask' random nodes for each graph.
            # We generate random noise for every node.
            noise = torch.rand(x.size(0), device=x.device)
            if real_node_mask is not None:
                # eligible nodes in [0, 0.5), padded ones in [0.5, 1): within each graph every eligible
                # node ranks before every padded node, so only eligible nodes fall under the threshold
                noise = torch.where(real_node_mask, noise * 0.5, 0.5 + noise * 0.5)
            
            # We create a sort key that groups by batch but randomizes within the batch.
            # Since 'batch' is integers [0, 1, 2...] and noise is [0, 1),
            # adding them preserves the batch order but shuffles the internal node order.
            sort_key = batch.float() + noise
            
            # 'perm' is the list of node indices sorted by graph, then randomly
            perm = torch.argsort(sort_key)
            
            # 4. Determine which nodes in the sorted list to keep
            # We need the "rank" of each node within its graph to compare against 'num_to_mask'.
            # We calculate the starting index (ptr) of each graph in the batch.
            ptr = torch.cat([torch.zeros(1, device=x.device, dtype=torch.long), nodes_per_graph.cumsum(0)])
            
            # For every node in the permutation, we find which graph it belongs to
            graph_ids_in_perm = batch[perm]
            
            # The 'rank' is the node's position relative to the start of its graph
            # e.g. if Graph 0 has 5 nodes, the first 5 entries in perm have ranks 0,1,2,3,4.
            graph_starts = ptr[graph_ids_in_perm]
            ranks = torch.arange(x.size(0), device=x.device) - graph_starts
            
            # 5. Create the Mask
            # If a node's random rank is less than the number we need to mask for that graph, mask it.
            thresholds = num_to_mask[graph_ids_in_perm]
            is_masked_in_perm = ranks < thresholds
            
            # Map back to original indices: We select the indices from 'perm' that met the criteria
            indices_to_mask = perm[is_masked_in_perm]
            
            mask = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
            mask[indices_to_mask] = True
            
            # Apply mask embedding
            x_masked = x.clone()
            x_masked[mask] = self.mask_embed.to(x.dtype)
            
            return x_masked, mask

    def get_masked_node_modelling_logits(self, x: torch.Tensor) -> torch.Tensor:
        projs = [proj(x) for proj in self.pseudo_classifier_projections]
        return [
            torch.cosine_similarity(
                proj.unsqueeze(1),
                embedding.weight.unsqueeze(0),
                dim=-1
            ) / self.temperature
            for proj, embedding in zip(projs, self.pseudo_label_embeddings)
        ]# tasks=K x [batch_size * L * S, vocab_size]
        
    def _drop_edge(
        self,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        edge_weight: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        if not self.training or self.config.edge_drop_prob == 0:
            return edge_index, edge_attr, edge_weight

        # 1. Distinguish Intra vs Inter edges
        # We use integer division to find which subgraph a node belongs to.
        # If source and target are in the same subgraph bucket, it's an INTRA edge.
        S = self.config.nodes_per_lead
        row, col = edge_index
        
        # Vectorized check: boolean mask for intra-subgraph edges
        is_intra_mask = (row // S) == (col // S)

        # 2. Separate the edges we want to protect (Inter) from those we might drop (Intra)
        # We maintain the indices to reconstruct attributes/weights later
        intra_indices_loc = is_intra_mask.nonzero(as_tuple=True)[0]
        inter_indices_loc = (~is_intra_mask).nonzero(as_tuple=True)[0]

        # Extract the actual intra edges
        intra_edge_index = edge_index[:, is_intra_mask]
        
        # 3. Perform Symmetric Drop on INTRA edges only
        # Filter for unique pairs (u < v) to ensure we make the same decision for u->v and v->u
        intra_row, intra_col = intra_edge_index
        upper_tri_mask = intra_row < intra_col # Ignore lower tri and self-loops
        
        # Get indices of unique upper-triangular edges relative to the 'intra' subset
        unique_edge_locs = upper_tri_mask.nonzero(as_tuple=True)[0]
        num_unique = unique_edge_locs.size(0)
        
        # Determine how many unique pairs to KEEP
        num_keep = int(num_unique * (1 - self.config.edge_drop_prob))
        
        # Randomly select indices to keep
        perm = torch.randperm(num_unique, device=edge_index.device)[:num_keep]
        kept_unique_locs = unique_edge_locs[perm]
        
        # 4. Reconstruct the undirected Intra-Graph
        # Get the u->v edges we kept
        kept_row = intra_row[kept_unique_locs]
        kept_col = intra_col[kept_unique_locs]
        
        # Create the v->u mirror
        final_intra_row = torch.cat([kept_row, kept_col])
        final_intra_col = torch.cat([kept_col, kept_row])
        final_intra_edge_index = torch.stack([final_intra_row, final_intra_col], dim=0)
        
        # 5. Handle Attributes/Weights for Intra edges
        # We need to map our decision back to the original attributes
        # We kept 'kept_unique_locs'. We need to select the attributes at those locs, 
        # and then duplicate them for the mirror edges.
        
        # Helper to slice and mirror attributes
        def slice_and_mirror(tensor, original_locs, kept_locs):
            if tensor is None: return None
            # Get attributes of the intra-edges
            intra_tensor = tensor[original_locs]
            # Get attributes of the specific unique pairs we kept
            kept_vals = intra_tensor[kept_locs]
            # Mirror them (kept_vals for u->v, kept_vals for v->u)
            return torch.cat([kept_vals, kept_vals], dim=0)

        final_intra_attr = slice_and_mirror(edge_attr, intra_indices_loc, kept_unique_locs)
        final_intra_weight = slice_and_mirror(edge_weight, intra_indices_loc, kept_unique_locs)

        # 6. Recombine with the protected Inter-edges
        # Get the protected inter-components
        inter_edge_index = edge_index[:, ~is_intra_mask]
        inter_attr = edge_attr[inter_indices_loc] if edge_attr is not None else None
        inter_weight = edge_weight[inter_indices_loc] if edge_weight is not None else None

        # Concatenate results
        new_edge_index = torch.cat([final_intra_edge_index, inter_edge_index], dim=1)
        
        new_edge_attr = None
        if edge_attr is not None:
            new_edge_attr = torch.cat([final_intra_attr, inter_attr], dim=0)
            
        new_edge_weight = None
        if edge_weight is not None:
            new_edge_weight = torch.cat([final_intra_weight, inter_weight], dim=0)

        return new_edge_index, new_edge_attr, new_edge_weight

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Union[torch.Tensor],
        criterion: Callable,
        labels: torch.LongTensor,
        node_lead_ids : torch.LongTensor,
        node_temporal_pos : torch.LongTensor,
        metrics: Dict[str, Metric],
        edge_weight: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        num_sampled_nodes_per_hop: Optional[List[int]] = None,
        num_sampled_edges_per_hop: Optional[List[int]] = None,
        output_hidden_states: Optional[bool] = False,
        output_attention_weights: Optional[bool] = False,
        return_dict: Optional[bool] = False,
        real_node_mask: Optional[torch.Tensor] = None, # nodes eligible for masking (zero-padded leads: False)
        
    ):  
                
        assert torch.isfinite(x).all(), "Input contains non-finite values"
        
        x = self.embedder(x) # sum_{i=1}^{BS} L_i, time_steps -> sum_{i=1}^{BS} L_i, in_channels, S
        x = x.permute(0, 2, 1) # sum_{i=1}^{BS} L_i, S, conv_out
        x = x.reshape(x.size(0) * x.size(1), -1) # sum_{i=1}^{BS} L_i * S, in_channels = all_nodes_in_batch, conv_out
        
        x = self.feature_projection(x) # project to in_channels if necessary

        if output_hidden_states:
            hidden_states_0 = x

        assert torch.isfinite(x).all(), "Node embeddings contain non-finite values"
                
        assert x.size(0) == batch.size(0), f"Shape mismatch. Node embeddings {x.size()} and batch {batch.size()}"

        x, mask = self._vectorized_mask_node_features(x, batch, real_node_mask)
        
        assert torch.isfinite(x).all(), "Masked node embeddings contain non-finite values"
        
        if self.config.use_temporal_encoding:
            x = self.temporal_encoding(x, node_temporal_pos)
        
        if self.config.use_lead_encoding:
            x = self.lead_encoding(x, node_lead_ids)
        
        assert torch.isfinite(x).all(), "Node embeddings with spatiotemporal pos contain non-finite values"
        
        # with torch.autocast("cuda", dtype=torch.bfloat16):
        
        edge_index, edge_attr, edge_weight = self._drop_edge(
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_weight=edge_weight,
        )

        out = self.encoder(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_attr=edge_attr,
            batch=batch,
            batch_size=batch_size,
            num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
            num_sampled_edges_per_hop=num_sampled_edges_per_hop,
            output_hidden_states=output_hidden_states,
            output_attention_weights=output_attention_weights
        )
        
        if output_hidden_states:
            if output_attention_weights:
                x, hidden_states, attention_weights = out
            else:
                x, hidden_states = out
                attention_weights = None
            
            assert torch.isfinite(x).all(), "Encoder output contains non-finite values"
            assert torch.isfinite(torch.stack(hidden_states)).all(), "Encoder hidden states contain non-finite values"
            assert type(hidden_states) == list, f"Hidden states should be a list but got {type(hidden_states)}"
            
            hidden_states = [hidden_states_0] + hidden_states
            
        else:
            if output_attention_weights:
                x, attention_weights = out
            else:
                x = out # all_nodes_in_batch, hidden_channels
                attention_weights = None
            hidden_states = None
            
            assert torch.isfinite(x).all(), "Encoder output contains non-finite values"
            
        
        if self.encoder.encoder.supports_norm_edge_index:
            x = self.norm_post_encoder(x, edge_index)
        elif self.encoder.encoder.supports_norm_batch:
            x = self.norm_post_encoder(x, batch, batch_size)
        else:
            x = self.norm_post_encoder(x)

        assert torch.isfinite(x).all(), "Normalized encoder output contains non-finite values"
            
        if self.config.use_pseudo_label_embeddings:
            logits = self.get_masked_node_modelling_logits(x)
        else:
            logits = [classifier(x) for classifier in self.classifiers] # K x [all_nodes_in_batch, task_vocab_size]
        
        for i, logit in enumerate(logits):
            assert torch.isfinite(logit).all(), f"Logits task {i} contains non-finite values" # failed

        assert labels.size(0) == logits[0].size(0), f"Shape mismatch. Labels {labels.size()} and logits {len(logits)} x {logits[0].size()}"

        labels = labels.permute(1, 0) # all_nodes_in_batch, K -> K, all_nodes_in_batch

        loss = 0
        # with torch.autocast("cuda", enabled=False):
        for task_logits, task_labels, metric in zip(logits, labels, metrics.values()):
            if self.training:
                assert mask.sum() > 0, f"The mask is empty. found {mask.sum()} masked nodes"
                loss += criterion(task_logits[mask], task_labels[mask])
                metric.update(task_logits[mask].detach(), task_labels[mask].detach())
            elif real_node_mask is not None:
                # zero-padded leads: score only the sampled leads' nodes, as the rls arm does
                loss += criterion(task_logits[real_node_mask], task_labels[real_node_mask])
                metric.update(task_logits[real_node_mask].detach(), task_labels[real_node_mask].detach())
            else:
                loss += criterion(task_logits, task_labels)
                metric.update(task_logits.detach(), task_labels.detach())
                
        # assert torch.isfinite(loss).all(), "Loss is non-finite"

        if not return_dict:
            return (loss, logits, hidden_states, mask, attention_weights)
        
        if output_attention_weights:
            return MaskedNodeModellingOuputWithAttention(
                loss=loss,
                logits=logits,
                hidden_states=hidden_states,
                mask=mask,
                attention_weights=attention_weights
            )
        else:
            return MaskedNodeModellingOutput(
                loss=loss,
                logits=logits,
                hidden_states=hidden_states,
                mask=mask
            )
        
# --- Graph classification ---

class GraphClassificationBase(PreTrainedModel):
    def __init__(self, config, encoder_cls):
        super().__init__(config)
        
        if config.num_classes is None:
            raise ValueError(f"Argument num_classes cannot be None for downstream tasks")
        
        self.config = config
        self.embedder = HubertFeatureEncoder(config)
        
        if config.use_lead_encoding:
            self.lead_encoding = InterLeadSpatialEncoding(config)
            
        if config.use_temporal_encoding:
            self.temporal_encoding = IntraLeadTemporalEncoding(config)
            
        self.encoder = encoder_cls(config)
        
        self.norm_post_encoder = copy.deepcopy(self.encoder.encoder.norms[0])

        if config.pooling == 'global_mean': # todo: add here other pooling layers if necessary
            self.pooling = nng.pool.global_mean_pool
        elif config.pooling == 'global_max':
            self.pooling = nng.pool.global_max_pool
        elif config.pooling == "global_att":
            self.gate_nn = nn.Linear(config.hidden_channels, 1)
            self.pooling = nng.GlobalAttention(gate_nn=self.gate_nn)
        else:
            raise NotImplementedError(f"{config.pooling} configuration has not been implemented yet")
        
        if config.linear_head:
            self.output_layer = nn.Linear(
                config.out_channels if config.out_channels is not None else config.hidden_channels,
                config.num_classes,
                bias=False
            )
        else:
            self.output_layer = nn.Sequential(
                nn.Linear(config.out_channels if config.out_channels is not None else config.hidden_channels,
                          config.hidden_channels,
                        ),
                nn.GELU(),
                nn.Dropout(p=config.classifier_dropout),
                nn.Linear(config.hidden_channels, config.num_classes)
            )
                

    def get_logits_for_graph_classification(self, x: torch.Tensor, **pooling_kwargs: Any) -> torch.Tensor:
        x_pooled = self.pooling(x, **pooling_kwargs) # B, hidden_channels
        return self.output_layer(x_pooled) # B, num_classes
    
    def _drop_edge(
        self,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        edge_weight: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        # Return immediately if not training or drop prob is 0
        if not self.training or self.config.edge_drop_prob == 0:
            return edge_index, edge_attr, edge_weight

        # 1. Distinguish Intra vs Inter edges
        # We use integer division to find which subgraph a node belongs to.
        # If source and target are in the same subgraph bucket, it's an INTRA edge.
        S = self.config.nodes_per_lead
        row, col = edge_index
        
        # Vectorized check: boolean mask for intra-subgraph edges
        is_intra_mask = (row // S) == (col // S)

        # 2. Separate the edges we want to protect (Inter) from those we might drop (Intra)
        # We maintain the indices to reconstruct attributes/weights later
        intra_indices_loc = is_intra_mask.nonzero(as_tuple=True)[0]
        inter_indices_loc = (~is_intra_mask).nonzero(as_tuple=True)[0]

        # Extract the actual intra edges
        intra_edge_index = edge_index[:, is_intra_mask]
        
        # 3. Perform Symmetric Drop on INTRA edges only
        # Filter for unique pairs (u < v) to ensure we make the same decision for u->v and v->u
        intra_row, intra_col = intra_edge_index
        upper_tri_mask = intra_row < intra_col # Ignore lower tri and self-loops
        
        # Get indices of unique upper-triangular edges relative to the 'intra' subset
        unique_edge_locs = upper_tri_mask.nonzero(as_tuple=True)[0]
        num_unique = unique_edge_locs.size(0)
        
        # Determine how many unique pairs to KEEP
        num_keep = int(num_unique * (1 - self.config.edge_drop_prob))
        
        # Randomly select indices to keep
        perm = torch.randperm(num_unique, device=edge_index.device)[:num_keep]
        kept_unique_locs = unique_edge_locs[perm]
        
        # 4. Reconstruct the undirected Intra-Graph
        # Get the u->v edges we kept
        kept_row = intra_row[kept_unique_locs]
        kept_col = intra_col[kept_unique_locs]
        
        # Create the v->u mirror
        final_intra_row = torch.cat([kept_row, kept_col])
        final_intra_col = torch.cat([kept_col, kept_row])
        final_intra_edge_index = torch.stack([final_intra_row, final_intra_col], dim=0)
        
        # 5. Handle Attributes/Weights for Intra edges
        # We need to map our decision back to the original attributes
        # We kept 'kept_unique_locs'. We need to select the attributes at those locs, 
        # and then duplicate them for the mirror edges.
        
        # Helper to slice and mirror attributes
        def slice_and_mirror(tensor, original_locs, kept_locs):
            if tensor is None: return None
            # Get attributes of the intra-edges
            intra_tensor = tensor[original_locs]
            # Get attributes of the specific unique pairs we kept
            kept_vals = intra_tensor[kept_locs]
            # Mirror them (kept_vals for u->v, kept_vals for v->u)
            return torch.cat([kept_vals, kept_vals], dim=0)

        final_intra_attr = slice_and_mirror(edge_attr, intra_indices_loc, kept_unique_locs)
        final_intra_weight = slice_and_mirror(edge_weight, intra_indices_loc, kept_unique_locs)

        # 6. Recombine with the protected Inter-edges
        # Get the protected inter-components
        inter_edge_index = edge_index[:, ~is_intra_mask]
        inter_attr = edge_attr[inter_indices_loc] if edge_attr is not None else None
        inter_weight = edge_weight[inter_indices_loc] if edge_weight is not None else None

        # Concatenate results
        new_edge_index = torch.cat([final_intra_edge_index, inter_edge_index], dim=1)
        
        new_edge_attr = None
        if edge_attr is not None:
            new_edge_attr = torch.cat([final_intra_attr, inter_attr], dim=0)
            
        new_edge_weight = None
        if edge_weight is not None:
            new_edge_weight = torch.cat([final_intra_weight, inter_weight], dim=0)

        return new_edge_index, new_edge_attr, new_edge_weight

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Union[torch.Tensor,],
        criterion: Callable,
        labels: torch.LongTensor,
        node_lead_ids : torch.LongTensor,
        node_temporal_pos : torch.LongTensor,
        metrics : Dict[str, Metric],
        edge_weight: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        num_sampled_nodes_per_hop: Optional[List[int]] = None,
        num_sampled_edges_per_hop: Optional[List[int]] = None,
        output_hidden_states: Optional[bool] = False,
        output_attention_weights: Optional[bool] = False, 
        return_dict: Optional[bool] = None,
        
    ):
        
        assert torch.isfinite(x).all(), "Input contains non-finite values"
        
        x = self.embedder(x) # sum_{i=1}^{BS} L_i, time_steps -> sum_{i=1}^{BS} L_i, in_channels, S
        x = x.permute(0, 2, 1) # sum_{i=1}^{BS} L_i, S, in_channels
        x = x.reshape(x.size(0) * x.size(1), -1) # sum_{i=1}^{BS} L_i * S, in_channels = all_nodes_in_batch, in_channels
        
        assert torch.isfinite(x).all(), "Node embeddings contain non-finite values"
        
        if self.config.use_temporal_encoding:
            x = self.temporal_encoding(x, node_temporal_pos)
            
        if self.config.use_lead_encoding:
            x = self.lead_encoding(x, node_lead_ids)
        
        assert torch.isfinite(x).all(), "Node embeddings with spatiotemporal pos contain non-finite values"
        
        edge_index, edge_attr, edge_weight = self._drop_edge(
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_weight=edge_weight,
        )

        out = self.encoder(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_attr=edge_attr,
            batch=batch,
            batch_size=batch_size,
            num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
            num_sampled_edges_per_hop=num_sampled_edges_per_hop,
            output_hidden_states=output_hidden_states,
            output_attention_weights=output_attention_weights
        )
        
        if output_hidden_states:
            if output_attention_weights:
                x, hidden_states, (edge_idx, attention_weights) = out
            else:
                x, hidden_states = out
                edge_idx, attention_weights = None, None
            
            assert torch.isfinite(x).all(), "Encoder output contains non-finite values"
            assert torch.isfinite(torch.stack(hidden_states)).all(), "Encoder hidden states contain non-finite values"
            
        else:
            if output_attention_weights:
                x, (edge_idx, attention_weights) = out
            else:
                x = out # all_nodes_in_batch, hidden_channels
                edge_idx, attention_weights = None, None
            hidden_states = None
            
            assert torch.isfinite(x).all(), "Encoder output contains non-finite values"

        if self.encoder.encoder.supports_norm_edge_index:
            x = self.norm_post_encoder(x, edge_index)
        elif self.encoder.encoder.supports_norm_batch:
            x = self.norm_post_encoder(x, batch, batch_size)
        else:
            x = self.norm_post_encoder(x)

        assert torch.isfinite(x).all(), "Normalized encoder output contains non-finite values"

        logits = self.get_logits_for_graph_classification(x, batch=batch) # todo: to adapt in case of more complex pooling
        
        if isinstance(criterion, nn.CrossEntropyLoss):
            labels = labels.view(-1).long()
        else:        
            labels = labels.reshape(logits.size(0), -1)

        loss = None
        # with torch.autocast("cuda", enabled=False):
        loss = criterion(logits, labels)
        for _, metric in metrics.items():
            metric.update(logits.detach(), labels.long().detach())

        if not return_dict:
            return (loss, logits, hidden_states, (edge_idx, attention_weights))
        
        if output_attention_weights:
            return GraphClassificationOutputWithAttention(
                loss=loss,
                logits=logits,
                hidden_states=hidden_states,
                attention_weights=(edge_idx, attention_weights)
            )
        else:
            return GraphClassificationOutput(
                loss=loss,
                logits=logits,
                hidden_states=hidden_states
            )

# --- Concrete Encoders (GIN, GAT, GraphTransformer) ---

# ! IMPORTANT !
# The forward method of these concrete encoders has to be the same for everyone and must contain the largest set of parameters in the signature.
# Then, the truly underlying encoder will take only the necessary parameters
# e.g. MLP encoder accepts way less parameters than GIN or GAT but the forward signature is the same and the body uses only the required parameters
# todo: So far GIN/GAT has the longest list of parameters but if GraphTransformer will take more params, we'll have to modify the forward accordingly

class ECGGIN(PreTrainedModel):
    config_class = ECGGINConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.encoder = GIN(
            in_channels=config.in_channels,
            hidden_channels=config.hidden_channels,
            out_channels=config.out_channels,
            num_layers=config.num_layers,
            dropout=config.dropout,
            act=config.act,
            act_first=config.act_first,
            act_kwargs=config.act_kwargs,
            norm=GranolaNorm(config) if config.norm == "granola" else config.norm, # todo: ask Moshe if it's ok not to have an edge index passed into granola feature extractor and 
            norm_kwargs=config.norm_kwargs,
            jk=config.jk,
            **config.ginConv_kwargs if config.ginConv_kwargs is not None else {}
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Union[torch.Tensor, ],
        edge_weight: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        num_sampled_nodes_per_hop: Optional[List[int]] = None,
        num_sampled_edges_per_hop: Optional[List[int]] = None,
        output_hidden_states: Optional[bool] = False,
        output_attention_weights: Optional[bool] = False
    ) -> torch.Tensor:
        return self.encoder(
            x=x, 
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_attr=edge_attr,
            batch=batch,
            batch_size=batch_size,
            num_sampled_edges_per_hop=num_sampled_edges_per_hop,
            num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
            output_hidden_states=output_hidden_states
            )

class ECGGAT(PreTrainedModel):
    config_class = ECGGATConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.encoder = GAT(
            in_channels=config.in_channels,
            hidden_channels=config.hidden_channels,
            out_channels=config.out_channels,
            num_layers=config.num_layers,
            dropout=config.dropout,
            act=config.act,
            act_first=config.act_first,
            act_kwargs=config.act_kwargs,
            norm=GranolaNorm(config) if config.norm == 'granola' else config.norm,
            norm_kwargs=config.norm_kwargs,
            jk=config.jk,
            **config.gat_kwargs if config.gat_kwargs is not None else {}
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Union[torch.Tensor, ],
        edge_weight: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        num_sampled_nodes_per_hop: Optional[List[int]] = None,
        num_sampled_edges_per_hop: Optional[List[int]] = None,
        output_hidden_states: Optional[bool] = False,
        output_attention_weights: Optional[bool] = False
    ) -> torch.Tensor:
        return self.encoder(
            x=x, 
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_attr=edge_attr,
            batch=batch,
            batch_size=batch_size,
            num_sampled_edges_per_hop=num_sampled_edges_per_hop,
            num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
            output_hidden_states=output_hidden_states,
            output_attention_weights=output_attention_weights
            )
    
class ECGMLP(PreTrainedModel):
    config=ECGMLPConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.encoder = MLP(
            in_channels=config.in_channels,
            hidden_channels=config.hidden_channels,
            out_channels=config.out_channels,
            num_layers=config.num_layers,
            dropout=config.dropout,
            act=config.act,
            act_first=config.act_first,
            act_kwargs=config.act_kwargs,
            norm=config.norm, # cannot work with GRANOLA because it uses the edge_index
            norm_kwargs=config.norm_kwargs,
            bias=config.bias,
            plain_last=config.plain_last,
            **config.mlp_kwargs if config.mlp_kwargs is not None else {}
        )
    def forward(
        self,
        x: torch.Tensor,
        edge_index: Union[torch.Tensor, ],
        edge_weight: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        num_sampled_nodes_per_hop: Optional[List[int]] = None,
        num_sampled_edges_per_hop: Optional[List[int]] = None,
        output_hidden_states: Optional[bool] = False,
        output_attention_weights: Optional[bool] = False
    ):
        return self.encoder(
            x=x,
            batch=batch,
            batch_size=batch_size,
            output_hidden_states=output_hidden_states
        )

class ECGGraphTransformer(PreTrainedModel):
    config_class = ECGGraphTransformerConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.encoder = GraphTransformer(
            in_channels=config.in_channels,
            hidden_channels=config.hidden_channels,
            num_layers=config.num_layers, 
            out_channels=config.out_channels,
            dropout=config.dropout,
            act=config.act,
            act_first=config.act_first, 
            act_kwargs=config.act_kwargs,
            norm=GranolaNorm(config) if config.norm == 'granola' else config.norm,
            norm_kwargs=config.norm_kwargs, 
            jk=config.jk,
            **config.graph_transformer_kwargs if config.graph_transformer_kwargs is not None else {}
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Union[torch.Tensor, ],
        edge_weight: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        num_sampled_nodes_per_hop: Optional[List[int]] = None,
        num_sampled_edges_per_hop: Optional[List[int]] = None,
        output_hidden_states: Optional[bool] = False,
        output_attention_weights: Optional[bool] = False
    ) -> torch.Tensor:
        return self.encoder(
            x=x, 
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_attr=edge_attr,
            batch=batch,
            batch_size=batch_size,
            num_sampled_edges_per_hop=num_sampled_edges_per_hop,
            num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
            output_hidden_states=output_hidden_states,
            output_attention_weights=output_attention_weights
            )


class ECGGCN(PreTrainedModel):
    
    config_class = ECGGCNConfig
    
    def __init__(self, config: ECGGCNConfig):
        super().__init__(config)
        self.config = config
        self.encoder = GCN(
            in_channels=config.in_channels,
            hidden_channels=config.hidden_channels,
            out_channels=config.out_channels,
            num_layers=config.num_layers,
            dropout=config.dropout,
            act=config.act,
            act_first=config.act_first,
            act_kwargs=config.act_kwargs,
            norm=GranolaNorm(config) if config.norm == "granola" else config.norm,
            norm_kwargs=config.norm_kwargs,
            jk=config.jk,
            **config.gcn_kwargs if config.gcn_kwargs is not None else {}
        )
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: Union[torch.Tensor, ],
        edge_weight: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        num_sampled_nodes_per_hop: Optional[List[int]] = None,
        num_sampled_edges_per_hop: Optional[List[int]] = None,
        output_hidden_states: Optional[bool] = False,
        output_attention_weights: Optional[bool] = False
        ) -> torch.Tensor:
        
        return self.encoder(
            x=x, 
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_attr=edge_attr,
            batch=batch,
            batch_size=batch_size,
            num_sampled_edges_per_hop=num_sampled_edges_per_hop,
            num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
            output_hidden_states=output_hidden_states,
            )
    
    
# --- Concrete Model Classes ---

# --- Models for Masked Node Modelling ---

class ECGGINForMaskedNodeModelling(MaskedNodeModellingBase):
    config_class = ECGGINConfig
    def __init__(self, config):
        super().__init__(config, ECGGIN)

class ECGGATForMaskedNodeModelling(MaskedNodeModellingBase):
    config_class = ECGGATConfig
    def __init__(self, config):
        super().__init__(config, ECGGAT)

class ECGGraphTransformerForMaskedNodeModelling(MaskedNodeModellingBase):
    config_class = ECGGraphTransformerConfig
    def __init__(self, config):
        super().__init__(config, ECGGraphTransformer)

class ECGMLPForMaskedNodeModelling(MaskedNodeModellingBase):
    config_class = ECGMLPConfig
    def __init__(self, config):
        super().__init__(config, ECGMLP)
        
class ECGGCNForMaskedNodeModelling(MaskedNodeModellingBase):
    config_class = ECGGCNConfig
    def __init__(self, config):
        super().__init__(config, ECGGCN)

# --- Models for Graph Classification ---

class ECGGINForGraphClassification(GraphClassificationBase):
    config_class = ECGGINConfig
    def __init__(self, config):
        super().__init__(config, ECGGIN)

class ECGGATForGraphClassification(GraphClassificationBase):
    config_class = ECGGATConfig
    def __init__(self, config):
        super().__init__(config, ECGGAT)

class ECGGraphTransformerForGraphClassification(GraphClassificationBase):
    config_class = ECGGraphTransformerConfig
    def __init__(self, config):
        super().__init__(config, ECGGraphTransformer)

class ECGMLPForGraphClassification(GraphClassificationBase):
    config_class = ECGMLPConfig
    def __init__(self, config):
        super().__init__(config, ECGMLP)
        
class ECGGCNForGraphClassification(GraphClassificationBase):
    config_class = ECGGCNConfig
    def __init__(self, config):
        super().__init__(config, ECGGCN)

# --- quick fetching ---


MODEL_NAME_2_PRETRAINING_MODEL = {
    'gin' : ECGGINForMaskedNodeModelling,
    'gat' : ECGGATForMaskedNodeModelling,
    'graph_transformer' : ECGGraphTransformerForMaskedNodeModelling,
    'mlp' : ECGMLPForMaskedNodeModelling
}

MODEL_NAME_2_FINETUNING_MODEL = {
    'gin' : ECGGINForGraphClassification,
    'gat' : ECGGATForGraphClassification,
    'graph_transformer' : ECGGraphTransformerForGraphClassification,
    'mlp' : ECGMLPForGraphClassification
}