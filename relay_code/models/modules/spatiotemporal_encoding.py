import torch
from torch import nn
import math


class IntraLeadTemporalEncoding(nn.Module):
    """
    Intra-lead temporal encoding class.
    The purpose of this module is to add temporal encoding to nodes in subgraphs corresponding to parts of the same lead (intra-lead).
    This encoding encapsulates the temporal position of nodes in subgraphs, i.e. the temporal position of segments within a lead.
    This encoding is applied in the same fashion to all subgraphs as leads are recorded simultaneously but differently among subgraph node
    """
    def __init__(self, config):
        super().__init__()
        self.embedding_dim = config.in_channels
        self.max_length = config.nodes_per_lead

        pe = torch.zeros(self.max_length, self.embedding_dim)
        position = torch.arange(0, self.max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2).float() * (-math.log(10000.0) / self.embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("temporal_encoding", pe) # non learnable buffer

    def forward(self, x: torch.Tensor, node_temporal_pos : torch.LongTensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor of node features (batch_size * sum(sampled_leads) * num_nodes, in_channels)
            node_temporal_pos: torch.LongTensor of shape (batch_size * sum(sampled_leads) * num_nodes)
        
        Returns:
            x: node features with added temporal encoding
        """
        
        return x + self.temporal_encoding[node_temporal_pos]  # (batch_size * sum(sampled_leads) * num_nodes, in_channels)

class InterLeadSpatialEncoding(nn.Module):
    """
    Inter-lead spatial encoding class.
    The purpose of this module is to add spatial encoding to nodes in subgraphs, corresponding to segments in leads, depending on the subgraph/lead.
    This encoding encapsulates the spatial position of node in subgraphs, i.e. the spatial position of leads/electrodes on the body.
    This encoding is applied to all subgraph nodes in the same fashion as segments come from the same lead but differently among subgraphs
    """

    def __init__(self, config):
        super().__init__()
        self.spatial_embedding = nn.Embedding(
            num_embeddings=12, # max number of leads
            embedding_dim=config.in_channels
        )
        

    def forward(self, x : torch.Tensor, node_lead_ids : torch.LongTensor) -> torch.Tensor:
        """
        Args:
        - x : torch.Tensor of node features shape (batch_size * sum(sampled_leads) * num_nodes, in_channels)
        - node_lead_ids : torch.LongTensor of shape (batch_size * sum(sampled_leads) * num_nodes)
        Returns:
        - x : torch.Tensor
        """
        return x + self.spatial_embedding(node_lead_ids)

