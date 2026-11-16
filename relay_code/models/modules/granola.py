from typing import Callable
import torch
from torch import nn
from inspect import signature
from torch_geometric.nn.conv import GINConv, GATv2Conv, GCNConv

class GNNFeatureExtractor(nn.Module):
    def __init__(self, gnn : Callable):
        super().__init__()
        self.gnn = gnn
        self.act = nn.GELU()
        # compute actual output dim of the GNN (handle GAT/GATv2 with heads/concat)
        heads = getattr(gnn, "heads", 1)
        concat = getattr(gnn, "concat", True)
        out_dim = gnn.out_channels * heads if concat else gnn.out_channels
        self.norm = nn.LayerNorm(out_dim)
        self.out_dim = out_dim  # CHANGED: expose actual output dimension for downstream use

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.norm(self.act(self.gnn(x, edge_index)))


class GranolaNorm(nn.Module):
    r"""
    Compact GRANOLA-style normalization.

    Design decisions for a lighter-weight variant:
    - Use a tiny MLP as default feature extractor (1-2 layers) instead of full GNNs.
    - Small random feature dimensionality by default.
    - Small parameter predictor (single hidden layer) with configurable hidden size.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        # Core channel sizes (defaults fall back to small values to save parameters)
        self.in_channels = config.norm_kwargs.get("granola_in_channels", getattr(config, "hidden_channels", 64))
        self.hidden_channels = config.norm_kwargs.get("granola_hidden_channels", max(16, self.in_channels // 2))
        self.out_channels = config.norm_kwargs.get("granola_out_channels", self.hidden_channels)
        # small RNF dimension by default
        self.random_features_channels = config.norm_kwargs.get("granola_random_features_channels", max(8, self.in_channels // 8))
        self.feat_extractor_type = config.norm_kwargs.get("granola_feat_extractor_type", "gat")
        # hidden dim for predictor -- keep small to reduce params
        self.norm_param_pred_hidden_channels = config.norm_kwargs.get(
            "granola_norm_param_pred_hidden_channels",
            max(32, self.hidden_channels + self.random_features_channels)
        )

        self.feat_extractor = self._build_feat_extractor(self.feat_extractor_type)
        self.param_predictor = self._build_param_predictor()

    def _build_feat_extractor(self, extractor_type):
        """
        Build a compact feature extractor.

        - 'tiny_mlp' (default): 1-2 layer MLP with GELU and optional layernorm.
        - 'linear': single linear projection (minimal params).
        - If extractor_type == 'heavy' and the external GNNs are available, you can re-enable them
          by replacing this branch with the commented-out original code.
        """
        if extractor_type == "linear":
            return nn.Sequential(
                nn.Linear(self.in_channels, self.out_channels, bias=False),
                nn.LayerNorm(self.out_channels),
                nn.GELU()
            )

        if extractor_type == "mlp":
            hidden = max(self.hidden_channels, min(self.in_channels * 2, 64))
            return nn.Sequential(
                nn.Linear(self.in_channels, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.out_channels),
                nn.LayerNorm(self.out_channels),
                nn.GELU()
            )
            
        if extractor_type == "gcn":
            return GNNFeatureExtractor(
                GCNConv(in_channels=self.in_channels, out_channels=self.out_channels, bias=False),
                )
        
        if extractor_type == "gat":
            return GNNFeatureExtractor(
                GATv2Conv(in_channels=self.in_channels, out_channels=self.out_channels, heads=2, concat=True, bias=False),
                )

        return nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
            nn.GELU()
        )

    def _build_param_predictor(self):
        # CHANGED: compute actual feature dimension from the extractor (handles GNN concat/heads)
        feat_dim = getattr(self.feat_extractor, "out_dim", self.out_channels)  # CHANGED: use feat_extractor.out_dim if present
        input_dim = feat_dim + self.random_features_channels
        hidden_dim = self.norm_param_pred_hidden_channels

        # small predictor: one hidden layer
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * self.in_channels)  # outputs gamma and beta
        )

    def reset_parameters(self):
        # reset small extractor and predictor
        for m in self.modules():
            if hasattr(m, "reset_parameters"):
                try:
                    m.reset_parameters()
                except Exception:
                    pass

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: node features (N, H)
        edge_index: optional, passed to extractor only if it accepts it (for compatibility)
        """
        
        if "edge_index" in signature(self.feat_extractor.forward).parameters:
            H_tilde = self.feat_extractor(x, edge_index)
        else:
            H_tilde = self.feat_extractor(x)

        # small RNF to inject stochasticity
        rnf = torch.randn((H_tilde.size(0), self.random_features_channels), device=H_tilde.device)

        gamma_beta = self.param_predictor(torch.cat([H_tilde, rnf], dim=-1))
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)

        # standardize per-node (feature-wise) and apply predicted affine transform
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + 1e-5)

        return gamma * x_norm + beta
