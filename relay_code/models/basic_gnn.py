import copy
import inspect
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Linear, ModuleList
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.loader import CachedLoader, NeighborLoader
from torch_geometric.nn.conv import MessagePassing

from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils._trim_to_layer import TrimToLayer



class BasicGNN(torch.nn.Module):
    r"""An abstract class for implementing basic GNN models.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of the underlying
            :class:`torch_geometric.nn.conv.MessagePassing` layers.
    """
    supports_edge_weight: Final[bool]
    supports_edge_attr: Final[bool]
    supports_norm_batch: Final[bool]
    supports_norm_edge_index: Final[bool]
    supports_attention_weights: Final[bool]

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        layerdrop : float = 0.1, # NEW
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        jk: Optional[str] = None,
        supports_skip_connection: bool = True, 
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.dropout = torch.nn.Dropout(p=dropout)
        self.layerdrop = layerdrop 
        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.jk_mode = jk
        self.act_first = act_first
        if isinstance(norm, str):
            self.norm = norm
        elif isinstance(norm, Callable):
            self.norm = norm
        else:
            self.norm = None
        self.norm_kwargs = norm_kwargs

        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = hidden_channels
            
        # --- populate conv stack ---

        self.convs = ModuleList()
        
        # --- addded ---
        
        self.lin_expansion = None
        self.lin_compression = None
        self.expansion_norm = None
        self.compression_norm = None
        
        
        if self.supports_expansion_compression:
            
            assert not jk, f"Jumping knowledge disabled for GNN working in expanded spaces"
            
            self.lin_expansion = Linear(in_channels, hidden_channels, bias=False)
            self.lin_compression = Linear(hidden_channels, self.out_channels, bias=False)
                
            for _ in range(num_layers):
                self.convs.append(
                    self.init_conv(
                        hidden_channels, hidden_channels, **kwargs
                        )
                    )
                
        else:        
            if num_layers > 1:
                
                self.convs.append(self.init_conv(in_channels, hidden_channels, **kwargs))
                
                if isinstance(in_channels, (tuple, list)):
                    in_channels = (hidden_channels, hidden_channels)
                else:
                    in_channels = hidden_channels
                    
            for _ in range(num_layers - 2):
                
                self.convs.append(self.init_conv(in_channels, hidden_channels, **kwargs))
                
                if isinstance(in_channels, (tuple, list)):
                    in_channels = (hidden_channels, hidden_channels)
                else:
                    in_channels = hidden_channels
                    
            if out_channels is not None and jk is None:
                self._is_conv_to_out = True
                self.convs.append(
                    self.init_conv(in_channels, out_channels, **kwargs))
            else:
                self.convs.append(
                    self.init_conv(in_channels, hidden_channels, **kwargs))
            
        if hasattr(self.convs[-1], "forward"):
            conv_params = inspect.signature(self.convs[-1].forward).parameters
            self.supports_attention_weights = 'return_attention_weights' in conv_params
        else:
            self.supports_attention_weights = False
            
        # --- populate norm stack ---

        self.norms = ModuleList()
        
        norm_layer = normalization_resolver( # this returns None is norm is not a string
            norm,
            hidden_channels,
            **(norm_kwargs or {}),
        )
        
        if norm_layer is None:
            norm_layer = norm if isinstance(norm, Callable) else torch.nn.Identity() # this allows for custom, callable normalizations

        self.supports_norm_batch = False
        self.supports_norm_edge_index = False
        if hasattr(norm_layer, 'forward'):
            norm_params = inspect.signature(norm_layer.forward).parameters
            self.supports_norm_batch = 'batch' in norm_params
            self.supports_norm_edge_index = 'edge_index' in norm_params
            
        if self.supports_expansion_compression:            
            self.expansion_norm = copy.deepcopy(norm_layer)
            self.compression_norm = copy.deepcopy(norm_layer)

        for _ in range(num_layers - 1):
            self.norms.append(copy.deepcopy(norm_layer))

        if jk is not None:
            self.norms.append(copy.deepcopy(norm_layer))
        else:
            self.norms.append(torch.nn.Identity())

        if jk is not None and jk != 'last':
            self.jk = JumpingKnowledge(jk, hidden_channels, num_layers)

        if jk is not None:
            if jk == 'cat':
                in_channels = num_layers * hidden_channels
            else:
                in_channels = hidden_channels
            self.lin = Linear(in_channels, self.out_channels)
            
        # todo: NEW
        self.supports_skip_connection = supports_skip_connection
        # logger.warning(f"Supports skip connection: {self.supports_skip_connection}")

        # We define `trim_to_layer` functionality as a module such that we can
        # still use `to_hetero` on-top.
        self._trim = TrimToLayer()

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:
        raise NotImplementedError

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()
        if hasattr(self, 'jk'):
            self.jk.reset_parameters()
        if hasattr(self, 'lin'):
            self.lin.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
        num_sampled_nodes_per_hop: Optional[List[int]] = None,
        num_sampled_edges_per_hop: Optional[List[int]] = None,
        output_hidden_states : Optional[bool] = False,
        output_attention_weights : Optional[bool] = False
    ) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_weight (torch.Tensor, optional): The edge weights (if
                supported by the underlying GNN layer). (default: :obj:`None`)
            edge_attr (torch.Tensor, optional): The edge features (if supported
                by the underlying GNN layer). (default: :obj:`None`)
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
            batch_size (int, optional): The number of examples :math:`B`.
                Automatically calculated if not given.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
            num_sampled_nodes_per_hop (List[int], optional): The number of
                sampled nodes per hop.
                Useful in :class:`~torch_geometric.loader.NeighborLoader`
                scenarios to only operate on minimal-sized representations.
                (default: :obj:`None`)
            num_sampled_edges_per_hop (List[int], optional): The number of
                sampled edges per hop.
                Useful in :class:`~torch_geometric.loader.NeighborLoader`
                scenarios to only operate on minimal-sized representations.
                (default: :obj:`None`)
            output_hidden_states (Bool, optional): Whether to return a list of per-layer
                node representations. `num_layers` x `[N, D_l]`
                (default: :obj:`False`)
            output_attention_weights (Bool, optional): Whether to return a list of per-layer
                attention weights. `num_layers` x `[E, ]`
                (default: :obj:`False`)
        """
        if (num_sampled_nodes_per_hop is not None
                and isinstance(edge_weight, Tensor)
                and isinstance(edge_attr, Tensor)):
            raise NotImplementedError("'trim_to_layer' functionality does not "
                                      "yet support trimming of both "
                                      "'edge_weight' and 'edge_attr'")

        xs: List[Tensor] = []
        
        if output_hidden_states:
            hidden_states : List[Tensor] = []
            
        if output_attention_weights and self.supports_attention_weights:
            attention_weights: Optional[Tensor] = []
            
        
        assert len(self.convs) == len(self.norms), f"Found {len(self.convs)} convs and {len(self.norms)} norms. They should be the same"
        
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            
            layerdrop_probability = torch.rand([])
            
            # following conditions ensures that:
            # - we are in training mode
            # - layerdrop_probability is less than self.layerdrop
            # - we are not at the last layer (to ensure at least one layer is executed)
            
            skip_layer = self.training and layerdrop_probability < self.layerdrop and i < self.num_layers - 1 
            
            if self.supports_expansion_compression and (i == 0 or i == self.num_layers - 1):
                skip_layer = False # never skip expansion or compression layers
            
            if not skip_layer:  
                
                residual = x 
                
                # --- trimming ---
                
                if (not torch.jit.is_scripting() and num_sampled_nodes_per_hop is not None):
                    x, edge_index, value = self._trim(
                        i,
                        num_sampled_nodes_per_hop,
                        num_sampled_edges_per_hop,
                        x,
                        edge_index,
                        edge_weight if edge_weight is not None else edge_attr,
                    )
                    if edge_weight is not None:
                        edge_weight = value
                    else:
                        edge_attr = value

                # Tracing the module is not allowed with *args and **kwargs :(
                # As such, we rely on a static solution to pass optional edge
                # weights and edge attributes to the module.
                
                # --- convolution ---
                
                if self.supports_edge_weight and self.supports_edge_attr:
                    if self.supports_attention_weights and output_attention_weights:
                        x, (edge_index, att_weights) = conv(x, edge_index, edge_weight=edge_weight, edge_attr=edge_attr, return_attention_weights=True)
                    else:
                        x = conv(x, edge_index, edge_weight=edge_weight, edge_attr=edge_attr)
                    
                elif self.supports_edge_weight:
                    if self.supports_attention_weights and output_attention_weights:
                        x, (edge_index, att_weights) = conv(x, edge_index, edge_weight=edge_weight, return_attention_weights=True)
                    else:
                        x = conv(x, edge_index, edge_weight=edge_weight)

                elif self.supports_edge_attr:
                    if self.supports_attention_weights and output_attention_weights:
                        x, (edge_index, att_weights) = conv(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
                    else:
                        x = conv(x, edge_index, edge_attr=edge_attr)
                        
                elif self.supports_expansion_compression:
                    if i == 0:
                        x = self.lin_expansion(x)
                        x = self.expansion_norm(x)
                        x = self.act(x)
                        x = self.dropout(x)                        
                        x = conv(x, edge_index)
                        
                    elif i == self.num_layers -1:
                        x = conv(x, edge_index)                        
                        x = self.lin_compression(x)
                        x = self.compression_norm(x)
                    else:
                        x = conv(x, edge_index)

                else:
                    if self.supports_attention_weights and output_attention_weights:
                        x, (edge_index, att_weights) = conv(x, edge_index, return_attention_weights=True)
                    else:
                        x = conv(x, edge_index)   
                        
                if output_attention_weights and self.supports_attention_weights:
                    attention_weights.append(att_weights)
                    
                # --- normalization, activation, dropout ---

                if i < self.num_layers - 1 or self.jk_mode is not None:
                    
                    if self.act is not None and self.act_first:
                        x = self.act(x)
                                            
                    if self.supports_norm_batch:
                        x = norm(x, batch, batch_size)
                        
                    elif self.supports_norm_edge_index: # to enable GRANOLA
                        x = norm(x, edge_index)
                        
                    else:
                        x = norm(x)
                                            
                    if self.act is not None and not self.act_first:
                        x = self.act(x)
                                            
                    x = self.dropout(x)
                    
                    if hasattr(self, 'jk'):
                        xs.append(x)
                        
                    # --- skip connection: add (if necessary projected) residual ---
                    if self.supports_skip_connection:
                        
                        if residual.size(-1) != x.size(-1):
                            proj_layer_name = f'_skip_proj_{i}'
                            
                            if not hasattr(self, proj_layer_name):
                                setattr(self, proj_layer_name, torch.nn.Linear(residual.size(-1), x.size(-1), device=x.device))
                                
                            residual = getattr(self, proj_layer_name)(residual)
                            
                        x = x + residual
                        
            
            # NEW: append hidden_states even if we skip the layer (None in that case)        
            if output_hidden_states:      
                hidden_states.append(x.detach().clone() if not skip_layer else None)

        x = self.jk(xs) if hasattr(self, 'jk') else x
        x = self.lin(x) if hasattr(self, 'lin') else x
        
        if output_hidden_states and output_attention_weights and self.supports_attention_weights:
            return x, hidden_states, att_weights
        elif output_hidden_states:
            return x, hidden_states
        elif output_attention_weights and self.supports_attention_weights:
            return x, att_weights
        else:
            return x

    @torch.no_grad()
    def inference_per_layer(
        self,
        layer: int,
        x: Tensor,
        edge_index: Adj,
        batch_size: int,
    ) -> Tensor:

        x = self.convs[layer](x, edge_index)[:batch_size]

        if layer == self.num_layers - 1 and self.jk_mode is None:
            return x

        if self.act is not None and self.act_first:
            x = self.act(x)
        if self.norms is not None:
            x = self.norms[layer](x)
        if self.act is not None and not self.act_first:
            x = self.act(x)
        if layer == self.num_layers - 1 and hasattr(self, 'lin'):
            x = self.lin(x)

        return x

    @torch.no_grad()
    def inference(
        self,
        loader: NeighborLoader,
        device: Optional[Union[str, torch.device]] = None,
        embedding_device: Union[str, torch.device] = 'cpu',
        progress_bar: bool = False,
        cache: bool = False,
    ) -> Tensor:
        r"""Performs layer-wise inference on large-graphs using a
        :class:`~torch_geometric.loader.NeighborLoader`, where
        :class:`~torch_geometric.loader.NeighborLoader` should sample the
        full neighborhood for only one layer.
        This is an efficient way to compute the output embeddings for all
        nodes in the graph.
        Only applicable in case :obj:`jk=None` or `jk='last'`.

        Args:
            loader (torch_geometric.loader.NeighborLoader): A neighbor loader
                object that generates full 1-hop subgraphs, *i.e.*,
                :obj:`loader.num_neighbors = [-1]`.
            device (torch.device, optional): The device to run the GNN on.
                (default: :obj:`None`)
            embedding_device (torch.device, optional): The device to store
                intermediate embeddings on. If intermediate embeddings fit on
                GPU, this option helps to avoid unnecessary device transfers.
                (default: :obj:`"cpu"`)
            progress_bar (bool, optional): If set to :obj:`True`, will print a
                progress bar during computation. (default: :obj:`False`)
            cache (bool, optional): If set to :obj:`True`, caches intermediate
                sampler outputs for usage in later epochs.
                This will avoid repeated sampling to accelerate inference.
                (default: :obj:`False`)
        """
        assert self.jk_mode is None or self.jk_mode == 'last'
        assert isinstance(loader, NeighborLoader)
        assert len(loader.dataset) == loader.data.num_nodes
        assert len(loader.node_sampler.num_neighbors) == 1
        assert not self.training
        # assert not loader.shuffle  # TODO (matthias) does not work :(
        if progress_bar:
            pbar = tqdm(total=len(self.convs) * len(loader))
            pbar.set_description('Inference')

        x_all = loader.data.x.to(embedding_device)

        if cache:

            # Only cache necessary attributes:
            def transform(data: Data) -> Data:
                kwargs = dict(n_id=data.n_id, batch_size=data.batch_size)
                if hasattr(data, 'adj_t'):
                    kwargs['adj_t'] = data.adj_t
                else:
                    kwargs['edge_index'] = data.edge_index

                return Data.from_dict(kwargs)

            loader = CachedLoader(loader, device=device, transform=transform)

        for i in range(self.num_layers):
            xs: List[Tensor] = []
            for batch in loader:
                x = x_all[batch.n_id].to(device)
                batch_size = batch.batch_size
                if hasattr(batch, 'adj_t'):
                    edge_index = batch.adj_t.to(device)
                else:
                    edge_index = batch.edge_index.to(device)

                x = self.inference_per_layer(i, x, edge_index, batch_size)
                xs.append(x.to(embedding_device))

                if progress_bar:
                    pbar.update(1)

            x_all = torch.cat(xs, dim=0)

        if progress_bar:
            pbar.close()

        return x_all

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_layers={self.num_layers})')
