from .basic_gnn import BasicGNN
from typing import Final, Union, Tuple
from torch_geometric.nn.conv import TransformerConv, MessagePassing

class GraphTransformer(BasicGNN):

    supports_edge_attr: Final[bool] = True
    supports_edge_weight: Final[bool] = False
    supports_attention_weights: Final[bool] = True
    supports_norm_batch: Final[bool]
    supports_norm_edge_index: Final[bool]

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels : int, **kwargs) -> MessagePassing:
        
        heads = kwargs.pop("heads", 1)
        concat = kwargs.pop("concat", True)

        if getattr(self, "_is_conv_to_out", False):
            concat=False

        if concat and out_channels % heads != 0:
            raise ValueError(f"Ensure that the number of output channels of "
                             f"'TransformerConv' (got '{out_channels}') is divisible "
                             f"by the number of heads (got '{heads}')")
        
        if concat:
            out_channels = out_channels // heads

        return TransformerConv(in_channels, out_channels, heads=heads, concat=concat, dropout=self.dropout.p, **kwargs)

    