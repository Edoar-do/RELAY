from token import OP
from transformers import PretrainedConfig
from typing import *

class ECGGraphTransformerConfig(PretrainedConfig):

    model_type = "ECGGraphTransformerConfig"

    def __init__(
            self,
            # --- for GNN architecture ---
            in_channels : int = 512,
            hidden_channels : int = 512,
            out_channels : int = 512,
            num_layers : int = 12,
            dropout : Optional[float] = 0.0, 
            layerdrop : Optional[float] = 0.1,
            act : Optional[Union[str, Callable]] = 'gelu',
            act_first : Optional[bool]  = False,
            act_kwargs : Optional[Dict[str, Any]] = {},
            norm : Optional[Union[str, Callable]] = None,
            norm_kwargs : Optional[Dict[str, Any]] = {},
            jk : Optional[str] = None,
            bias : Optional[bool] = False,
            graph_transformer_kwargs : Optional[Dict[str, Any]] = {},
            # --- for other architecture components ---
            use_lead_encoding : Optional[bool]  = False,
            use_temporal_encoding : Optional[bool] = False,
            edge_drop_prob : Optional[float] = 0.0,
            # --- for signal-to-graph ---
            conv_kernel : Optional[Tuple[int]] = (10, 3, 3, 2),
            conv_stride : Optional[Tuple[int]]= (3, 2, 2, 2),
            conv_dim : Optional[Tuple[int]] = (512, 512, 512, 512),
            feat_extract_norm : Optional[str] = "layer",
            conv_bias = False,
            feat_extract_activation = 'gelu',
            nodes_per_lead : Optional[int] = 20,
            # --- for masked node modelling ---
            mask_node_prob : Optional[float] = None,
            min_node_masked : Optional[int] = 1,
            pseudo_vocab_sizes : Optional[List[int]] = None,
            pseudo_classifier_projection_size : Optional[int] = None,
            use_pseudo_label_embeddings : Optional[bool] = False,
            # --- for graph classification ---
            pooling : Optional[str] = 'global_mean',
            num_classes : Optional[int] = None,
            **kwargs
    ):

        super().__init__(**kwargs)
        
        if mask_node_prob is not None:
            if mask_node_prob < 0 or mask_node_prob > 1:
                raise ValueError(f'mask_node_prob must be in [0, 1], but got {mask_node_prob}')
            if use_pseudo_label_embeddings:
                if pseudo_vocab_sizes is None:
                    raise ValueError(f"If mask_node_prob is passed, then mask node modelling pre-training has to be performed, requiring pseudo_vocab_sizes to be passed. Received {pseudo_vocab_sizes}")
                if pseudo_classifier_projection_size is None:
                    raise ValueError(f"If mask_node_prob is passed, then mask node modelling pre-training has to be performed, requiring pseudo_classifier_projection_sizes to be passed. Received {pseudo_classifier_projection_size}")
            if min_node_masked < 1:
                raise ValueError("min_node_masked must be greater than 1 if masked node modelling is being performed")
        
        # --- for GNN architecture ---
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.layerdrop = layerdrop
        self.num_layers = num_layers
        self.act = act
        self.act_first = act_first
        self.act_kwargs = act_kwargs
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.jk = jk
        self.bias = bias
        self.graph_transformer_kwargs = graph_transformer_kwargs
        # --- for other architecture components ---
        self.use_lead_encoding = use_lead_encoding
        self.use_temporal_encoding = use_temporal_encoding
        self.edge_drop_prob = edge_drop_prob
        # --- for signal-to-graph ---
        self.conv_kernel = tuple(conv_kernel)
        self.conv_stride = tuple(conv_stride)
        self.conv_dim = tuple(conv_dim)
        self.feat_extract_norm = feat_extract_norm
        self.conv_bias = conv_bias
        self.feat_extract_activation = feat_extract_activation
        self.num_feat_extract_layers = len(conv_dim)
        self.nodes_per_lead = nodes_per_lead
        # --- for masked node modelling ---
        self.mask_node_prob = mask_node_prob
        self.min_node_masked = min_node_masked 
        self.pseudo_vocab_sizes = pseudo_vocab_sizes
        self.pseudo_classifier_projection_size = pseudo_classifier_projection_size
        self.use_pseudo_label_embeddings = use_pseudo_label_embeddings
        # for graph classification ---
        self.pooling = pooling
        self.num_classes = num_classes

class ECGMLPConfig(PretrainedConfig):

    model_type = "ECGMLPConfig"

    def __init__(
            self,
            # --- for GNN architecture ---
            in_channels : int = 512,
            hidden_channels : int = 512, 
            num_layers : int = 12,
            out_channels : Optional[int] = 512,
            dropout : Optional[float] = 0.0,
            layerdrop : Optional[float] = 0.1,
            act : Optional[Union[str, Callable]] = 'gelu',
            act_first : Optional[bool]  = False,
            act_kwargs : Optional[Dict[str, Any]] = {},
            norm : Optional[Union[str, Callable]] = None,
            norm_kwargs : Optional[Dict[str, Any]] = {},
            plain_last : Optional[bool] = True,
            bias : Optional[bool] = False,
            mlp_kwargs : Optional[Dict[str, Any]] = {},
            # --- for other architecture components ---
            use_lead_encoding : Optional[bool]  = False, 
            use_temporal_encoding : Optional[bool] = False,
            edge_drop_prob : Optional[float] = 0.0,
            # --- for signal-to-graph ---
            conv_kernel : Optional[Tuple[int]] = (10, 3, 3, 2),
            conv_stride : Optional[Tuple[int]]= (3, 2, 2, 2),
            conv_dim : Optional[Tuple[int]] = (512, 512, 512, 512),
            feat_extract_norm : Optional[str] = "layer",
            conv_bias = False,
            feat_extract_activation = 'gelu',
            nodes_per_lead : Optional[int] = 20,
            # --- for masked node modelling ---
            mask_node_prob : Optional[float] = None,
            min_node_masked : Optional[int] = 1,
            pseudo_vocab_sizes : Optional[List[int]] = None,
            pseudo_classifier_projection_size : Optional[int] = None,
            use_pseudo_label_embeddings : Optional[bool] = False,
            # --- for graph classification ---
            pooling : Optional[str] = 'global_mean',
            num_classes : Optional[int] = None,
            **kwargs
    ):
        super().__init__(**kwargs)  

        if mask_node_prob is not None:
            if mask_node_prob < 0 or mask_node_prob > 1:
                raise ValueError(f'mask_node_prob must be in [0, 1], but got {mask_node_prob}')
            if use_pseudo_label_embeddings:
                if pseudo_vocab_sizes is None:
                    raise ValueError(f"If mask_node_prob is passed, then mask node modelling pre-training has to be performed, requiring pseudo_vocab_sizes to be passed. Received {pseudo_vocab_sizes}")
                if pseudo_classifier_projection_size is None:
                    raise ValueError(f"If mask_node_prob is passed, then mask node modelling pre-training has to be performed, requiring pseudo_classifier_projection_sizes to be passed. Received {pseudo_classifier_projection_size}")
            if min_node_masked < 1:
                raise ValueError(f"min_node_maks argument must be greater than 1 if masked node modelling is being performed")
        
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.dropout = dropout
        self.layerdrop = layerdrop
        self.act = act
        self.act_first = act_first
        self.act_kwargs = act_kwargs
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias
        self.plain_last = plain_last 
        self.mlp_kwargs = mlp_kwargs
        # --- for other architecture components ---
        self.use_lead_encoding = use_lead_encoding
        self.use_temporal_encoding = use_temporal_encoding
        self.edge_drop_prob = edge_drop_prob
        # --- for signal-to-grap ---
        self.conv_kernel = tuple(conv_kernel)
        self.conv_stride = tuple(conv_stride)
        self.conv_dim = tuple(conv_dim)
        self.feat_extract_norm = feat_extract_norm
        self.conv_bias = conv_bias
        self.feat_extract_activation = feat_extract_activation
        self.num_feat_extract_layers = len(conv_dim)
        self.nodes_per_lead = nodes_per_lead
        # --- for masked node modelling ---
        self.mask_node_prob = mask_node_prob
        self.min_node_masked = min_node_masked  
        self.pseudo_vocab_sizes = pseudo_vocab_sizes
        self.pseudo_classifier_projection_size = pseudo_classifier_projection_size
        self.use_pseudo_label_embeddings = use_pseudo_label_embeddings
        # --- for graph classification ---
        self.pooling = pooling
        self.num_classes = num_classes      

class ECGGINConfig(PretrainedConfig):

    model_type = "ECGGINConfig"

    def __init__(
            self,
            # --- for GNN architecture ---
            in_channels : int = 512,
            hidden_channels : int = 512, 
            num_layers : int = 12,
            out_channels : Optional[int] = None,
            dropout : Optional[float] = 0.0,
            layerdrop : Optional[float] = 0.1,
            act : Optional[Union[str, Callable]] = 'gelu',
            act_first : Optional[bool]  = False,
            act_kwargs : Optional[Dict[str, Any]] = {},
            norm : Optional[Union[str, Callable]] = None,
            norm_kwargs : Optional[Dict[str, Any]] = {},
            jk : Optional[str] = None,
            bias : Optional[bool] = False,
            ginConv_kwargs : Optional[Dict[str, Any]] = {},
            # --- fot other architecture components ---
            use_lead_encoding : Optional[bool]  = False,
            use_temporal_encoding : Optional[bool] = False,
            edge_drop_prob : Optional[float] = 0.0,
            # --- for signal-to-graph ---
            conv_kernel : Optional[Tuple[int]] = (10, 3, 3, 2),
            conv_stride : Optional[Tuple[int]]= (3, 2, 2, 2),
            conv_dim : Optional[Tuple[int]] = (512, 512, 512, 512),
            feat_extract_norm : Optional[str] = "layer",
            conv_bias = False,
            feat_extract_activation = 'gelu',
            nodes_per_lead : Optional[int] = 20,
            # --- for masked node modelling ---
            mask_node_prob : Optional[float] = None,
            min_node_masked : Optional[int] = 1,
            pseudo_vocab_sizes : Optional[List[int]] = None,
            pseudo_classifier_projection_size : Optional[int] = None,
            use_pseudo_label_embeddings : Optional[bool] = False,
            # --- for graph classification ---
            pooling : Optional[str] = 'global_mean',
            num_classes : Optional[int] = None,
            **kwargs
    ):

        super().__init__(**kwargs)
    
        
        if mask_node_prob is not None:
            if mask_node_prob < 0 or mask_node_prob > 1:
                raise ValueError(f'mask_node_prob must be in [0, 1], but got {mask_node_prob}')
            if use_pseudo_label_embeddings:
                if pseudo_vocab_sizes is None:
                    raise ValueError(f"If mask_node_prob is passed, then mask node modelling pre-training has to be performed, requiring pseudo_vocab_sizes to be passed. Received {pseudo_vocab_sizes}")
                if pseudo_classifier_projection_size is None:
                    raise ValueError(f"If mask_node_prob is passed, then mask node modelling pre-training has to be performed, requiring pseudo_classifier_projection_sizes to be passed. Received {pseudo_classifier_projection_size}")
            if min_node_masked < 1:
                raise ValueError(f"min_node_maks argument must be greater than 1 if masked node modelling is being performed")
        
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.dropout = dropout
        self.layerdrop = layerdrop
        self.act = act
        self.act_first = act_first
        self.act_kwargs = act_kwargs
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.jk = jk 
        self.bias = bias
        self.ginConv_kwargs = ginConv_kwargs
        # --- for other architecture components ---
        self.use_lead_encoding = use_lead_encoding
        self.use_temporal_encoding = use_temporal_encoding
        self.edge_drop_prob = edge_drop_prob
        # --- for signal-to-grap ---
        self.conv_kernel = tuple(conv_kernel)
        self.conv_stride = tuple(conv_stride)
        self.conv_dim = tuple(conv_dim)
        self.feat_extract_norm = feat_extract_norm
        self.conv_bias = conv_bias
        self.feat_extract_activation = feat_extract_activation
        self.num_feat_extract_layers = len(conv_dim)
        self.nodes_per_lead = nodes_per_lead
        # --- for masked node modelling ---
        self.mask_node_prob = mask_node_prob
        self.min_node_masked = min_node_masked  
        self.pseudo_vocab_sizes = pseudo_vocab_sizes
        self.pseudo_classifier_projection_size = pseudo_classifier_projection_size
        self.use_pseudo_label_embeddings = use_pseudo_label_embeddings
        # --- for graph classification ---
        self.pooling = pooling
        self.num_classes = num_classes  

class ECGGATConfig(PretrainedConfig):

    model_type = "ECGGATConfig"

    def __init__(
            self,
            # --- for GNN architecture ---
            in_channels: int = 512,
            hidden_channels: int = 512,
            num_layers: int = 12,
            out_channels: Optional[int] = None,
            dropout: float = 0.0,
            layerdrop : Optional[float] = 0.1,
            act: Optional[Union[str, Callable]] = 'gelu',
            act_first: bool = False,
            act_kwargs: Optional[Dict[str, Any]] = {},
            norm: Optional[Union[str, Callable]] = None,
            norm_kwargs: Optional[Dict[str, Any]] = {},
            jk: Optional[str] = None,
            bias : Optional[bool] = False,
            gat_kwargs : Optional[Dict[str, Any]] = {},
            # --- for other architecture components ---
            use_lead_encoding : Optional[bool]  = False, 
            use_temporal_encoding : Optional[bool] = False,
            edge_drop_prob : Optional[float] = 0.0,
            # --- for signal-to-graph ---
            conv_kernel : Optional[Tuple[int]] = (10, 3, 3, 2),
            conv_stride : Optional[Tuple[int]]= (3, 2, 2, 2),
            conv_dim : Optional[Tuple[int]] = (512, 512, 512, 512),
            feat_extract_norm : Optional[str] = "layer",
            conv_bias = False,
            feat_extract_activation = 'gelu',
            nodes_per_lead : Optional[int] = 20,
            # --- for masked node modelling ---
            mask_node_prob : Optional[float] = None,
            min_node_masked : Optional[int] = 1,
            pseudo_vocab_sizes : Optional[List[int]] = None,
            pseudo_classifier_projection_size : Optional[int] = None,
            use_pseudo_label_embeddings : Optional[bool] = False,
            # --- for graph classification ---
            pooling : Optional[str] = 'global_mean',
            num_classes : Optional[int] = None,
            linear_head : Optional[bool] = True,
            classifier_dropout : Optional[float] = 0.0,   
            **kwargs
    ):

        super().__init__(**kwargs)
        
        if mask_node_prob is not None:
            if mask_node_prob < 0 or mask_node_prob > 1:
                raise ValueError(f'mask_node_prob must be in [0, 1], but got {mask_node_prob}')
            if use_pseudo_label_embeddings:
                if pseudo_vocab_sizes is None:
                    raise ValueError(f"If mask_node_prob is passed, then mask node modelling pre-training has to be performed, requiring pseudo_vocab_sizes to be passed. Received {pseudo_vocab_sizes}")
                if pseudo_classifier_projection_size is None:
                    raise ValueError(f"If mask_node_prob is passed, then mask node modelling pre-training has to be performed, requiring pseudo_classifier_projection_sizes to be passed. Received {pseudo_classifier_projection_size}")
            if min_node_masked < 1:
                raise ValueError("min_node_masked must be greater than 1 if masked node modelling is performed")
        
        # --- for GNN architecture ---
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.dropout = dropout
        self.layerdrop = layerdrop
        self.act = act
        self.act_first = act_first
        self.act_kwargs = act_kwargs
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.jk = jk
        self.bias = bias
        self.gat_kwargs = gat_kwargs
        # --- for other architecture components ---
        self.use_lead_encoding = use_lead_encoding
        self.use_temporal_encoding = use_temporal_encoding
        self.edge_drop_prob = edge_drop_prob
        # --- for signal-to-graph ---
        self.conv_kernel = tuple(conv_kernel)
        self.conv_stride = tuple(conv_stride)
        self.conv_dim = tuple(conv_dim)
        self.feat_extract_norm = feat_extract_norm
        self.conv_bias = conv_bias
        self.feat_extract_activation = feat_extract_activation
        self.num_feat_extract_layers = len(conv_dim)
        self.nodes_per_lead = nodes_per_lead
        # --- for masked node modelling ---
        self.mask_node_prob = mask_node_prob
        self.min_node_masked = min_node_masked 
        self.pseudo_vocab_sizes = pseudo_vocab_sizes
        self.pseudo_classifier_projection_size = pseudo_classifier_projection_size
        self.use_pseudo_label_embeddings = use_pseudo_label_embeddings
        # --- for graph classification ---
        self.pooling = pooling
        self.num_classes = num_classes
        self.linear_head = linear_head
        self.classifier_dropout = classifier_dropout
        
class ECGGCNConfig(PretrainedConfig):
    
    model_type = "ECGGCNConfig"

    def __init__(
            self,
            # --- for GNN architecture ---
            in_channels: int = 512,
            hidden_channels: int = 512,
            num_layers: int = 12,
            out_channels: Optional[int] = None,
            dropout: float = 0.0,
            layerdrop : Optional[float] = 0.1,
            act: Optional[Union[str, Callable]] = 'gelu',
            act_first: bool = False,
            act_kwargs: Optional[Dict[str, Any]] = {},
            norm: Optional[Union[str, Callable]] = None,
            norm_kwargs: Optional[Dict[str, Any]] = {},
            jk: Optional[str] = None,
            gcn_kwargs : Optional[Dict[str, Any]] = {},
            # --- for other architecture components ---
            use_lead_encoding : Optional[bool]  = False, 
            use_temporal_encoding : Optional[bool] = False,
            edge_drop_prob : Optional[float] = 0.0,
            # --- for signal-to-graph ---
            conv_kernel : Optional[Tuple[int]] = (10, 3, 3, 2),
            conv_stride : Optional[Tuple[int]]= (3, 2, 2, 2),
            conv_dim : Optional[Tuple[int]] = (512, 512, 512, 512),
            feat_extract_norm : Optional[str] = "layer",
            conv_bias : Optional[bool] = False,
            feat_extract_activation : str = 'gelu',
            nodes_per_lead : Optional[int] = 20,
            # --- for masked node modelling ---
            mask_node_prob : Optional[float] = None,
            min_node_masked : Optional[int] = 1,
            pseudo_vocab_sizes : Optional[List[int]] = None,
            pseudo_classifier_projection_size : Optional[int] = None,
            use_pseudo_label_embeddings : Optional[bool] = False,
            # --- for graph classification ---
            pooling : Optional[str] = 'global_mean',
            num_classes : Optional[int] = None,
            **kwargs
    ):

        super().__init__(**kwargs)
        
        if mask_node_prob is not None:
            if mask_node_prob < 0 or mask_node_prob > 1:
                raise ValueError(f'mask_node_prob must be in [0, 1], but got {mask_node_prob}')
            if use_pseudo_label_embeddings:
                if pseudo_vocab_sizes is None:
                    raise ValueError(f"If mask_node_prob is passed, then mask node modelling pre-training has to be performed, requiring pseudo_vocab_sizes to be passed. Received {pseudo_vocab_sizes}")
                if pseudo_classifier_projection_size is None:
                    raise ValueError(f"If mask_node_prob is passed, then mask node modelling pre-training has to be performed, requiring pseudo_classifier_projection_sizes to be passed. Received {pseudo_classifier_projection_size}")
            if min_node_masked < 1:
                raise ValueError("min_node_masked must be greater than 1 if masked node modelling is performed")    
            
        # --- for GNN architecture ---
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.dropout = dropout
        self.layerdrop = layerdrop
        self.act = act
        self.act_first = act_first
        self.act_kwargs = act_kwargs
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.jk = jk
        self.gcn_kwargs = gcn_kwargs
        # --- for other architecture components ---
        self.use_lead_encoding = use_lead_encoding
        self.use_temporal_encoding = use_temporal_encoding
        self.edge_drop_prob = edge_drop_prob
        # --- for signal-to-graph ---
        self.conv_kernel = tuple(conv_kernel)
        self.conv_stride = tuple(conv_stride)
        self.conv_dim = tuple(conv_dim)
        self.feat_extract_norm = feat_extract_norm
        self.conv_bias = conv_bias
        self.feat_extract_activation = feat_extract_activation
        self.num_feat_extract_layers = len(conv_dim)
        self.nodes_per_lead = nodes_per_lead
        # --- for masked node modelling ---
        self.mask_node_prob = mask_node_prob
        self.min_node_masked = min_node_masked
        self.pseudo_vocab_sizes = pseudo_vocab_sizes
        self.pseudo_classifier_projection_size = pseudo_classifier_projection_size
        self.use_pseudo_label_embeddings = use_pseudo_label_embeddings
        # --- for graph classification ---
        self.pooling = pooling
        self.num_classes = num_classes
       


# if __name__ == '__main__':

#     # NOTE: this __main__ is supposed to be called at the end of the project to dump configuration file in json format.
#     # This is the first step for pushing a pretrained model to the hub as well as ensuring fast reproducibility

#     parser = argparse.ArgumentParser()

#     parser.add_argument("model_type", 
#                         type=str, 
#                         help="Model type for HF Config",
#                         choices=["ECGTransformerConfig", "ECGGINConfig", "ECGGATConfig"]
#                         )
    
#     parser.add_argument("config_dir_path", 
#                         type=str,
#                         help="Path to dir to dump Config obj in JSON format",
#                         )

#     args = parser.parse_args()

#     if not os.path.exists(args.config_dir_path):
#         raise ValueError(f"Arg config_path {args.config_dir_path} does not exist")

#     if args.model_type == "ECGTransformerConfig":
#         config = ECGGraphTransformerConfig() # fill in with BEST args
#         config.save_pretrained(args.config_dir_path)
#         ...
#     elif args.model_type == "ECGGINConfig":
#         config = ECGGINConfig() # fill in with BEST args
#         config.save_pretrained(args.config_dir_path)
#         ...
#     else:
#         config = ECGGATConfig() # fill in with BEST args
#         config.save_pretrained(args.config_dir_path)
#         ...

#     logger.info(f"Dumped configuration file at {args.config_dir_path}")

MODEL_NAME_2_CONFIG = {
    'gin' : ECGGINConfig,
    'gat' : ECGGATConfig,
    'graph_transformer' : ECGGraphTransformerConfig,
    'mlp' : ECGMLPConfig,
}



