from transformers.utils.generic import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple
import torch

@dataclass
class MaskedNodeModellingOutput(ModelOutput):
    """
    Class for masked node modelling output for GNNs.

    Args:
        `loss` : torch.FloatTensor for loss value
        `hidden_states` : Optional[torch.FloatTensor] per-layer node representations
        `logits` : Optional[torch.FloatTensor] for cosine similarities scores that act as per-node logits
        `mask` : Optional[torch.BoolTensor] mask of the node indices that were masked
    """

    loss: Optional[torch.FloatTensor] = None
    logits : Optional[torch.FloatTensor] = None
    hidden_states: Optional[torch.FloatTensor] = None
    mask : Optional[torch.BoolTensor] = None
    
@dataclass
class GraphClassificationOutput(ModelOutput):
    """
    Class for graph classification output for GNNs.
    Args:
        `loss` : torch.FloatTensor for loss value
        `hidden_states` : Optional[torch.FloatTensor] per-layer graph representations
        `logits` : Optional[torch.FloatTensor] for cosine similarities scores that act as per-graph logits
    """

    loss: Optional[torch.FloatTensor] = None
    logits : Optional[torch.FloatTensor] = None
    hidden_states: Optional[torch.FloatTensor] = None

@dataclass
class MaskedNodeModellingOuputWithAttention(ModelOutput):
    """
    Class for masked node modelling output with attention for GNNs.

    Args:
        `loss` : torch.FloatTensor for loss value
        `hidden_states` : Optional[torch.FloatTensor] per-layer node representations
        `logits` : Optional[torch.FloatTensor] for cosine similarities scores that act as per-node logits
        `mask_node_indices` : Optional[torch.BoolTensor] for indices of masked nodes in the input graph
        `attention_weights` : Optional[torch.FloatTensor] for attention weights applied to the nodes
    """

    loss: Optional[torch.FloatTensor] = None
    logits : Optional[torch.FloatTensor] = None
    hidden_states: Optional[torch.FloatTensor] = None
    mask : Optional[torch.BoolTensor] = None
    attention_weights: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None

@dataclass
class GraphClassificationOutputWithAttention(ModelOutput):
    """
    Class for graph classification output with attention for GNNs.

    Args:
        `loss` : torch.FloatTensor for loss value
        `hidden_states` : Optional[torch.FloatTensor] per-layer graph representations
        `logits` : Optional[torch.FloatTensor] for cosine similarities scores that act as per-graph logits
        `attention_weights` : Optional[torch.FloatTensor] for attention weights applied to the graph
    """

    loss: Optional[torch.FloatTensor] = None
    logits : Optional[torch.FloatTensor] = None
    hidden_states: Optional[torch.FloatTensor] = None
    attention_weights: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None