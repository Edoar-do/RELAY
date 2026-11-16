import torch
from loguru import logger
import joblib
from typing import List, Any
from hubert_ecg.hubert_ecg import HuBERTECG
from torchmetrics.classification import (
    MultilabelF1Score,
    MultilabelRecall,
    MultilabelPrecision,
    MultilabelSpecificity,
    MultilabelAUROC, 
)
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassRecall,
    MulticlassPrecision,
    MulticlassSpecificity,
    MulticlassAUROC, 
)
from torcheval.metrics import (
    MultilabelAUPRC,
    MulticlassAUPRC
)

from torch.utils.data import Sampler
import torch

class ResumableRandomSampler(Sampler):
    def __init__(self, data_source, batch_size, start_idx=0, generator=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.start_idx = start_idx # Number of SAMPLES to skip (not batches)
        self.generator = generator
    
    def __iter__(self):
            n = len(self.data_source)
            if self.generator is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
                generator = torch.Generator()
                generator.manual_seed(seed)
            else:
                generator = self.generator

            # 1. Keep as Tensor (Compact C++ memory)
            indices = torch.randperm(n, generator=generator)
            
            # 2. Slice the Tensor (Zero-copy view)
            indices = indices[self.start_idx:]
            
            # 3. Convert only the necessary part to Python list
            return iter(indices.tolist())
        
    def __len__(self):
        return len(self.data_source) - self.start_idx
    
    def set_epoch(self, epoch, start_idx=0):
        self.generator.manual_seed(epoch + 42)
        self.start_idx = start_idx 

def get_criterions(task, train_weigths=None, val_weights=None, label_smoothing=0.0):
    if task == "multi_class":
        return (torch.nn.CrossEntropyLoss(weight=train_weigths, label_smoothing=label_smoothing), torch.nn.CrossEntropyLoss(weight=val_weights))
    elif task == "multi_label":
        return (torch.nn.BCEWithLogitsLoss(weight=train_weigths),torch.nn.BCEWithLogitsLoss(weight=val_weights))
    elif task == 'regression':
        return (torch.nn.MSELoss(), torch.nn.MSELoss())
    else:
        raise NotImplementedError(f"There is no criterion for task {task} yet")
    
def get_metrics(task, num_classes):
    keys = ["f1-score", "recall", "specificity", "precision", "auroc", "auprc"]
    if task == "multi_class":
        values = [
            MulticlassF1Score(num_classes=num_classes, average=None),
            MulticlassRecall(num_classes=num_classes, average=None),
            MulticlassSpecificity(num_classes=num_classes, average=None),
            MulticlassPrecision(num_classes=num_classes, average=None),
            MulticlassAUROC(num_classes=num_classes, average=None),
            MulticlassAUPRC(num_classes=num_classes, average=None)
        ]
    elif task == "multi_label":
        values = [
            MultilabelF1Score(num_labels=num_classes, average=None),
            MultilabelRecall(num_labels=num_classes, average=None),
            MultilabelSpecificity(num_labels=num_classes, average=None),
            MultilabelPrecision(num_labels=num_classes, average=None),
            MultilabelAUROC(num_labels=num_classes, average=None),
            MultilabelAUPRC(num_labels=num_classes, average=None)
        ]
    elif task == 'regression':
        values = [] # todo
    else:
        raise NotImplementedError(f"There is no metric for task {task} yet")
    return {k : v for k, v in zip(keys, values)}

def retrieve_kmeans_models(kmeans_path : str) -> List[Any]:

    """
    Given a `kmeans_path` path to a file containing paths to kmeans models, this function returns a list of such models    
    """

    with open(kmeans_path, 'r') as f:
        kmeans_paths = f.readlines()
        
    # filter out commented lines
    kmeans_paths = [path for path in kmeans_paths if not path.startswith("#")]
    
    # logging just for testing purposes
    logger.info(f"Loading {len(kmeans_paths)} kmeans models...")
    for path in kmeans_paths:
        logger.info(f"Loading {path.strip()}...")
        
    return [joblib.load(path.strip()) for path in kmeans_paths]


def retrieve_hubert_ecg_conv_embedder(checkpoint_path: str) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    hubert_ecg = HuBERTECG(checkpoint['model_config'])
    conv_embeddder = hubert_ecg.feature_extractor
    return conv_embeddder

def entropy_profile(*args, **kwargs):
    raise NotImplementedError("This function is not implemented yet.")