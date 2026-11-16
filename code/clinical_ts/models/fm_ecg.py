from dataclasses import dataclass
from itertools import chain
# PROPOSED: import os  # needed for the is_1lead detection block in ECGFounderWrapper.__init__
import os

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from safetensors import safe_open

from clinical_ts.utils.heads import LearnableQueryAttentionPoolingHead, LearnableQueryAttentionPoolingHeadConfig

from main_lite_base import FMWrapperBase

from clinical_ts.models.ecg_foundation_models.ecg_founder import Net1D
from clinical_ts.models.ecg_foundation_models.ecg_jepa.ecg_jepa import ecg_jepa
from clinical_ts.models.ecg_foundation_models.ecg_jepa.ecg_jepa_utils import load_encoder
from clinical_ts.models.ecg_foundation_models.st_mem.st_mem import st_mem_vit_base_dec256d4b

from clinical_ts.models.ecg_foundation_models.merl.resnet1d import ResNet18
from clinical_ts.models.ecg_foundation_models.merl.vit1d import vit_tiny

from clinical_ts.models.ecg_foundation_models.ecgfm_ked import xresnet1d101

from clinical_ts.models.ecg_foundation_models.hubert_ecg.hubert_ecg import HuBERTECG
from clinical_ts.models.ecg_foundation_models.hubert_ecg.hubert_ecg_classification import HuBERTForECGClassification
from clinical_ts.models.ecg_foundation_models.hubert_ecg.utils import ecg_preprocessing
from clinical_ts.models.ecg_foundation_models.hubert_ecg.config import hubert_config, HUBERT_CONFIGS

from clinical_ts.models.ecg_foundation_models.ecg_cpc.basic_io import load_model_from_config


class ECGFounderWrapper(FMWrapperBase):
    """
        Paper: https://arxiv.org/abs/2410.04133
        Code: https://github.com/PKUDigitalHealth/ECGFounder
        Checkpoints: https://huggingface.co/PKUDigitalHealth/ECGFounder/tree/main
        Model sampling frequency: 500 Hz
        Pretraining dataset: HEEDB
    """
    def __init__(self, num_classes, num_output_tokens, pretrained_path=None, eval_mode="finetuning_linear", lr=1e-3, discriminative_lr_factor=0.1):
        super().__init__(num_classes, num_output_tokens)

        assert eval_mode in ["finetuning_linear", "finetuning_nonlinear", "frozen", "linear"]
        self.eval_mode = eval_mode
        self.lr = lr
        self.discriminative_lr_factor = discriminative_lr_factor

        # --- PROPOSED CHANGE: detect 1-lead checkpoint and set in_channels accordingly ---
        # Add `import os` at the top of this file before enabling these lines.
        is_1lead = pretrained_path is not None and "1_lead" in os.path.basename(pretrained_path)
        self.is_1lead = is_1lead
        _in_channels = 1 if is_1lead else 12
        # --- END PROPOSED CHANGE ---

        self.model = Net1D(
            # in_channels=12,   # PROPOSED: replace 12 with _in_channels (see block above)
            in_channels=_in_channels,
            base_filters=64,
            ratio=1,
            filter_list=[64, 160, 160, 400, 400, 1024, 1024],
            m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
            kernel_size=16,
            stride=2,
            groups_width=16,
            verbose=False,
            use_bn=False,
            use_do=False,
            n_classes=num_classes
        )

        if pretrained_path:
            checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=False)
            state_dict = {k: v for k, v in checkpoint["state_dict"].items() if not k.startswith("dense.")}
            self.model.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError("ECGFounderWrapper requires a valid `pretrained_path` to load the encoder.")


        self.feature_dim = self.model.dense.in_features
        self.model.dense = nn.Identity()

        # Nonlinear head configurations
        nonlinear_head_config = LearnableQueryAttentionPoolingHeadConfig(
            multi_prediction=False,
            heads=16,
            bias=False
        )
        
        @dataclass
        class InputShape:
            channels: int
            length: int
            static_dim: int

        input_shape = InputShape(channels=self.feature_dim, length=0, static_dim=0)
        
        self.nonlinear_head = LearnableQueryAttentionPoolingHead(
            hparams_head=nonlinear_head_config,
            hparams_input_shape=input_shape,
            target_dim=num_classes
        )
        
        if self.eval_mode == "finetuning_linear":
            self.head = nn.Linear(self.feature_dim, num_classes)
        elif self.eval_mode == "finetuning_nonlinear":
            self.head = self.nonlinear_head
        elif self.eval_mode == "frozen":
            self.head = self.nonlinear_head
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()        
        else:
            self.head = nn.Linear(self.feature_dim, num_classes)
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

        self._override_forward()

    def _override_forward(self):
        def custom_forward(x):
            out = x
            # First conv
            out = self.model.first_conv(out)
            if self.model.use_bn:
                out = self.model.first_bn(out)
            out = self.model.first_activation(out)

            # Stages
            for i_stage in range(self.model.n_stages):
                net = self.model.stage_list[i_stage]
                out = net(out)

            sequence_features = out  # (batch, channels, seq_len)
            pooled_features = out.mean(-1)  # GAP
            return sequence_features, pooled_features

        self.model.forward = custom_forward
    
    def get_params(self):
        head_params = list(self.head.parameters())

        if self.eval_mode in ["frozen", "linear"]:
            return [{"params": head_params, "lr": self.lr}]    
        
        encoder_params = []
        predictor_params = []
        first_conv_bn_params = []
        
        for name, param in self.model.named_parameters():
            if name.startswith(("first_conv", "first_bn")):
                first_conv_bn_params.append(param)
            elif any(name.startswith(f"stage_list.{i}.") for i in [0, 1, 2]):
                encoder_params.append(param)
            elif any(name.startswith(f"stage_list.{i}.") for i in [3, 4, 5, 6]):
                predictor_params.append(param)
        
        encoder_params.extend(first_conv_bn_params)

        print("Returning layer dependent learning rate from ECGFounderWrapper...")
        
        return [
            {"params": head_params, "lr": self.lr},
            {"params": predictor_params, "lr": self.lr * self.discriminative_lr_factor},
            {"params": encoder_params, "lr": self.lr * self.discriminative_lr_factor * self.discriminative_lr_factor}
        ]

    def forward(self, x, **kwargs):
        x = torch.nan_to_num(x)
        sequence_features, pooled_features = self.model(x)

        if self.eval_mode in ["frozen", "finetuning_nonlinear"]:
            seq_feats = sequence_features.transpose(1, 2)  # (batch, seq_len, channels)
            output_dict = self.head(seq=seq_feats)
            x = output_dict["seq"]  
        else:
            x = self.head(pooled_features)

        return torch.nan_to_num(x)


class ECGJEPAWrapper(FMWrapperBase):
    """
        Paper: https://arxiv.org/abs/2410.08559
        Code: https://github.com/sehunfromdaegu/ECG_JEPA
        Checkpoints: https://drive.google.com/file/d/1mh-XL0XOvvhFbhvuZ9c2KnTHa9B4F3Wx/view
                     https://drive.google.com/file/d/1gMOT4xjQQg0GZkY1iE6NuDzua4ALw00l/view
        Model sampling frequency: 250 Hz
        Pretraining dataset: ptb-xl, cpsc2018
        Note: 2 checkpoints available; one for random masking and other for multi-block masking
    """
    def __init__(self, num_classes, num_output_tokens, pretrained_path=None, eval_mode="finetuning_linear", lr=1e-3, discriminative_lr_factor=0.1):
        super().__init__(num_classes, num_output_tokens)

        assert eval_mode in ["finetuning_linear", "finetuning_nonlinear", "frozen", "linear"]
        self.eval_mode = eval_mode
        self.lr = lr
        self.discriminative_lr_factor = discriminative_lr_factor

        if pretrained_path:
            self.encoder, self.feature_dim = load_encoder(pretrained_path)
        else:
            raise ValueError("ECGJepaWrapper requires a valid `pretrained_path` to load the encoder.")
        
        # Nonlinear head configurations
        nonlinear_head_config = LearnableQueryAttentionPoolingHeadConfig(
            multi_prediction=False,
            heads=16,
            bias=False
        )
        
        @dataclass
        class InputShape:
            channels: int
            length: int
            static_dim: int

        input_shape = InputShape(channels=self.feature_dim, length=0, static_dim=0)
        
        self.nonlinear_head = LearnableQueryAttentionPoolingHead(
            hparams_head=nonlinear_head_config,
            hparams_input_shape=input_shape,
            target_dim=num_classes
        )
        
        if self.eval_mode == "finetuning_linear":
            self.head = nn.Linear(self.feature_dim, num_classes)
        elif self.eval_mode == "finetuning_nonlinear":
            self.head = self.nonlinear_head
        elif self.eval_mode == "frozen":
            self.head = self.nonlinear_head
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()      
        else:
            self.head = nn.Linear(self.feature_dim, num_classes)
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()
        
        self._override_representation()
    
    def get_params(self):
        head_params = list(self.head.parameters())

        if self.eval_mode in ["frozen", "linear"]:
            return [{"params": head_params, "lr": self.lr}]
        
        encoder_params = []
        predictor_params = []
        
        for name, param in self.encoder.named_parameters():
            if name in ["pos_embed", "W_P.weight", "W_P.bias"]:
                encoder_params.append(param)
            elif any(name.startswith(f"encoder_blocks.blocks.{i}.") for i in range(3)):
                encoder_params.append(param)
            elif any(name.startswith(f"encoder_blocks.blocks.{i}.") for i in range(3, 12)):
                predictor_params.append(param)
            elif name.startswith(("norm.weight", "norm.bias")):
                predictor_params.append(param)
        
        print("Returning layer dependent learning rate from ECGJepaWrapper (Multiblock)...")

        return [
            {"params": head_params, "lr": self.lr},
            {"params": predictor_params, "lr": self.lr * self.discriminative_lr_factor},
            {"params": encoder_params, "lr": self.lr * self.discriminative_lr_factor * self.discriminative_lr_factor}
        ]

    def _override_representation(self):
        def custom_representation(x):
            assert x.dim() == 3, f"Input should be of dimension 3, x.dim()={x.dim()}"
            assert x.shape[1] == len(self.encoder.leads), f"lead error"
            assert x.shape[2] <= 2500, f"Input too long: expected at most 2500 timesteps, got {x.shape[2]}"
            # Zero-pad to exactly 2500 timesteps if shorter (e.g. 8s heart sound recordings → 2000 after resample)
            if x.shape[2] < 2500:
                x = torch.nn.functional.pad(x, (0, 2500 - x.shape[2]))

            pos_embed = self.encoder.pos_embed
            attention_mask = self.encoder._cross_attention_mask().to(x.device) # (c*p, c*p)

            # restric leads
            if len(self.encoder.leads) < self.encoder.c:
                pos_embed = self.encoder.restrict_leads(pos_embed, type="vector")
                attention_mask = self.encoder.restrict_leads(attention_mask, type="matrix")

            bs, l, _ = x.shape
            x = x.reshape(bs, -1, 50)  # (bs,l,2500) -> (bs,l*p,50)
            x = self.encoder.W_P(x)  # (bs,l*p,50) -> (bs,l*p,embed_dim)

            x = self.encoder.encoder_blocks(x, pos_embed, attention_mask)
            
            if self.encoder.norm is not None:
                sequence_features = self.encoder.norm(x)

            # GAP
            pooled_features = torch.mean(sequence_features, dim=1) # (bs,l*50,embed_dim) -> (bs,embed_dim)
            
            return sequence_features, pooled_features
        
        self.encoder.representation = custom_representation
    
    def forward(self, x, **kwargs):
        # ECG-JEPA takes exactly 2500 time steps. 
        # ECG-JEPA uses 8-channels: I, II, V1, V2, V3, V4, V5, V6
        # Our dataset uses 12-channels: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
        selected_indices = [0, 1, 6, 7, 8, 9, 10, 11]
        x = torch.nan_to_num(x)
        x = x[:, selected_indices, :]

        sequence_features, pooled_features = self.encoder.representation(x)

        if self.eval_mode in ["frozen", "finetuning_nonlinear"]:
            output_dict = self.head(seq=sequence_features)
            x = output_dict["seq"]
        else:
            x = self.head(pooled_features)

        return torch.nan_to_num(x)

class ECG_JEPA_Scratch(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = ecg_jepa(
            encoder_embed_dim=768,
            encoder_num_heads=16,
            predictor_embed_dim=384,
            predictor_depth=6,
            predictor_num_heads=12,
            mask_scale=(0, 0)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x, **kwargs):
        # ECG-JEPA takes exactly 2500 time steps. 
        # ECG-JEPA uses 8-channels: I, II, V1, V2, V3, V4, V5, V6
        # Our dataset uses 12-channels: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
        selected_indices = [0, 1, 6, 7, 8, 9, 10, 11]
        x = torch.nan_to_num(x)
        x = x[:, selected_indices, :]

        features = self.model.encoder.representation(x)
        output = self.classifier(features)

        return torch.nan_to_num(output)


class StMemWrapper(FMWrapperBase):
    """
        Paper: https://arxiv.org/abs/2402.09450
        Code: https://github.com/bakqui/ST-MEM
        Checkpoints: https://drive.google.com/file/d/1E7J-A1HqWa2f08T6Sfk5uWk-_CFJhOYQ/view?usp=share_link
        Model sampling frequency: 250 Hz
        Pretraining dataset: chapman, ningbo, code15
    """
    def __init__(self, num_classes, num_output_tokens, pretrained_path=None, eval_mode="finetuning_linear", lr=1e-3, discriminative_lr_factor=0.1, native_leads=False):
        super().__init__(num_classes, num_output_tokens)

        assert eval_mode in ["finetuning_linear", "finetuning_nonlinear", "frozen", "linear"]
        self.eval_mode = eval_mode
        self.lr = lr
        self.discriminative_lr_factor = discriminative_lr_factor
        # native_leads: encode only the leads that carry signal, with their true lead embeddings,
        # instead of encoding the zeroed leads too (see forward)
        self.native_leads = native_leads
        if native_leads:
            print("StMemWrapper: native lead mode, zeroed leads are dropped before encoding")

        self.model = st_mem_vit_base_dec256d4b(seq_len=2250, patch_size=75, num_leads=12)

        if pretrained_path:
            checkpoint = torch.load(pretrained_path, map_location="cpu")
            state_dict = checkpoint["model"]
            self.model.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError("StMemWrapper requires a valid `pretrained_path` to load the encoder.")
        
        self.feature_dim = self.model.encoder.width

        # Nonlinear head configurations
        nonlinear_head_config = LearnableQueryAttentionPoolingHeadConfig(
            multi_prediction=False,
            heads=16,
            bias=False
        )
        
        @dataclass
        class InputShape:
            channels: int
            length: int
            static_dim: int

        input_shape = InputShape(channels=self.feature_dim, length=0, static_dim=0)
        
        self.nonlinear_head = LearnableQueryAttentionPoolingHead(
            hparams_head=nonlinear_head_config,
            hparams_input_shape=input_shape,
            target_dim=num_classes
        )
        
        if self.eval_mode == "finetuning_linear":
            self.head = nn.Linear(self.feature_dim, num_classes)
        elif self.eval_mode == "finetuning_nonlinear":
            self.head = self.nonlinear_head
        elif self.eval_mode == "frozen":
            self.head = self.nonlinear_head
            for p in self.model.encoder.parameters():
                p.requires_grad = False
            self.model.encoder.eval()      
        else:
            self.head = nn.Linear(self.feature_dim, num_classes)
            for p in self.model.encoder.parameters():
                p.requires_grad = False
            self.model.encoder.eval()
        
        self._override_forward_encoding()
    
    def get_params(self):
        head_params = list(self.head.parameters())

        if self.eval_mode in ["frozen", "linear"]:
            return [{"params": head_params, "lr": self.lr}]
        
        encoder_params = []
        predictor_params = []
        
        for name, param in self.model.encoder.named_parameters():
            if name in ["encoder.pos_embedding", "encoder.sep_embedding"]:
                encoder_params.append(param)
            elif name.startswith(("encoder.lead_embeddings", "encoder.to_patch_embedding")):
                encoder_params.append(param)
            elif any(name.startswith(f"encoder.block{i}.") for i in [0, 1, 2]):
                encoder_params.append(param)
            elif any(name.startswith(f"encoder.block{i}.") for i in range(3, 12)):
                predictor_params.append(param)            
            elif name.startswith(("encoder.norm", "encoder.dropout")):
                predictor_params.append(param)
        
        print("Returning layer dependent learning rate from StMemWrapper...")

        return [
            {"params": head_params, "lr": self.lr},
            {"params": predictor_params, "lr": self.lr * self.discriminative_lr_factor},
            {"params": encoder_params, "lr": self.lr * self.discriminative_lr_factor * self.discriminative_lr_factor}
        ]

    def _override_forward_encoding(self):
        def custom_forward_encoding(x, lead_idx=None):
            # lead_idx: optional (b, num_leads) long tensor with the true lead index of each input
            # channel. If None, channel i gets lead embedding i (original behaviour, 12 channels).
            num_leads = x.shape[1]
            if num_leads > len(self.model.encoder.lead_embeddings):
                raise ValueError(f'Number of leads ({num_leads}) exceeds the number of lead embeddings')

            x = self.model.encoder.to_patch_embedding(x)
            b, _, n, _ = x.shape
            x = x + self.model.encoder.pos_embedding[:, 1:n + 1, :].unsqueeze(1)

            # lead indicating modules
            sep_embedding = self.model.encoder.sep_embedding[None, None, None, :]
            left_sep = sep_embedding.expand(b, num_leads, -1, -1) + self.model.encoder.pos_embedding[:, :1, :].unsqueeze(1)
            right_sep = sep_embedding.expand(b, num_leads, -1, -1) + self.model.encoder.pos_embedding[:, -1:, :].unsqueeze(1)
            x = torch.cat([left_sep, x, right_sep], dim=2)
            lead_embedding_table = torch.stack([lead_embedding for lead_embedding in self.model.encoder.lead_embeddings]) # (12, width)
            if lead_idx is None:
                lead_embeddings = lead_embedding_table.unsqueeze(0)
            else:
                assert lead_idx.shape == (b, num_leads), f"lead_idx must be ({b}, {num_leads}), got {tuple(lead_idx.shape)}"
                lead_embeddings = lead_embedding_table[lead_idx] # (b, num_leads, width)
            lead_embeddings = lead_embeddings.unsqueeze(2).expand(b, -1, n + 2, -1)
            # guards against silent broadcasting, e.g. (b, 1, n+2, w) + (b, 12, n+2, w)
            assert lead_embeddings.shape == x.shape, f"lead embeddings {tuple(lead_embeddings.shape)} vs tokens {tuple(x.shape)}"
            x = x + lead_embeddings
            x = rearrange(x, 'b c n p -> b (c n) p')
            assert x.shape[1] == num_leads * (n + 2), f"expected {num_leads * (n + 2)} tokens, got {x.shape[1]}"

            x = self.model.encoder.dropout(x)
            for i in range(self.model.encoder.depth):
                x = getattr(self.model.encoder, f'block{i}')(x)

            # remove SEP embeddings
            x = rearrange(x, 'b (c n) p -> b c n p', c=num_leads)
            x = x[:, :, 1:-1, :]
            sequence_features = rearrange(x, 'b c n p -> b (c n) p')

            pooled_features = torch.mean(x, dim=(1, 2))
            pooled_features = self.model.encoder.norm(pooled_features)

            return sequence_features, pooled_features
        
        self.model.encoder.forward_encoding = custom_forward_encoding
    
    def _encode_and_classify(self, x, lead_idx=None):
        sequence_features, pooled_features = self.model.encoder.forward_encoding(x, lead_idx)

        if self.eval_mode in ["frozen", "finetuning_nonlinear"]:
            output_dict = self.head(seq=sequence_features)
            return output_dict["seq"]
        return self.head(pooled_features)

    def _forward_native(self, x):
        """Encodes only the channels that carry signal. The test-time lead masking zeroes the
        dropped leads, so a channel is present iff it is not all-zero (the masking only keeps
        non-flat leads, so a kept lead is never mistaken for a dropped one). Samples are grouped
        by their number of present leads, so every group is a dense (b_c, c, T) tensor.
        A sample with no signal at all falls back to the 12-channel input, like every other model."""
        assert x.shape[1] == len(self.model.encoder.lead_embeddings) == 12, \
            f"native lead mode expects 12 input channels in standard order, got {x.shape[1]}"
        present = x.abs().sum(dim=-1) > 0 # (B, 12)
        n_present = present.sum(dim=1) # (B,)

        if os.environ.get("ST_MEM_NATIVE_DEBUG"):
            # per batch: how many samples have c present leads, and how often each lead is present
            counts = {c: int((n_present == c).sum()) for c in torch.unique(n_present).tolist()}
            print(f"[st_mem_native] leads/sample: {counts} | per-lead presence: {present.sum(dim=0).tolist()}", flush=True)

        out = None
        for c in torch.unique(n_present).tolist():
            rows = (n_present == c).nonzero(as_tuple=True)[0]
            if c == 0:
                y = self._encode_and_classify(x[rows])
            else:
                # nonzero() lists columns in ascending order per row, so this reshape keeps lead order
                lead_idx = present[rows].nonzero(as_tuple=True)[1].view(len(rows), c) # (b_c, c)
                x_c = torch.gather(x[rows], 1, lead_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])) # (b_c, c, T)
                if os.environ.get("ST_MEM_NATIVE_FORCE_EMB"):
                    # diagnostics only: tag every present lead with this lead embedding instead of its own
                    lead_idx = torch.full_like(lead_idx, int(os.environ["ST_MEM_NATIVE_FORCE_EMB"]))
                y = self._encode_and_classify(x_c, lead_idx)
            if out is None:
                out = y.new_zeros((x.shape[0],) + tuple(y.shape[1:]))
            out[rows] = y
        return out

    def forward(self, x, **kwargs):
        x = torch.nan_to_num(x)
        if self.native_leads:
            x = self._forward_native(x)
        else:
            x = self._encode_and_classify(x)
        return torch.nan_to_num(x)
    

class MerlWrapper(FMWrapperBase):
    """
        Paper: https://arxiv.org/abs/2403.06659
        Code: https://github.com/cheliu-computation/MERL-ICML2024
        Checkpoints: https://drive.google.com/drive/folders/13wb4DppUciMn-Y_qC2JRWTbZdz3xX0w2
        Model sampling frequency: 500 Hz
        Pretraining dataset: mimic_iv_ecg
        Note: checkpoints available for both resnet and vit
    """

    def __init__(self, num_classes, num_output_tokens, backbone="resnet", pretrained_path=None, eval_mode="finetuning_linear", lr=1e-3, discriminative_lr_factor=0.1):
        super().__init__(num_classes, num_output_tokens)

        assert eval_mode in ["finetuning_linear", "finetuning_nonlinear", "frozen", "linear"]
        assert backbone in ["resnet", "vit"]
        self.eval_mode = eval_mode
        self.lr = lr
        self.discriminative_lr_factor = discriminative_lr_factor
        self.backbone = backbone

        # Initialize model
        if self.backbone == "resnet":
            self.model = ResNet18(num_classes=num_classes)
            self.feature_dim = 512
        else:
            self.model = vit_tiny(num_leads=12, num_classes=num_classes, seq_len=5000, patch_size=50)
            self.feature_dim = self.model.width
        
        if pretrained_path:
            state_dict = torch.load(pretrained_path, map_location="cpu")
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("linear.")}        
            self.model.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError("MERLWrapper requires a valid `pretrained_path` to load the encoder.")
        
        # Nonlinear head configurations
        nonlinear_head_config = LearnableQueryAttentionPoolingHeadConfig(
            multi_prediction=False,
            heads=16,
            bias=False
        )
        
        @dataclass
        class InputShape:
            channels: int
            length: int
            static_dim: int

        input_shape = InputShape(channels=self.feature_dim, length=0, static_dim=0)
        
        self.nonlinear_head = LearnableQueryAttentionPoolingHead(
            hparams_head=nonlinear_head_config,
            hparams_input_shape=input_shape,
            target_dim=num_classes
        )

        if self.eval_mode == "finetuning_linear":
            self.head = nn.Linear(self.feature_dim, num_classes)
        elif self.eval_mode == "finetuning_nonlinear":
            self.head = self.nonlinear_head
        elif self.eval_mode == "frozen":
            self.head = self.nonlinear_head
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()     
        else:
            self.head = nn.Linear(self.feature_dim, num_classes)
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        
        self._override_forward()
    
    def get_params(self):
        """The following code is only applicable for MERL ResNet architecture"""
        head_params = list(self.head.parameters())

        if self.eval_mode in ["frozen", "linear"]:
            return [{"params": head_params, "lr": self.lr}]
        
        encoder_params = []
        predictor_params = []
        
        for name, param in self.model.named_parameters():
            if name.startswith(("conv1", "bn1", "layer1", "layer2")):
                encoder_params.append(param)
            elif name.startswith(("layer3", "layer4")):
                predictor_params.append(param)
        
        print("Returning layer dependent learning rate from MERLWrapper (Resnet)...")
        
        return [
            {"params": head_params, "lr": self.lr},
            {"params": predictor_params, "lr": self.lr * self.discriminative_lr_factor},
            {"params": encoder_params, "lr": self.lr * self.discriminative_lr_factor * self.discriminative_lr_factor}
        ]
    
    def _override_forward(self):
        if self.backbone == "resnet":
            def custom_forward(x):
                out = torch.relu(self.model.bn1(self.model.conv1(x)))
                out = self.model.layer1(out)
                out = self.model.layer2(out)
                out = self.model.layer3(out)
                out = self.model.layer4(out)

                sequence_features = out
                sequence_features = rearrange(sequence_features, "b c l -> b l c")

                pooled_features = self.model.avgpool(out)
                pooled_features = pooled_features.view(pooled_features.size(0), -1)
                
                return sequence_features, pooled_features
            
            self.model.forward = custom_forward
        else:
            def custom_forward_encoding(x):
                # for conv patch
                x = self.model.to_patch_embedding(x)
                x = rearrange(x, 'b c n -> b n c')
                x = x + self.model.pos_embedding

                # transformer blocks
                sequence_features = self.model.dropout(x)
                for i in range(self.model.depth):
                    sequence_features = getattr(self.model, f"block{i}")(sequence_features)

                pooled_features = torch.mean(sequence_features, dim=1)  # global average pooling
                pooled_features = self.model.norm(pooled_features)

                return sequence_features, pooled_features
            
            self.model.forward_encoding = custom_forward_encoding

    def forward(self, x, **kwargs):
        x = torch.nan_to_num(x)

        if self.backbone == "resnet":
            sequence_features, pooled_features = self.model(x)
        else:
            sequence_features, pooled_features = self.model.forward_encoding(x)

        if self.eval_mode in ["frozen", "finetuning_nonlinear"]:
            output_dict = self.head(seq=sequence_features)
            x = output_dict["seq"]
        else:
            x = self.head(pooled_features)

        return torch.nan_to_num(x)


class EcgFmKEDWrapper(FMWrapperBase):
    """
        Paper: https://doi.org/10.1016/j.xcrm.2024.101875
        Code: https://github.com/control-spiderman/ECGFM-KED
        Checkpoints: https://zenodo.org/records/14881564
        Model sampling frequency: 100 Hz
        Pretraining dataset: mimic_iv_ecg
    """

    def __init__(self, num_classes, num_output_tokens, pretrained_path=None, eval_mode="finetuning_linear", lr=1e-3, discriminative_lr_factor=0.1):
        super().__init__(num_classes, num_output_tokens)

        assert eval_mode in ["finetuning_linear", "finetuning_nonlinear", "frozen", "linear"]
        self.eval_mode = eval_mode
        self.lr = lr
        self.discriminative_lr_factor = discriminative_lr_factor

        self.model = xresnet1d101(
            num_classes=768,
            input_channels=12,
            kernel_size=5,
            ps_head=0.5
        )

        self.feature_dim = 768

        if pretrained_path:
            checkpoint = torch.load(pretrained_path, map_location="cpu")        
            state_dict = {}
            for k, v in checkpoint.items():
                if k.startswith("ecg_model."):
                    state_dict[k.replace("ecg_model.", "")] = v
                elif k.startswith("model."):
                    continue
                else:
                    state_dict[k] = v                    
            self.model.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError("EcgFmKEDWrapper requires a valid `pretrained_path` to load the encoder.")
        
        # Nonlinear head configurations
        nonlinear_head_config = LearnableQueryAttentionPoolingHeadConfig(
            multi_prediction=False,
            heads=16,
            bias=False
        )
        
        @dataclass
        class InputShape:
            channels: int
            length: int
            static_dim: int

        input_shape = InputShape(channels=self.feature_dim, length=0, static_dim=0)
        
        self.nonlinear_head = LearnableQueryAttentionPoolingHead(
            hparams_head=nonlinear_head_config,
            hparams_input_shape=input_shape,
            target_dim=num_classes
        )
        
        if self.eval_mode == "finetuning_linear":
            self.head = nn.Linear(self.feature_dim, num_classes)
        elif self.eval_mode == "finetuning_nonlinear":
            self.head = self.nonlinear_head
        elif self.eval_mode == "frozen":
            self.head = self.nonlinear_head
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()      
        else:
            self.head = nn.Linear(self.feature_dim, num_classes)
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        
        self._override_forward()

    def get_params(self):
        head_params = list(self.head.parameters())

        if self.eval_mode in ["frozen", "linear"]:
            return [{"params": head_params, "lr": self.lr}]
        
        encoder_params = []
        predictor_params = []
        
        for name, param in self.model.named_parameters():
            if name.startswith(("0.", "1.", "2.")):
                encoder_params.append(param)
            elif name.startswith(("4.", "5.", "6.", "7.")):
                predictor_params.append(param)
        
        print("Returning layer dependent learning rate from EcgFmKEDWrapper...")
        
        return [
            {"params": head_params, "lr": self.lr},
            {"params": predictor_params, "lr": self.lr * self.discriminative_lr_factor},
            {"params": encoder_params, "lr": self.lr * self.discriminative_lr_factor * self.discriminative_lr_factor}
        ]
    
    def _override_forward(self):
        original_forward = self.model.forward        
        def custom_forward(x):
            if x.dim() == 2:
                x = x.unsqueeze(1)
            elif x.dim() == 3 and x.shape[1] != 12:
                x = x.transpose(1, 2)
            
            x = torch.nan_to_num(x)
            sequence_features = original_forward(x)
            sequence_features = rearrange(sequence_features, "b c l -> b l c")  # (batch, seq_len, channels)
            
            pooled_features = torch.mean(sequence_features, dim=1)  # GAP
            
            return sequence_features, pooled_features
        
        self.model.forward = custom_forward
    
    def forward(self, x, **kwargs):
        sequence_features, pooled_features = self.model(x)

        if self.eval_mode in ["frozen", "finetuning_nonlinear"]:
            output_dict = self.head(seq=sequence_features)
            x = output_dict["seq"]
        else:
            x = self.head(pooled_features)

        return torch.nan_to_num(x)


class CPCWrapper(FMWrapperBase):
    def __init__(self, num_classes, num_output_tokens, config_path=None, eval_mode="finetuning_linear", lr=1e-3, discriminative_lr_factor=0.1):
        super().__init__(num_classes, num_output_tokens)
        assert eval_mode in ["finetuning_linear", "finetuning_nonlinear", "frozen", "linear"]

        self.eval_mode = eval_mode
        self.lr = lr
        self.discriminative_lr_factor = discriminative_lr_factor
        
        self.model, self.config = load_model_from_config(
            config_name=config_path         
        )
        
        self.feature_dim = 512
        
        if self.eval_mode == "finetuning_linear":
            self.head = nn.Linear(self.feature_dim, num_classes)
        elif self.eval_mode in ["finetuning_nonlinear", "frozen"]:
            nonlinear_head_config = LearnableQueryAttentionPoolingHeadConfig(
                multi_prediction=False,
                heads=16,
                bias=False
            )
            
            @dataclass
            class InputShape:
                channels: int
                length: int
                static_dim: int

            input_shape = InputShape(channels=self.feature_dim, length=0, static_dim=0)
            
            self.head = LearnableQueryAttentionPoolingHead(
                hparams_head=nonlinear_head_config,
                hparams_input_shape=input_shape,
                target_dim=num_classes
            )
            
            if self.eval_mode == "frozen":
                for p in self.model.parameters():
                    p.requires_grad = False
                self.model.eval()      
        else:
            self.head = nn.Linear(self.feature_dim, num_classes)
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
    
    def get_params(self):
        head_params = list(self.head.parameters())

        if self.eval_mode in ["frozen", "linear"]:
            return [{"params": head_params, "lr": self.lr}]
        
        encoder_module, predictor_module, _ = self.model.ts_encoder.get_modules()        
    
        encoder_params = list(chain(*[e.parameters() for e in encoder_module]))
        predictor_params = list(chain(*[p.parameters() for p in predictor_module]))

        print("I am here.")
        
        return [
            {"params": head_params, "lr": self.lr},
            {"params": predictor_params, "lr": self.lr * self.discriminative_lr_factor},
            {"params": encoder_params, "lr": self.lr * self.discriminative_lr_factor * self.discriminative_lr_factor}
        ]
    
    def forward(self, x, **kwargs):
        x = torch.nan_to_num(x)
        output = self.model(seq=x)        
        sequence_features = output["seq"]
        
        if self.eval_mode in ["frozen", "finetuning_nonlinear"]:
            output_dict = self.head(seq=sequence_features)
            x = output_dict["seq"]  
        else:
            pooled_features = sequence_features.mean(dim=1)
            x = self.head(pooled_features)

        return torch.nan_to_num(x)


class HubertEcgWrapper(FMWrapperBase):
    def __init__(self, num_classes, num_output_tokens, pretrained_path=None, eval_mode="finetuning_linear", lr=1e-3, discriminative_lr_factor=0.1, variant="base"):
        super().__init__(num_classes, num_output_tokens)

        assert eval_mode in ["finetuning_linear", "finetuning_nonlinear", "frozen", "linear"]
        if variant not in HUBERT_CONFIGS:
            raise ValueError(f"Unknown HuBERT-ECG variant '{variant}'. Expected one of {list(HUBERT_CONFIGS)}.")
        self.eval_mode = eval_mode
        self.lr = lr
        self.discriminative_lr_factor = discriminative_lr_factor
        self.variant = variant

        cfg = HUBERT_CONFIGS[variant]
        pretrained_hubert = HuBERTECG(cfg)
        self.model = HuBERTForECGClassification(
            hubert_ecg=pretrained_hubert,
            num_labels=num_classes,
            classifier_hidden_size=None,
            use_label_embedding=False
        )

        self.feature_dim = cfg.hidden_size

        if pretrained_path:
            with safe_open(pretrained_path, framework="pt") as f:
                state_dict = {k: f.get_tensor(k) for k in f.keys()}
            self.model.hubert_ecg.load_state_dict(state_dict, strict=False)            
        else:
            raise ValueError("HubertEcgWrapper requires a valid `pretrained_path` to load the encoder.")

        # Nonlinear head configurations    
        nonlinear_head_config = LearnableQueryAttentionPoolingHeadConfig(
            multi_prediction=False,
            heads=16,
            bias=False
        )
        
        @dataclass
        class InputShape:
            channels: int
            length: int
            static_dim: int

        input_shape = InputShape(channels=self.feature_dim, length=0, static_dim=0)
        
        self.nonlinear_head = LearnableQueryAttentionPoolingHead(
            hparams_head=nonlinear_head_config,
            hparams_input_shape=input_shape,
            target_dim=num_classes
        )
    
        if self.eval_mode == "finetuning_linear":
            self.head = nn.Linear(self.feature_dim, num_classes)
        elif self.eval_mode == "finetuning_nonlinear":
            self.head = self.nonlinear_head
        elif self.eval_mode == "frozen":
            self.head = self.nonlinear_head
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()      
        else:
            self.head = nn.Linear(self.feature_dim, num_classes)
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        
        self._override_forward()
    
    def get_params(self):
        head_params = list(self.head.parameters())
        
        if self.eval_mode in ["frozen", "linear"]:
            return [{"params": head_params, "lr": self.lr}]
        
        feature_extractor_params = []
        transformer_early_params = []
        transformer_late_params = []
        feature_projection_params = []
        
        for name, param in self.model.hubert_ecg.named_parameters():
            if "feature_extractor" in name:
                feature_extractor_params.append(param)
            elif "feature_projection" in name:
                feature_projection_params.append(param)
            elif "encoder.layers" in name:
                layer_num = int(name.split(".layers.")[1].split(".")[0])
                total_layers = self.model.hubert_ecg.config.num_hidden_layers
                
                if layer_num < total_layers // 2:
                    transformer_early_params.append(param)
                else:
                    transformer_late_params.append(param)
            elif "encoder" in name:
                transformer_early_params.append(param)
        
        print("Returning layer dependent learning rate from HubertEcgWrapper...")
        
        return [
            {"params": head_params, "lr": self.lr},
            {"params": transformer_late_params, "lr": self.lr * self.discriminative_lr_factor},
            {"params": transformer_early_params, "lr": self.lr * self.discriminative_lr_factor * self.discriminative_lr_factor},
            {"params": feature_projection_params, "lr": self.lr * self.discriminative_lr_factor * self.discriminative_lr_factor},
            {"params": feature_extractor_params, "lr": self.lr * self.discriminative_lr_factor * self.discriminative_lr_factor}
        ]

    def _override_forward(self):        
        def custom_forward(x, attention_mask=None, output_attentions=None, output_hidden_states=None, return_dict=None):
            return_dict = return_dict if return_dict is not None else self.model.hubert_ecg.config.use_return_dict
            output_hidden_states = True if self.model.hubert_ecg.config.use_weighted_layer_sum else output_hidden_states
            
            encodings = self.model.hubert_ecg(
                x,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
            
            sequence_features = encodings.last_hidden_state
            pooled_features = sequence_features.mean(dim=1)
            
            return sequence_features, pooled_features
        
        self.model.forward = custom_forward

    def forward(self, x, **kwargs):
        x = torch.nan_to_num(x)
        x_np = x.detach().cpu().numpy()
        preprocessed = [ecg_preprocessing(sig) for sig in x_np]
        x = torch.from_numpy(np.stack(preprocessed, axis=0)).to(x.device).float()
        x = x.reshape(x.shape[0], -1)

        sequence_features, pooled_features = self.model(x)
        
        if self.eval_mode in ["finetuning_nonlinear", "frozen"]:            
            output_dict = self.nonlinear_head(seq=sequence_features)
            x = output_dict["seq"]
            
        if self.eval_mode in ["finetuning_linear", "linear"]:
            x = self.head(pooled_features)
            
        return torch.nan_to_num(x)


# from clinical_ts.models.ecg_foundation_models.ecg_fm_config import ECG_FM_CONFIG
# from fairseq_signals.models.ecg_transformer import ECGTransformerFinetuningModel

# class ECG_FM_Wrapper(FMWrapperBase):
#     def __init__(self, num_classes, num_output_tokens, pretrained_path=None, eval_mode="finetuning_linear", lr=1e-3, discriminative_lr_factor=0.1):
#         super().__init__(num_classes, num_output_tokens)

#         assert eval_mode in ["finetuning_linear", "finetuning_nonlinear", "frozen", "linear"]
#         self.eval_mode = eval_mode
#         self.lr = lr
#         self.discriminative_lr_factor = discriminative_lr_factor

#         if pretrained_path:
#             cfg = replace(ECG_FM_CONFIG, model_path=pretrained_path)
#             print(f"Model path: {cfg.model_path}")

#             self.model = ECGTransformerFinetuningModel.build_model(cfg, task=None)
#             self.feature_dim = 768
#         else:
#             raise ValueError("ECG_FM_Wrapper requires a valid `pretrained_path` to load the encoder.")

#         # Nonlinear head configuration
#         nonlinear_head_config = LearnableQueryAttentionPoolingHeadConfig(
#             multi_prediction=False,
#             heads=16,
#             bias=False
#         )

#         @dataclass
#         class InputShape:
#             channels: int
#             length: int
#             static_dim: int

#         input_shape = InputShape(channels=self.feature_dim, length=0, static_dim=0)

#         self.nonlinear_head = LearnableQueryAttentionPoolingHead(
#             hparams_head=nonlinear_head_config,
#             hparams_input_shape=input_shape,
#             target_dim=num_classes
#         )

#         # Set evaluation mode
#         if self.eval_mode == "finetuning_linear":
#             self.head = nn.Linear(self.feature_dim, num_classes)
#         elif self.eval_mode == "finetuning_nonlinear":
#             self.head = self.nonlinear_head
#         elif self.eval_mode == "frozen":
#             self.head = self.nonlinear_head
#             for p in self.model.parameters():
#                 p.requires_grad = False
#             self.model.eval()
#         else:
#             self.head = nn.Linear(self.feature_dim, num_classes)
#             for p in self.model.parameters():
#                 p.requires_grad = False
#             self.model.eval()
    
#     def get_params(self):
#         head_params = list(self.head.parameters())

#         if self.eval_mode in ["frozen", "linear"]:
#             return [{"params": head_params, "lr": self.lr}]
        
#         encoder_params = []
#         predictor_params = []
#         feature_extractor_params = []
        
#         for name, param in self.model.named_parameters():
#             if "feature_extractor" in name:
#                 feature_extractor_params.append(param)
#             elif "encoder.layers" in name:
#                 layer_num = int(name.split(".layers.")[1].split(".")[0])
#                 if layer_num < 6:
#                     encoder_params.append(param)
#                 else:
#                     predictor_params.append(param)
#             elif "encoder.layer_norm" in name or "post_extract_proj" in name or "conv_pos" in name:
#                 encoder_params.append(param)
#             elif "layer_norm" in name or "quantizer" in name or "project_q" in name or "final_proj" in name:
#                 predictor_params.append(param)
        
#         print("Returning layer dependent learning rate from ECG_FM_Wrapper...")
        
#         return [
#             {"params": head_params, "lr": self.lr},
#             {"params": predictor_params, "lr": self.lr * self.discriminative_lr_factor},
#             {"params": encoder_params, "lr": self.lr * self.discriminative_lr_factor * self.discriminative_lr_factor},
#             {"params": feature_extractor_params, "lr": self.lr * self.discriminative_lr_factor * self.discriminative_lr_factor}
#         ]

#     def forward(self, x, **kwargs):
#         x = torch.nan_to_num(x)
#         output_dict = self.model.extract_features(x, padding_mask=None)
#         sequence_features = output_dict["x"]

#         if self.eval_mode in ["frozen", "finetuning_nonlinear"]:
#             output_dict = self.head(seq=sequence_features)
#             x = output_dict["seq"]
#         else:
#             pooled_features = sequence_features.mean(dim=1)
#             x = self.head(pooled_features)

#         return torch.nan_to_num(x)


# ==============================================================================
# RelayWrapper — RELAY GAT model for the ECG-FM benchmarking pipeline
#
# DESIGN NOTES
# ------------
#   ECG-FM calls forward(x: Tensor[B, L, T]) and expects logits (B, num_classes).
#   RELAY's GraphClassificationBase.forward() instead takes a PyG mega-graph
#   and computes the loss internally.
#
#   This wrapper bridges the gap by replicating the inference steps of
#   GraphClassificationBase.forward() and returning logits only (ECG-FM handles
#   loss externally).
#
#   RELAY's core strength is processing genuine k-lead graphs — it was pretrained
#   on randomly sampled subsets of leads and must never see zero-padded graphs at
#   test time.  The ECG-FM DataLoader signals inactive leads by zeroing them out.
#   This wrapper detects those zeros, runs the HuBERT CNN only on active lead
#   signals (one vectorised call for the whole batch), then assembles per-sample
#   graphs containing exactly the k_i active leads for sample i.
#
#   Edge-index templates for every possible k (1..num_leads) are precomputed at
#   construction time and registered as buffers (device-safe).
#
#   _drop_edge compatibility: RELAY's _drop_edge classifies intra/inter-lead
#   edges via `row // S == col // S` (a stride formula, not node_lead_ids).  This
#   works correctly for variable-k graphs as long as nodes are laid out
#   consecutively per lead in blocks of S — which this wrapper guarantees.
# ==============================================================================

import os
import sys
import json

class RelayWrapper(FMWrapperBase):

    def __init__(
        self,
        num_classes: int,
        num_output_tokens: int,
        config_path: str,
        relay_ecg_code_path: str = None,
        pretrained_path: str = None,
        eval_mode: str = "finetuning_linear",
        lr: float = 1e-3,
        discriminative_lr_factor: float = 0.1,
        num_leads: int = 12,        # maximum leads in input (= --input-channels)
        nodes_per_lead: int = 20,   # override if CLI value differs from config JSON
        edge_strategy: str = "spatiotemporal",
    ):
        super().__init__(num_classes, num_output_tokens)

        # --- make RELAY importable (default: <repo_root>/relay_code) ----------
        if relay_ecg_code_path is None:
            relay_ecg_code_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "relay_code"))
        if relay_ecg_code_path not in sys.path:
            sys.path.insert(0, relay_ecg_code_path)
        from modelling_graph_ecg import ECGGATForGraphClassification  # noqa: E402
        from config import ECGGATConfig                               # noqa: E402

        self.eval_mode = eval_mode
        self.lr = lr
        self.discriminative_lr_factor = discriminative_lr_factor
        self.num_leads = num_leads
        self.edge_strategy = edge_strategy

        # --- build config --------------------------------------------------------
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config_dict["num_classes"] = num_classes
        config_dict["mask_node_prob"] = None          # disable pretraining-only fields
        config_dict["pseudo_vocab_sizes"] = None
        config_dict["pseudo_classifier_projection_size"] = None
        # --gat-nodes-per-lead overrides the JSON value only when explicitly set
        # to something different; otherwise the JSON config takes precedence.
        if nodes_per_lead != config_dict.get("nodes_per_lead", nodes_per_lead):
            config_dict["nodes_per_lead"] = nodes_per_lead
        config = ECGGATConfig(**config_dict)
        # S must always come from the (possibly overridden) config — _drop_edge
        # uses config.nodes_per_lead internally, so the two must stay in sync.
        self.S = config.nodes_per_lead

        self.gat = ECGGATForGraphClassification(config)

        # --- load pretrained weights --------------------------------------------
        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            # Drop pretraining-only keys and the old classification head.
            skip_prefixes = (
                "mask_embed",
                "classifiers.",
                "pseudo_classifier_projections.",
                "pseudo_label_embeddings.",
                "output_layer",
            )
            state_dict = {
                k: v for k, v in state_dict.items()
                if not any(k.startswith(p) for p in skip_prefixes)
            }
            result = self.gat.load_state_dict(state_dict, strict=False)
            print(f"[RelayWrapper] missing_keys   : {result.missing_keys}")
            print(f"[RelayWrapper] unexpected_keys: {result.unexpected_keys}")

        # --- freeze encoder for frozen / linear eval modes ----------------------
        if eval_mode in ("frozen", "linear"):
            self.gat.embedder.requires_grad_(False)
            self.gat.encoder.requires_grad_(False)
            self.gat.norm_post_encoder.requires_grad_(False)
            if hasattr(self.gat, "lead_encoding"):
                self.gat.lead_encoding.requires_grad_(False)
            if hasattr(self.gat, "temporal_encoding"):
                self.gat.temporal_encoding.requires_grad_(False)

        # --- pre-build edge-index templates for every k in [1, num_leads] ------
        # One template per possible active-lead count; registered as buffers so
        # they are moved to the correct device automatically.
        # Template for k leads: local node indices 0 .. k*S-1, laid out as
        #   [lead-0 nodes] [lead-1 nodes] ... [lead-(k-1) nodes]
        # This layout is what _drop_edge's `row // S` formula expects.
        for k in range(1, num_leads + 1):
            tmpl = RelayWrapper._build_edge_index_template(k, self.S, edge_strategy)
            self.register_buffer(f"_edge_tmpl_{k}", tmpl)

    # --------------------------------------------------------------------------
    # Edge-index builder (mirrors ECGDataset.precompile_edge_index)
    # --------------------------------------------------------------------------

    @staticmethod
    def _build_edge_index_template(
        num_leads: int,
        nodes_per_lead: int,
        strategy: str,
    ) -> torch.Tensor:
        """
        Undirected edge index for a *single* graph with `num_leads` leads and
        `nodes_per_lead` temporal nodes per lead.

        Node layout (required by _drop_edge's stride formula):
            nodes  0 ..   S-1  → local lead 0
            nodes  S .. 2S-1  → local lead 1
            ...
        Physical lead identities (0-11) are tracked separately in node_lead_ids
        and used only for lead_encoding — not for edge construction.
        """
        L, S = num_leads, nodes_per_lead
        row, col = [], []

        if strategy == "spatiotemporal":
            # Intra-lead: fully-connected within each lead subgraph
            for lead_idx in range(L):
                start = lead_idx * S
                nodes = torch.arange(start, start + S)
                i, j = torch.combinations(nodes, r=2).unbind(1)
                row.extend(i.tolist() + j.tolist())   # both directions
                col.extend(j.tolist() + i.tolist())
            # Inter-lead: homologous temporal position across all lead subgraphs
            for t in range(S):
                nodes = torch.arange(L) * S + t
                i, j = torch.combinations(nodes, r=2).unbind(1)
                row.extend(i.tolist() + j.tolist())
                col.extend(j.tolist() + i.tolist())

        elif strategy == "fully_connected":
            all_nodes = torch.arange(L * S)
            i, j = torch.combinations(all_nodes, r=2).unbind(1)
            row.extend(i.tolist() + j.tolist())
            col.extend(j.tolist() + i.tolist())

        else:
            raise ValueError(
                f"edge_strategy must be 'spatiotemporal' or 'fully_connected', "
                f"got '{strategy}'"
            )

        return torch.tensor([row, col], dtype=torch.long)  # (2, E)

    # --------------------------------------------------------------------------
    # Forward — genuine variable-k graph processing (no zero-lead padding)
    # --------------------------------------------------------------------------

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, L, T)
            ECG-FM dense tensor.  Inactive leads are zeroed by the DataLoader
            for random-lead-sampling runs.  This wrapper detects which leads
            carry a non-zero signal and includes only those in the graph, so
            RELAY always processes a genuine k-lead graph — never a padded one.

        Returns
        -------
        logits : (B, num_classes)
        """
        B, L, T = x.shape
        assert L == self.num_leads, (
            f"RelayWrapper: expected {self.num_leads} input leads, got {L}. "
            "Check --input-channels."
        )

        m = self.gat  # ECGGATForGraphClassification; shorthand below

        # ------------------------------------------------------------------
        # 1. Detect active leads per sample from the RAW input.
        #    Must be done before preprocessing because ecg_preprocessing maps
        #    zero-padded leads to all-(-1), which would confuse the detection.
        # ------------------------------------------------------------------
        active_mask = x.abs().sum(dim=-1) > 0               # (B, L) bool
        # nonzero() returns sorted (row, col) pairs — i.e. sample then lead.
        sample_idx, lead_idx = active_mask.nonzero(as_tuple=True)  # (N_active,) each
        assert sample_idx.numel() > 0, \
            "RelayWrapper: every lead in this batch is zero."

        # ------------------------------------------------------------------
        # 1b. Apply HuBERT ECG preprocessing (bandpass filter + min-max
        #     scaling to [-1, 1]).  This matches the preprocessing used during
        #     HuBERT ECG pre-training, on which RELAY's CNN embedder is built.
        #     Applied to the full (B, L, T) tensor; inactive-lead signals
        #     remain near-zero because filtering and scaling are linear for
        #     constant inputs (the active_mask computed above is unaffected).
        # ------------------------------------------------------------------
        x_np = x.detach().cpu().numpy()                     # (B, L, T)
        preprocessed = [
            ecg_preprocessing(x_np[i]) for i in range(B)   # each (L, T)
        ]
        x = torch.from_numpy(
            np.stack(preprocessed, axis=0)
        ).to(x.device).float()                              # (B, L, T)

        # ------------------------------------------------------------------
        # 2. HuBERT CNN — one vectorised call on active lead signals only.
        #    active_signals : (N_active, T)
        #    cnn_out        : (N_active, C, S)
        # ------------------------------------------------------------------
        active_signals = x[sample_idx, lead_idx, :]         # (N_active, T)
        cnn_out = m.embedder(active_signals)                 # (N_active, C, S)
        S_actual = cnn_out.size(2)
        assert S_actual == self.S, (
            f"RelayWrapper: CNN output length {S_actual} != config.nodes_per_lead "
            f"{self.S}.  Verify --input-size * --fs-model gives the expected T."
        )

        # Split CNN output by sample for per-sample graph construction.
        active_counts = active_mask.sum(dim=1)               # (B,) = k_i per sample
        cnn_per_sample  = cnn_out.split(active_counts.tolist(), dim=0)
        lead_ids_per_sample = lead_idx.split(active_counts.tolist(), dim=0)

        # ------------------------------------------------------------------
        # 3. Build per-sample node features and graph topology.
        #    For each sample i:
        #      - nodes are laid out consecutively per active lead (blocks of S)
        #      - the k_i-lead edge template is looked up by k_i and shifted
        #        into the mega-graph by node_offset
        #      - physical lead IDs are passed to lead_encoding (not edge index)
        # ------------------------------------------------------------------
        all_node_feats, all_edge_idx, all_batch_vecs = [], [], []
        node_offset = 0

        for i in range(B):
            k_i = int(active_counts[i].item())
            assert k_i > 0, f"RelayWrapper: sample {i} has no active leads."

            # Node features: (k_i, C, S) → (k_i*S, C)
            nf = cnn_per_sample[i].permute(0, 2, 1).reshape(k_i * self.S, -1)

            # Physical lead IDs for positional encodings (original 0-11 indices).
            phys_ids      = lead_ids_per_sample[i]                       # (k_i,)
            node_phys_ids = phys_ids.repeat_interleave(self.S)           # (k_i*S,)
            node_temp_pos = torch.arange(self.S, device=x.device).repeat(k_i)  # (k_i*S,)

            if m.config.use_temporal_encoding:
                nf = m.temporal_encoding(nf, node_temp_pos)
            if m.config.use_lead_encoding:
                nf = m.lead_encoding(nf, node_phys_ids)

            # Edge index for a k_i-lead graph, shifted into the mega-batch.
            # Local layout — blocks of S per active lead — is what _drop_edge
            # expects: `row // S == col // S` identifies intra-lead pairs.
            tmpl = getattr(self, f"_edge_tmpl_{k_i}")        # (2, E_k)
            all_edge_idx.append(tmpl + node_offset)

            all_node_feats.append(nf)
            all_batch_vecs.append(
                torch.full((k_i * self.S,), i, device=x.device, dtype=torch.long)
            )
            node_offset += k_i * self.S

        node_feats = torch.cat(all_node_feats, dim=0)        # (N_total, C)
        edge_index = torch.cat(all_edge_idx,   dim=1)        # (2, E_total)
        batch_vec  = torch.cat(all_batch_vecs, dim=0)        # (N_total,)

        # ------------------------------------------------------------------
        # 4. Edge dropout (no-op at eval time or when edge_drop_prob == 0).
        # ------------------------------------------------------------------
        edge_index, _, _ = m._drop_edge(edge_index, None, None)

        # ------------------------------------------------------------------
        # 5. GNN encoder.
        # ------------------------------------------------------------------
        enc_out    = m.encoder(
            x=node_feats,
            edge_index=edge_index,
            batch=batch_vec,
            batch_size=B,
        )
        node_feats = enc_out[0] if isinstance(enc_out, tuple) else enc_out

        # ------------------------------------------------------------------
        # 6. Post-encoder normalisation (mirrors GraphClassificationBase).
        # ------------------------------------------------------------------
        if m.encoder.encoder.supports_norm_edge_index:
            node_feats = m.norm_post_encoder(node_feats, edge_index)
        elif m.encoder.encoder.supports_norm_batch:
            node_feats = m.norm_post_encoder(node_feats, batch_vec, B)
        else:
            node_feats = m.norm_post_encoder(node_feats)

        # ------------------------------------------------------------------
        # 7. Graph-level pooling + classifier head → logits (B, num_classes).
        # ------------------------------------------------------------------
        logits = m.get_logits_for_graph_classification(node_feats, batch=batch_vec)
        return logits

    # --------------------------------------------------------------------------
    # Discriminative learning-rate parameter groups
    # --------------------------------------------------------------------------

    def get_params(self):
        """
        Returns optimizer param groups for configure_optimizers() in Main_Lite_ECG.

        frozen / linear      → only the classification head is trained.
        finetuning_linear /
        finetuning_nonlinear → head at full lr; encoder (embedder + GNN +
                               post-norm) at lr * discriminative_lr_factor.
        """
        head_params = list(self.gat.output_layer.parameters())

        if self.eval_mode in ("frozen", "linear"):
            return [{"params": head_params, "lr": self.lr}]

        encoder_params = [
            p for name, p in self.gat.named_parameters()
            if not name.startswith("output_layer")
        ]
        return [
            {"params": head_params,    "lr": self.lr},
            {"params": encoder_params, "lr": self.lr * self.discriminative_lr_factor},
        ]
#
# # END RelayWrapper
# ==============================================================================