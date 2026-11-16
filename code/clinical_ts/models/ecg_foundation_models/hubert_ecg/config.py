from clinical_ts.models.ecg_foundation_models.hubert_ecg.hubert_ecg import HuBERTECGConfig


def _make_config(hidden_size, num_hidden_layers, intermediate_size,
                 num_attention_heads, classifier_proj_size=256, layerdrop=0.1):
    return HuBERTECGConfig(
        activation_dropout=0.1,
        apply_spec_augment=True,
        attention_dropout=0.1,
        bos_token_id=1,
        classifier_proj_size=classifier_proj_size,
        conv_bias=False,
        conv_dim=(512, 512, 512, 512, 512),
        conv_kernel=(10, 3, 3, 2, 2),
        conv_pos_batch_norm=False,
        conv_stride=(4, 2, 2, 2, 2),
        ctc_loss_reduction="sum",
        ctc_zero_infinity=False,
        do_stable_layer_norm=False,
        ensemble_length=1,
        eos_token_id=2,
        feat_extract_activation="gelu",
        feat_extract_norm="group",
        feat_proj_dropout=0.0,
        feat_proj_layer_norm=True,
        final_dropout=0.1,
        hidden_act="gelu",
        hidden_dropout=0.1,
        hidden_size=hidden_size,
        initializer_range=0.02,
        intermediate_size=intermediate_size,
        layer_norm_eps=1e-05,
        layerdrop=layerdrop,
        mask_feature_length=10,
        mask_feature_min_masks=0,
        mask_feature_prob=0.0,
        mask_time_length=1,
        mask_time_min_masks=2,
        mask_time_prob=0.33,
        model_type="hubert_ecg",
        num_attention_heads=num_attention_heads,
        num_conv_pos_embedding_groups=16,
        num_conv_pos_embeddings=128,
        num_feat_extract_layers=5,
        num_hidden_layers=num_hidden_layers,
        pad_token_id=0,
        transformers_version="4.53.2",
        use_weighted_layer_sum=False,
        vocab_size=100,
        vocab_sizes=(500),
    )


# Variant configs matching the official HuBERT-ECG pretraining settings.
hubert_config_small = _make_config(
    hidden_size=512, num_hidden_layers=8,  intermediate_size=2048,
    num_attention_heads=8,  classifier_proj_size=256, layerdrop=0.1,
)
hubert_config_base = _make_config(
    hidden_size=768, num_hidden_layers=12, intermediate_size=3072,
    num_attention_heads=12, classifier_proj_size=256, layerdrop=0.1,
)
hubert_config_large = _make_config(
    hidden_size=960, num_hidden_layers=16, intermediate_size=3840,
    num_attention_heads=12, classifier_proj_size=512, layerdrop=0.0,
)

# Back-compat alias for existing callers.
hubert_config = hubert_config_base

HUBERT_CONFIGS = {
    "small": hubert_config_small,
    "base":  hubert_config_base,
    "large": hubert_config_large,
}