from transformer_lens import HookedTransformer
from transformer_lens.pretrained.weight_conversions import convert_qwen2_weights
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformers import AutoModelForCausalLM, AutoConfig
import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.backends.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def get_cfg_dict(based_model_name, hf_config):
    if "qwen2" in based_model_name.lower():
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "n_key_value_heads": hf_config.num_key_value_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": 2048, 
            "eps": hf_config.rms_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "use_attn_scale": True,
            "initializer_range": hf_config.initializer_range,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_base": int(hf_config.rope_theta),
            "rotary_adjacent_pairs": False,
            "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads,
            "tokenizer_prepends_bos": True,
            "final_rms": True,
            "gated_mlp": True,
            "default_prepend_bos": False,
        }
        return cfg_dict
    else:
        raise ValueError(f"Unknown model family for {based_model_name}")
    

def load_finetuned_model(based_model_name, fine_tuned_model_path):
    model_qwen = AutoModelForCausalLM.from_pretrained(
        fine_tuned_model_path,
        torch_dtype=torch.float16,      
        trust_remote_code=True,         
    )

    hf_config = AutoConfig.from_pretrained(
                based_model_name,
            )

    cfg_dict = get_cfg_dict(based_model_name, hf_config)

    cfg = HookedTransformerConfig.from_dict(cfg_dict)

    state_dict = convert_qwen2_weights(model_qwen, cfg)

    model = HookedTransformer.from_pretrained(
        model_name=based_model_name, 
        device=device
    )

    model.load_and_process_state_dict(state_dict)

    return model