Below is the code example to load the model:
```py
from transformer_lens import HookedTransformer
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
import pickle
import torch
def load_finetuned_model_lens_from_dir(dir: str, device: str = device) -> HookedTransformer:
    """
    Load a fine-tuned TransformerLens model from a specified directory.

    Args:
        dir (str): Directory containing the model files.
        device (str): Device to load the model onto.
    
    Returns:
        HookedTransformer: The loaded TransformerLens model.
    """
    with open(os.path.join(dir, 'model_config.pkl'), 'rb') as f:
        new_cfg_dict = pickle.load(f)
    new_cfg = HookedTransformerConfig.from_dict(new_cfg_dict)
    new_model = HookedTransformer(new_cfg)
    new_model.load_state_dict(torch.load(os.path.join(dir, 'model.pt'), map_location=device))
    return new_model

model = load_finetuned_model_lens_from_dir(model_path)
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
model.to(device)
model.eval()
```