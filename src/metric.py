import torch
from typing_extensions import Tuple, List, Union

def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor):
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)
    return logits[idx, input_length - 1]

def logit_diff(
    logits: torch.Tensor,
    clean_logits: torch.Tensor,
    input_length: torch.Tensor,
    labels: Union[torch.Tensor, Tuple[List[int], List[int]]],
    mean: bool = True,
    loss: bool = False
):
    # Handle tuple input: (correct_list, incorrect_list)
    if isinstance(labels, tuple):
        correct, incorrect = labels
        labels = torch.tensor(list(zip(correct, incorrect)), dtype=torch.long, device=logits.device)

    logits = get_logit_positions(logits, input_length)  # [batch, vocab]
    good_bad = torch.gather(logits, -1, labels)         # [batch, 2]
    results = good_bad[:, 0] - good_bad[:, 1]           # logit difference

    if loss:
        results = -results
    if mean:
        return results.mean()
    return results