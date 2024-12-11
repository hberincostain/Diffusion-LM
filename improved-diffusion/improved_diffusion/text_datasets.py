# from PIL import Image
# import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import sys, os
import torch
from collections import defaultdict
from functools import partial
from itertools import chain

# Add to existing imports
sys.path.append('/content/GuidedSymbolicGPT/src')
from symbolic_dataset import SymbolicDataset
from tokenizer_ops import OPS_Tokenizer, OPS_Tokenizer_Config

def load_data_text(
    *, 
    data_dir, 
    batch_size, 
    image_size, 
    class_cond=False, 
    deterministic=False, 
    data_args=None,
    task_mode='points',
    model=None, 
    padding_mode='block', 
    split='train',
    load_vocab=None,
):
    """
    For a dataset, create a generator over (points, kwargs) pairs.

    Each element of the dataset is a tuple (points, { 'points': points, 'target_formula': formula_tokens })
    where:
    - points is a tensor of shape [batch_size, n_vars, n_points]
    - target_formula is a 1D tensor of token IDs representing the formula.

    :param data_dir: a dataset directory containing "train", "val", "test" splits.
    :param batch_size: the batch size of each returned pair.
    :param image_size: unused here, but kept for interface consistency.
    :param class_cond: unused for points mode.
    :param deterministic: if True, yield results in a deterministic order (no shuffle).
    :param task_mode: should be 'points' for this code.
    :param data_args: contains configuration such as n_var, data_dir, etc.
    :param model: unused for points mode.
    :param padding_mode: unused for points mode.
    :param split: which dataset split to load ('train', 'valid', 'test').
    :param load_vocab: unused for points mode.
    """
    print(f'Loading data with task mode: {task_mode}')
    task_mode = 'points'
    if task_mode == 'points':
        print('Loading point cloud dataset')
        
        # Load training data to determine depth and thus n_const
        train_data = SymbolicDataset(os.path.join(data_args.data_dir, "train", "properties.json"))
        depth = train_data.get_depth()
        n_const = max(1, 2**(depth-2))
        
        # Initialize tokenizer with correct parameters
        tokenizer_config = OPS_Tokenizer_Config(
            n_var=data_args.n_var,  
            n_const=n_const
        )
        tokenizer = OPS_Tokenizer(tokenizer_config)

        if split == 'train':
    # Here, train_data is presumably already a SymbolicDataset initialized with a file
            dataset = train_data
        elif split == 'valid':
            dataset = SymbolicDataset(os.path.join(data_args.data_dir, "val", "properties.json"))
        elif split == 'test':
            dataset = SymbolicDataset(os.path.join(data_args.data_dir, "test", "properties.json"))
        else:
            raise ValueError(f"Unknown split {split}")

        # Create dataset with correct parameters
        point_dataset = PointCloudDataset(
            dataset=dataset,
            tokenizer=tokenizer
        )

        # Create dataloader
        if deterministic:
            data_loader = DataLoader(
                point_dataset,
                batch_size=batch_size,
                drop_last=True,
                shuffle=False,
                num_workers=1,
            )
        else:
            data_loader = DataLoader(
                point_dataset,
                batch_size=batch_size,
                drop_last=True,
                shuffle=True,
                num_workers=1,
            )

        while True:
            yield from data_loader
    else:
        # If any other mode is attempted, raise an error since we removed that logic.
        raise ValueError(f"Unsupported task_mode: {task_mode}. This code only supports 'points' mode.")


class PointCloudDataset(Dataset):
    def __init__(self, dataset: SymbolicDataset, tokenizer: OPS_Tokenizer) -> None:
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_obj = self.dataset[idx]
        points = torch.tensor(data_obj.points_to_table(), dtype=torch.float32)
        target = torch.tensor(
            self.tokenizer.encode(data_obj.formula.graph_list()),
            dtype=torch.long
        )

        # Return format for the diffusion model:
        # First item: points (the data to be diffused)
        # Second item: dict with additional conditioning info (target formula)
        return points, {
            'points': points,
            'target_formula': target
        }


def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False):
    result = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    mask_ = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result


def _torch_collate_batch(examples, pad_token_id, max_length):
    """Collate `examples` into a batch, padding if necessary."""
    import numpy as np
    import torch

    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    result = examples[0].new_full([len(examples), max_length], pad_token_id)
    for i, example in enumerate(examples):
        result[i, : example.shape[0]] = example
    return result
