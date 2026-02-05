from .sparse_gpt import SparseGPTPruner
from .wanda import WandaPruner, prune_with_wanda
from .slice_gpt import SliceGPTPruner, slice_model_with_slicegpt

__all__ = [
    'SparseGPTPruner',
    'WandaPruner',
    'prune_with_wanda',
    'SliceGPTPruner',
    'slice_model_with_slicegpt',
]
