from torch_geometric.datasets import LRGBDataset
import torch_geometric.transforms as T
from ..transforms.spectral import WaveGCSpectralTransform


class PeptidesStructDataset(LRGBDataset):

    def __init__(self, root):
        spectral_transform = T.Compose([
            # FIX: Add max_freqs=500 to fix the PyG crash
            WaveGCSpectralTransform(mode='long', top_k_pct=1.0, max_freqs=500)
        ])

        super().__init__(
            root=root,
            name='Peptides-struct',
            pre_transform=spectral_transform
        )

    def __getitem__(self, idx):
        return super().__getitem__(idx)