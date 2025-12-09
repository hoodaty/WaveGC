from torch_geometric.datasets import LRGBDataset
import torch_geometric.transforms as T
from ..transforms.spectral import WaveGCSpectralTransform


class PeptidesStructDataset(LRGBDataset):
    def __init__(self, root):
        pre_transform = T.Compose([
            # Long-range settings: Full spectrum
            WaveGCSpectralTransform(mode='long', top_k_pct=1.0)
        ])
        super().__init__(root=root, name='Peptides-struct', pre_transform=pre_transform)

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        return data
