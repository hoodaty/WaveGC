from torch_geometric.datasets import LRGBDataset
import torch_geometric.transforms as T
from ..transforms.spectral import WaveGCSpectralTransform


class PeptidesFuncDataset(LRGBDataset):
    def __init__(self, root):
        # Long-Range settings per Appendix C.1 [cite: 268]
        # Full Spectrum (top_k_pct=1.0), No Thresholding (threshold=0.0)
        pre_transform = T.Compose([
            WaveGCSpectralTransform(mode='long', top_k_pct=1.0)
        ])

        super().__init__(root=root, name='Peptides-func', pre_transform=pre_transform)

    def __getitem__(self, idx):
        return super().__getitem__(idx)
