from torch_geometric.datasets import Amazon
import torch_geometric.transforms as T
from ..transforms.spectral import WaveGCSpectralTransform
from ..transforms.train_mask import TrainTestSplit


class AmazonComputerDataset(Amazon):
    def __init__(self, root, name='Computers', top_k_pct=0.30, train_mask=None, test_size=0.2, seed=42):
        pre_transform = T.Compose([
            # T.NormalizeFeatures(),
            T.ToUndirected(),
            # Short-range settings
            WaveGCSpectralTransform(mode='short', top_k_pct=top_k_pct),
            TrainTestSplit(train_mask=train_mask, test_size=test_size, seed=seed)
        ])
        super().__init__(root=root, name=name, pre_transform=pre_transform)

    # Compatibility with your repo's BaseDataset style
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        return data  # The collate_fn will handle the unpacking
