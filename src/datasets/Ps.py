import torch
import os.path as osp
from torch_geometric.datasets import LRGBDataset
from torch_geometric.data import Dataset
import torch_geometric.transforms as T
from ..transforms.spectral import WaveGCSpectralTransform

class PeptidesStructDataset(Dataset):  # Inherit from Dataset, not InMemoryDataset
    def __init__(self, root, split='train'):
        self.name = 'Peptides-struct'
        self.split = split
        assert split in ['train', 'val', 'test']
        
        # Create a wrapper transform that fixes edge_attr before spectral transform
        pre_transform = T.Compose([
            FixEdgeAttr(),  # Remove edge_attr to avoid dtype issues
            WaveGCSpectralTransform(mode='long', top_k_pct=1.0)
        ])
        
        super().__init__(root, transform=None, pre_transform=pre_transform)
        
        # Load the appropriate split
        path = osp.join(self.processed_dir, f'{split}.pt')
        loaded_data = torch.load(path)
        
        # Check what we loaded and handle accordingly
        if isinstance(loaded_data, list):
            self._data_list = loaded_data
        elif isinstance(loaded_data, tuple):
            # It's (data, slices) or (data, slices, sizes) format - need to unpack
            from torch_geometric.data import Data
            data = loaded_data[0]
            slices = loaded_data[1]
            # Ignore sizes if present (loaded_data[2])
            
            self._data_list = []
            num_graphs = len(slices['x']) - 1
            
            for i in range(num_graphs):
                item = Data()
                for key in data.keys():
                    if key in slices:
                        start, end = slices[key][i], slices[key][i+1]
                        if key == 'edge_index':
                            item[key] = data[key][:, start:end]
                        else:
                            item[key] = data[key][start:end]
                    else:
                        item[key] = data[key]
                self._data_list.append(item)
            print(f"Unpacked {len(self._data_list)} graphs from tuple format")
        else:
            raise RuntimeError(f"Unexpected data format: {type(loaded_data)}")

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        # Use the parent LRGBDataset's raw files
        return ['peptides-structural-train.csv.gz', 
                'peptides-structural-val.csv.gz',
                'peptides-structural-test.csv.gz']

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        # Use LRGBDataset's download mechanism
        from torch_geometric.datasets import LRGBDataset
        temp_dataset = LRGBDataset(root=self.root, name=self.name, split='train')

    def process(self):
        # Use LRGBDataset to load raw data for each split
        from torch_geometric.datasets import LRGBDataset
        
        for split in ['train', 'val', 'test']:
            print(f"Processing {split} split...")
            
            # Load using parent class without pre_transform
            temp_dataset = LRGBDataset(
                root=self.root, 
                name=self.name, 
                split=split,
                pre_transform=None
            )
            
            # Extract individual graphs properly
            data_list = []
            for i in range(len(temp_dataset)):
                data = temp_dataset.get(i)  # Use get() instead of indexing
                data_list.append(data)
            
            print(f"Extracted {len(data_list)} individual graphs")
            
            # Apply our transforms to each graph
            if self.pre_transform is not None:
                print(f"Applying spectral transform...")
                transformed_list = []
                for idx, data in enumerate(data_list):
                    if idx % 1000 == 0:
                        print(f"  Processing graph {idx}/{len(data_list)}")
                    transformed_data = self.pre_transform(data)
                    transformed_list.append(transformed_data)
                data_list = transformed_list
            
            # Save as list (each element is a separate Data object)
            print(f"Saving {len(data_list)} graphs to {split}.pt")
            torch.save(data_list, osp.join(self.processed_dir, f'{split}.pt'))
            print(f"Finished processing {split} split")

    def len(self):
        return len(self._data_list)
    
    def get(self, idx):
        return self._data_list[idx]
    
    def get_data_list(self):
        """Public method to access the full list of graphs"""
        return self._data_list


class FixEdgeAttr:
    """Transform to remove edge_attr before spectral processing"""
    def __call__(self, data):
        # Remove edge_attr completely to avoid dimension mismatches
        if hasattr(data, 'edge_attr'):
            delattr(data, 'edge_attr')
        # Ensure edge_index is proper format
        if hasattr(data, 'edge_index'):
            data.edge_index = data.edge_index.long().contiguous()
        return data
    
    def __repr__(self):
        return f'{self.__class__.__name__}()'