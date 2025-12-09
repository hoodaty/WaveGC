import torch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch


def collate_fn(dataset_items: list):
    """
    Universal Dense Collator for WaveGC.
    Handles both Short-Range (Node Classif with masks) and Long-Range (Graph Regr without masks).
    """
    batch_size = len(dataset_items)
    
    # 1. Determine Dimensions
    max_nodes = max([item.num_nodes for item in dataset_items])
    # FIX: Handle 1D eigvs shape correctly
    max_k = max([item.eigvs.shape[0] for item in dataset_items])
    feat_dim = dataset_items[0].x.shape[1]
    
    # 2. Allocate Dense Tensors
    x_padded = torch.zeros((batch_size, max_nodes, feat_dim))
    eigvs_padded = torch.zeros((batch_size, max_k))
    U_padded = torch.zeros((batch_size, max_nodes, max_k))
    eigvs_mask = torch.ones((batch_size, max_k), dtype=torch.bool)  # Default True (Padding)
    
    # 3. Handle Optional Masks (Only allocate if first item has them)
    has_masks = hasattr(dataset_items[0], 'train_mask')
    if has_masks:
        train_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool)
        test_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool)
    else:
        train_mask = None
        test_mask = None

    labels = []

    for i, data in enumerate(dataset_items):
        num_n = data.num_nodes
        # FIX: Use shape[0] for 1D vector
        num_k = data.eigvs.shape[0]

        # Spatial Features
        x_padded[i, :num_n, :] = data.x.float()

        # Spectral Features
        eigvs_padded[i, :num_k] = data.eigvs
        U_padded[i, :num_n, :num_k] = data.U

        # Eigenvalue Mask (False = Real Data)
        eigvs_mask[i, :num_k] = False
        
        # Optional Node Masks (for Amazon Computers)
        if has_masks and train_mask is not None and test_mask is not None:
            train_mask[i, :num_n] = data.train_mask
            test_mask[i, :num_n] = data.test_mask

        labels.append(data.y)

    # Create Sparse Edge Index for MPNN branch
    pyg_batch = Batch.from_data_list(dataset_items)
    sparse_edge_index = pyg_batch.edge_index if hasattr(pyg_batch, 'edge_index') and pyg_batch.edge_index is not None else torch.zeros((2, 0), dtype=torch.long)  # type: ignore

    batch_dict = {
        "x": x_padded,
        "eigvs": eigvs_padded,
        "U": U_padded,
        "eigvs_mask": eigvs_mask,
        "labels": torch.stack(labels),
        "edge_index": sparse_edge_index,
        "batch_idx": pyg_batch.batch  # type: ignore
    }

    # Only add masks to batch if they exist
    if has_masks:
        batch_dict["train_mask"] = train_mask
        batch_dict["test_mask"] = test_mask

    return batch_dict
