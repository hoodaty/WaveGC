import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix


class WaveGCSpectralTransform:
    def __init__(self, mode='long', top_k=None, top_k_pct=1.0, threshold=0.0):
        """
        Args:
            mode (str): 'short' or 'long'.
            top_k (int, optional): Exact number of eigenvalues to keep.
                                   Prioritized over top_k_pct if set.
            top_k_pct (float): Percentage of eigenvalues to keep (default 1.0).
            threshold (float): Threshold for sparsifying U (default 0.0).
        """
        self.mode = mode
        self.top_k = top_k
        self.top_k_pct = top_k_pct
        self.threshold = threshold

    def __call__(self, data: Data) -> Data:
        N = data.num_nodes
        assert isinstance(N, int) and N > 0, "num_nodes must be a positive integer"

        # 1. Determine target k (number of eigenvalues)
        if self.top_k is not None:
            k = self.top_k
        else:
            k = int(N * self.top_k_pct)
            if k < 2:
                k = 2  # Safety floor

        # 2. Compute Sparse Laplacian
        # L = I - D^-0.5 A D^-0.5
        assert data.edge_index is not None, "Data object must have edge_index"
        edge_index, edge_weight = get_laplacian(
            data.edge_index,
            edge_weight=None,
            normalization='sym',
            num_nodes=N
        )

        # Convert directly to Scipy CSR (O(E) memory)
        L_sparse = to_scipy_sparse_matrix(edge_index, edge_attr=edge_weight, num_nodes=N)

        # 3. Eigendecomposition (The logic to handle N < k)        # Case A: Graph is smaller than requested k (Padding required)
        # OR Graph is too small for sparse iterative solver (k >= N-1)
        if k >= N - 1:
            # We must compute ALL eigenvalues.
            # For small graphs (Peptides ~150 nodes), converting to dense is efficient and stable.
            L_dense = L_sparse.toarray()
            eig_vals, U = eigh(L_dense)

            # Pad if we requested more than N
            if k > N:
                pad_size = k - N
                # Pad Eigenvalues with 0 (or a dummy value, usually 0 for frequency)
                eig_vals = np.pad(eig_vals, (0, pad_size), 'constant', constant_values=0)
                # Pad Eigenvectors with 0
                U = np.pad(U, ((0, 0), (0, pad_size)), 'constant', constant_values=0)

        # Case B: Large graph, seeking subset of spectrum (Optimization)
        else:
            # Use Arpack (Lanczos) for subset of eigenvalues
            # which='SM' -> Smallest Magnitude (Low Frequencies/Smooth signals)
            try:
                eig_vals, U = eigsh(L_sparse, k=k, which='SM', return_eigenvectors=True)
            except RuntimeError:
                # Fallback if Arpack fails to converge
                L_dense = L_sparse.toarray()
                eig_vals, U = eigh(L_dense)
                eig_vals = eig_vals[:k]
                U = U[:, :k]

        # 4. Convert to Tensor
        # eig_vals: [k]
        # U: [N, k]
        eig_vals = torch.from_numpy(eig_vals).float()
        U = torch.from_numpy(U).float()

        # Clamp for numerical stability (Theoretically in [0, 2])
        eig_vals = torch.clamp(eig_vals, 0.0, 2.0)

        # [cite_start]5. Sparsification (Short-Range Nuance) [cite: 763]
        if self.threshold > 0:
            mask = torch.abs(U) >= self.threshold
            U = U * mask

        # Store results
        data.eigvs = eig_vals.view(1, -1)
        data.U = U

        return data
