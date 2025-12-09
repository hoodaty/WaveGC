import torch
import torch.nn as nn
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.nn import GCN, GatedGraphConv

from typing import *

from src.layer.utils import FFN


# class WaveConv(nn.Module):

#     def __init__(self, hidden_dim, K, J, tight_frames):
#         super().__init__()
#         self.J = J
#         self.K = K
#         self.rho = K // 2
#         self.SM_kernels = nn.ModuleList(
#             [FFN(hidden_dim, hidden_dim) for j in range(self.J + 1)]
#         )
#         self.proj_final = nn.Linear(hidden_dim * (self.J + 1), hidden_dim)
#         self.act = nn.GELU()
#         self.tight_frames = tight_frames

#     def get_transformed_chebyshev(self, eigvs):
#         x = eigvs - 1
#         # T_0(x) = 1, T_1(x) = x
#         T_list = [torch.ones_like(x), x]

#         for k in range(2, self.K + 1):
#             # T_k = 2 * x * T_{k-1} - T_{k-2}
#             T_next = 2 * x * T_list[-1] - T_list[-2]
#             T_list.append(T_next)

#         # T_new = 0.5 * (-T_old + 1)
#         T_transformed = [0.5 * (-t + 1) for t in T_list]

#         return T_transformed

#     def generate_g(self, a, scaled_eigvs):
#         mask = (scaled_eigvs <= 2.0).float()
#         safe_eigvs = torch.clamp(scaled_eigvs, max=2.0)

#         T_even = self.get_transformed_chebyshev(safe_eigvs)[::2][1:]  # T0 is always 0
#         return torch.einsum("bi, bik-> bk", a, torch.stack(T_even, dim=1)) * mask

#     def generate_h(self, b, eigvs):
#         T_odd = self.get_transformed_chebyshev(eigvs)[1::2]
#         return torch.einsum("bi, bik-> bk", b, torch.stack(T_odd, dim=1))

#     def forward(self, x, Us, eigvs, a_tilde, b_tilde, scale_tilde):

#         h_g = []
#         v_sq = 0

#         for j in range(self.J + 1):
#             if j == 0:
#                 h_g.append(self.generate_h(b_tilde, eigvs))
#             else:
#                 h_g.append(
#                     self.generate_g(a_tilde, scale_tilde[:, j - 1].view(-1, 1) * eigvs)
#                 )

#             v_sq += h_g[j] ** 2

#         h_g = torch.stack(h_g)  # [1+J, B, N, N]

#         if self.tight_frames:
#             h_g /= torch.sqrt(v_sq) + 1e-6

#         U_lambda = torch.einsum("bik, jbk -> jbik", Us, h_g)
#         T = torch.einsum("jbil, blk -> jbik", U_lambda, Us.transpose(1, 2))

#         H = []
#         for j in range(self.J + 1):
#             WSH = self.SM_kernels[j](torch.einsum("bik, bkd -> bid", T[j], x))
#             H.append(torch.einsum("bik, bkj -> bij", T[j].transpose(-1, -2), WSH))

#         H = torch.cat(H, dim=-1)

#         return self.act(self.proj_final(H))


class WaveConv(nn.Module):

    def __init__(self, hidden_dim, K, J, tight_frames, threshold=0.1):
        super().__init__()
        self.J = J
        self.K = K
        self.rho = K // 2
        self.threshold = threshold
        self.SM_kernels = nn.ModuleList(
            [FFN(hidden_dim, hidden_dim) for j in range(self.J + 1)]
        )
        self.proj_final = nn.Linear(hidden_dim * (self.J + 1), hidden_dim)
        self.act = nn.GELU()
        self.tight_frames = tight_frames

    def get_transformed_chebyshev(self, eigvs):
        x = eigvs - 1
        # T_0(x) = 1, T_1(x) = x
        T_list = [torch.ones_like(x), x]

        for k in range(2, self.K + 1):
            # T_k = 2 * x * T_{k-1} - T_{k-2}
            T_next = 2 * x * T_list[-1] - T_list[-2]
            T_list.append(T_next)

        # T_new = 0.5 * (-T_old + 1)
        T_transformed = [0.5 * (-t + 1) for t in T_list]

        return T_transformed

    def generate_g(self, a, scaled_eigvs):
        mask = (scaled_eigvs <= 2.0).float()
        safe_eigvs = torch.clamp(scaled_eigvs, max=2.0)

        T_even = self.get_transformed_chebyshev(safe_eigvs)[::2][1:]  # T0 is always 0
        return torch.einsum("bi, bik-> bk", a, torch.stack(T_even, -2)) * mask # is masking ok ???

    def generate_h(self, b, eigvs):
        T_odd = self.get_transformed_chebyshev(eigvs)[1::2]
        return torch.einsum("bi, bik-> bk", b, torch.stack(T_odd, -2)) # (1, K)

    def forward(self, x, eigvs, U, a_tilde, b_tilde, scale_tilde):

        h_g = []
        v_sq = 0

        for j in range(self.J + 1):
            if j == 0:
                phi = self.generate_h(b_tilde, eigvs)
                h_g.append(torch.where(torch.abs(phi)>=self.threshold, phi, 0))
            else:
                psi = self.generate_g(a_tilde, scale_tilde[:, j - 1].view(-1, 1) * eigvs)
                h_g.append(
                    torch.where(torch.abs(psi)>=self.threshold, psi, 0)
                )

            v_sq += h_g[j] ** 2


        h_g = torch.stack(h_g).squeeze(1)

        if self.tight_frames:
            h_g /= torch.sqrt(v_sq) + 1e-6

        Z = torch.einsum("nk, nd -> kd", U, x)
        H = []
        for j in range(self.J + 1):

            Tj = torch.matmul(U, h_g[j].view(-1,1)*Z)
            Tj = self.SM_kernels[j](Tj)
            Tj = torch.einsum("nk, nd -> kd", U, Tj)
            Tj = torch.matmul(U, h_g[j].view(-1,1)*Tj)
            H.append( Tj )
            
        H = torch.cat(H, dim=-1)

        return self.act(self.proj_final(H))

class WaveGC(nn.Module):

    def __init__(
        self,
        inp_dim,
        hidden_dim,
        mpnn,
        K,
        J,
        tight_frames,
        dropout,
        ffn_hidden_num,
        mpnn_hidden_num,
        aggr,
    ):
        super().__init__()

        self.ffn = FFN(
            inp_dim=hidden_dim, out_dim=hidden_dim, hidden_dim=ffn_hidden_num
        )

        if mpnn == "gcn":
            self.mpnn = GCN(
                in_channels=inp_dim,
                hidden_channels=hidden_dim,
                num_layers=mpnn_hidden_num,
                out_channels=hidden_dim,
                act="gelu",
                dropout=dropout,
            )
        else:
            self.mpnn = GatedGraphConv(
                out_channels=hidden_dim, num_layers=mpnn_hidden_num, aggr=aggr
            )  # hidden_num = ?

        self.waveconv = WaveConv(hidden_dim=hidden_dim, K=K, J=J, tight_frames=tight_frames)

        self.norm_mpnn = nn.LayerNorm(hidden_dim)
        self.norm_wave = nn.LayerNorm(hidden_dim)
        self.norm_out = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: Tensor,
        edge_index: Union[Tensor, SparseTensor],
        eigvs: Tensor,
        U: Tensor,
        a_tilde: Tensor,
        b_tilde: Tensor,
        scale_tilde: Tensor,
        **kwargs
    ):

        z = self.mpnn(
            x,
            edge_index,
            **kwargs
        )
        z = self.norm_mpnn(x + z)

        w = self.waveconv(x, eigvs, U, a_tilde, b_tilde, scale_tilde)
        w = self.norm_wave(x + w)

        f = w + z

        return self.norm_out(f + self.ffn(f))
