import torch
import torch.nn as nn
import numpy as np
from graphormer_3d import Graphormer3D

class Dynamics(nn.Module):
    def __init__(
        self, 
        n_dims=3, 
        in_node_nf=9, 
        context_node_nf=1, 
        hidden_nf=128, 
        device='cpu', 
        n_layers=6, 
        condition_time=True, 
        tanh=False, 
        norm_constant=1e-6, 
        normalization_factor=100, 
        aggregation_method='sum', 
        model='graphormer',
        ffn_embedding_dim=3072,
        attention_heads=8,
        coords_range=10,
        dropout=0.1,
        activation_dropout= 0.1,
    ):
        super().__init__()
        self.n_dims = n_dims
        self.context_node_nf = context_node_nf
        self.condition_time = condition_time
        self.model = model
        self.device = device

        in_node_nf = in_node_nf + context_node_nf + condition_time
        if self.model == 'graphormer':
            self.dynamics = Graphormer3D(            
                in_node_nf=in_node_nf,
                hidden_nf=hidden_nf, 
                device=device,
                n_layers=n_layers,
                tanh=tanh,
                norm_constant=norm_constant,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                ffn_embedding_dim=ffn_embedding_dim,
                attention_heads=attention_heads,
                coords_range=coords_range,
                dropout=dropout,
                activation_dropout= activation_dropout,
            )
        else:
            raise NotImplementedError
        
        self.edge_cache = {}

    def forward(self, t, xh, node_mask, linker_mask, edge_mask, context, training):
        """
        - t: (B)
        - xh: (B, N, D), where D = 3 + nf
        - node_mask: (B, N, 1)
        - edge_mask: (B, N, N)
        - context: (B, N, C)
        """
        
        bs, n_nodes = xh.shape[0], xh.shape[1]

        edges = self.get_edges(n_nodes, bs)  # (2, B*N)
        node_mask = node_mask.view(bs * n_nodes, 1)  # (B*N, 1)
        linker_mask = linker_mask.view(bs * n_nodes, 1)  # (B*N, 1)
        
        # Reshaping node features & adding time feature
        xh = xh.view(bs * n_nodes, -1).clone() * node_mask  # (B*N, D)
        x = xh[:, :self.n_dims].clone()  # (B*N, 3)
        h = xh[:, self.n_dims:].clone()  # (B*N, nf)
        
        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t.view(bs, 1).repeat(1, n_nodes)
                h_time = h_time.view(bs * n_nodes, 1)
            h = torch.cat([h, h_time], dim=1)  # (B*N, nf+1)
        if context is not None:
            context = context.view(bs*n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)

        if self.model == 'graphormer':
            size=(bs, n_nodes)
            h_final, x_final = self.dynamics(
                h,
                x,
                edges,
                node_mask=node_mask,
                linker_mask=linker_mask,
                edge_mask=edge_mask,
                size=size,
            )
            vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case
        else:
            raise NotImplementedError

        # Slice off context size
        if context is not None:
            h_final = h_final[:, :-self.context_node_nf]

        # Slice off last dimension which represented time.
        if self.condition_time:
            h_final = h_final[:, :-1]

        vel = vel.view(bs, n_nodes, -1)  # (B, N, 3)
        h_final = h_final.view(bs, n_nodes, -1)  # (B, N, D)
        node_mask = node_mask.view(bs, n_nodes, 1)  # (B, N, 1)


        return torch.cat([vel, h_final], dim=2)
    
    def get_edges(self, n_nodes, batch_size):
        if n_nodes in self.edge_cache:
            edges_dic_b = self.edge_cache[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(self.device), torch.LongTensor(cols).to(self.device)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self.edge_cache[n_nodes] = {}
            return self.get_edges(n_nodes, batch_size)
