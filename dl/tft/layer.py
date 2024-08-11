"""
Google Temporal Fusion Transformer - Layers
"""

import torch
import torch.nn as nn


class GatedLinearUnit(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GatedLinearUnit, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.cg = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.fc(x) * self.sigmoid(self.cg(x))


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, context_dim=None, dropout=0.1):
        super(GatedResidualNetwork, self).__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # w1~w3 reference to the paper
        self.w2 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.context_dim is not None:
            self.w3 = nn.Linear(self.context_dim, self.hidden_dim, bias=False)

        # ELU activation
        self.elu = nn.ELU()

        # linear projection
        self.w1 = nn.Linear(self.hidden_dim, self.hidden_dim)

        # dropout
        self.dropout = nn.Dropout(dropout)

        # gated layer
        self.gated_layer = GatedLinearUnit(self.hidden_dim, self.output_dim)

        # residual layer and layer norm
        self.residual_layer = nn.Linear(self.input_dim, self.output_dim)
        self.layer_norm = nn.LayerNorm(self.output_dim)

    def forward(self, a, c=None):
        projected_a = self.w2(a)
        if self.context_dim is not None:
            projected_c = self.w3(c)
            if len(projected_a.shape) != len(projected_c.shape):
                projected_c = projected_c.unsqueeze(dim=1)
        n2 = self.elu(
            projected_a + projected_c if self.context_dim is not None else projected_a
        )
        n1 = self.w1(n2)
        n1 = self.dropout(n1)
        n1 = self.gated_layer(n1)
        output = self.layer_norm(n1 + self.residual_layer(a))
        return output


class VariableSelectionNetworks(nn.Module):
    def __init__(self, variable_num, hidden_dim, context_dim=None, dropout=0.1):
        super(VariableSelectionNetworks, self).__init__()
        self.variable_num = variable_num
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim

        self.per_feature_grn = nn.ModuleList(
            [
                GatedResidualNetwork(
                    self.hidden_dim, self.hidden_dim, self.hidden_dim, None, dropout
                )
                for _ in range(self.variable_num)
            ]
        )

        self.selection_weight_grn = GatedResidualNetwork(
            self.hidden_dim * self.variable_num,
            self.hidden_dim,
            self.variable_num,
            self.context_dim,
            dropout,
        )

    def forward(self, *x, c=None):
        # x is a list of tensor, each tensor has shape (batch_size, seq_len, hidden_dim), seq_len is optional
        # c is the context tensor, shape (batch_size, context_dim)
        # output is a tensor with shape (batch_size, hidden_dim)

        # unlinear projection for each feature
        unlinear_x = [grn(x[i]) for i, grn in enumerate(self.per_feature_grn)]

        # concat all features
        concat_x = torch.cat(
            unlinear_x, dim=-1
        )  # (batch_size, seq_len, hidden_dim * variable_num)

        # selection weight
        selection_weight = self.selection_weight_grn(
            concat_x, c
        )  # (batch_size, seq_len, variable_num)
        selection_weight = torch.softmax(selection_weight, dim=-1)

        # weighted sum
        unlinear_x = torch.stack(unlinear_x, dim=-1)
        output = (unlinear_x * selection_weight.unsqueeze(dim=-2)).sum(dim=-1)
        return output


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # q, k, v: (batch_size, seq_len, d_k)
        # mask: (batch_size, seq_len, seq_len)
        attn = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.d_k, dtype=torch.float32)
        )
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, dropout=0.1):
        super(InterpretableMultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = self.d_q = self.d_v = d_model // n_head
        self.dropout = nn.Dropout(dropout)

        self.v_layer = nn.Linear(self.d_model, self.d_v, bias=False)
        self.q_layers = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_q, bias=False) for _ in range(self.n_head)]
        )
        self.k_layers = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_k, bias=False) for _ in range(self.n_head)]
        )
        self.v_layers = nn.ModuleList([self.v_layer for _ in range(self.n_head)])

        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        self.fc = nn.Linear(self.d_v, self.d_model, bias=False)

    def forward(self, q, k, v, mask=None):
        # q, k, v: (batch_size, seq_len, d_model)
        # mask: (batch_size, seq_len, seq_len)
        heads = []
        attns = []
        for i in range(self.n_head):
            qs = self.q_layers[i](q)  # (batch_size, seq_len, d_q)
            ks = self.k_layers[i](k)  # (batch_size, seq_len, d_k)
            vs = self.v_layers[i](v)  # (batch_size, seq_len, d_v)
            head, attn = self.attention(qs, ks, vs, mask)
            heads.append(head)
            attns.append(attn)

        head = torch.stack(heads, dim=-1)  # (batch_size, seq_len, d_v, n_head)
        head = head.mean(dim=-1)  # (batch_size, seq_len, d_v)

        attn = torch.stack(attns, dim=-1)  # (batch_size, seq_len, seq_len, n_head)
        attn = attn.mean(dim=-1)  # (batch_size, seq_len, seq_len)

        output = self.fc(head)  # (batch_size, seq_len, d_model)
        output = self.dropout(output)
        return output, attn
