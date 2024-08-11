"""
Google Temporal Fusion Transformer
"""

import torch
from layer import (
    GatedLinearUnit,
    GatedResidualNetwork,
    InterpretableMultiHeadAttention,
    VariableSelectionNetworks,
)
from torch import nn


def quantile_loss(y_pred, y_true, quantiles):
    """
    Parameters
    ----------
    y_pred: torch.Tensor
        predicted quantiles, shape (batch_size, future_seq_len, len(output_quantiles))
    y_true: torch.Tensor
        true values, shape (batch_size, future_seq_len)
    quantiles: list

    Returns
    -------
    loss: torch.Tensor
        loss
    """
    quantiles = torch.FloatTensor(quantiles).to(y_pred.device)
    y_true = y_true.unsqueeze(dim=-1)
    e = y_true - y_pred
    loss = torch.max(
        quantiles * e, (quantiles - 1) * e
    )  # (batch_size, future_seq_len, len(output_quantiles))

    return loss.mean()


class TemporalFusionDecoder(nn.Module):
    def __init__(self, hidden_dim, n_head, dropout=0.1):
        super(TemporalFusionDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.dropout = dropout

        # static enrichment
        self.static_enrichment = GatedResidualNetwork(
            self.hidden_dim, self.hidden_dim, self.hidden_dim, dropout=dropout
        )

        # Temporal Self-Attention
        # masked interpretable multi-head attention
        self.masked_interpretable_attention = InterpretableMultiHeadAttention(
            n_head=n_head,
            d_model=self.hidden_dim,
            dropout=dropout,
        )
        # Gated Add & Norm
        self.temporal_self_attention_glu = GatedLinearUnit(
            self.hidden_dim, self.hidden_dim
        )
        self.temporal_self_attention_layer_norm = nn.LayerNorm(self.hidden_dim)

        # position-wise feed forward
        self.position_wise_feed_forward = GatedResidualNetwork(
            self.hidden_dim, self.hidden_dim, self.hidden_dim, dropout=dropout
        )

    def forward(self, c_e, x):
        """
        Parameters
        ----------
        c_e: torch.Tensor
            encoded static features , shape (batch_size, hidden_dim)
        x: torch.Tensor
            encoded time series features, shape (batch_size, seq_len, hidden_dim)

        Returns
        -------
        """
        # static enrichment
        x = self.static_enrichment(
            x, c=c_e.unsqueeze(dim=-2)
        )  # (batch_size, seq_len, hidden_dim)

        # masked interpretable multi-head attention
        mask = torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=0).T.to(x.device)
        mask = mask.unsqueeze(0).repeat(x.size(0), 1, 1)  # (batch_size, seq_len, seq_len)

        attentioned_x, attn = self.masked_interpretable_attention(x, x, x, mask=mask)

        # Gated Add & Norm
        gated_attention_x = self.temporal_self_attention_glu(attentioned_x)
        x = self.temporal_self_attention_layer_norm(x + gated_attention_x)

        # position-wise feed forward
        x = self.position_wise_feed_forward(x)

        return x, attn


# Hyperparameters class
class TFTConfig:
    def __init__(
        self,
        static_numeric_features: list = list(),
        static_categorical_features: dict = dict(),  # key, num_embeddings
        past_numeric_features: list = list(),
        past_categorical_features: dict = dict(),  # key, num_embeddings
        future_numeric_features: list = list(),
        future_categorical_features: dict = dict(),  # key, num_embeddings
        multi_hot_data: list = list(),  # indicate which features are multi-hot
        output_quantiles: list = list([0.1, 0.5, 0.9]),
        hidden_dim: int = 64,
        n_head: int = 8,
        dropout: float = 0.1,
    ):
        self.static_numeric_features = static_numeric_features
        self.static_categorical_features = static_categorical_features
        self.past_numeric_features = past_numeric_features
        self.past_categorical_features = past_categorical_features
        self.future_numeric_features = future_numeric_features
        self.future_categorical_features = future_categorical_features
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.dropout = dropout
        self.output_quantiles = output_quantiles
        self.multi_hot_data = multi_hot_data

    def __repr__(self):
        return str(vars(self))

    def __str__(self):
        return str(vars(self))

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def from_dict(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)
        return self


class TemporalFusionTransformer(nn.Module):
    def __init__(self, config: TFTConfig):
        super(TemporalFusionTransformer, self).__init__()
        self.config = config
        # static numeric projection
        if self.config.static_numeric_features:
            self.static_numeric_projection = nn.ModuleDict(
                {
                    feature: nn.Linear(1, self.config.hidden_dim)
                    for feature in self.config.static_numeric_features
                }
            )
        # static categorical projection
        if self.config.static_categorical_features:
            self.static_categorical_projection = nn.ModuleDict(
                {
                    feature: nn.Embedding(
                        num_embeddings, self.config.hidden_dim, padding_idx=0
                    )
                    for feature, num_embeddings in self.config.static_categorical_features.items()
                }
            )
        # past numeric projection
        if self.config.past_numeric_features:
            self.past_numeric_projection = nn.ModuleDict(
                {
                    feature: nn.Linear(1, self.config.hidden_dim)
                    for feature in self.config.past_numeric_features
                }
            )
        # past categorical projection
        if self.config.past_categorical_features:
            self.past_categorical_projection = nn.ModuleDict(
                {
                    feature: nn.Embedding(
                        num_embeddings, self.config.hidden_dim, padding_idx=0
                    )
                    for feature, num_embeddings in self.config.past_categorical_features.items()
                }
            )
        # future numeric projection
        if self.config.future_numeric_features:
            self.future_numeric_projection = nn.ModuleDict(
                {
                    feature: nn.Linear(1, self.config.hidden_dim)
                    for feature in self.config.future_numeric_features
                }
            )
        # future categorical projection
        if self.config.future_categorical_features:
            self.future_categorical_projection = nn.ModuleDict(
                {
                    feature: nn.Embedding(
                        num_embeddings, self.config.hidden_dim, padding_idx=0
                    )
                    for feature, num_embeddings in self.config.future_categorical_features.items()
                }
            )

        # variable selection network
        static_variable_num = len(self.config.static_numeric_features) + len(
            self.config.static_categorical_features
        )
        self.static_variable_selection = VariableSelectionNetworks(
            variable_num=static_variable_num,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
        )
        past_variable_num = len(self.config.past_numeric_features) + len(
            self.config.past_categorical_features
        )
        self.past_variable_selection = VariableSelectionNetworks(
            variable_num=past_variable_num,
            hidden_dim=self.config.hidden_dim,
            context_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
        )
        future_variable_num = len(self.config.future_numeric_features) + len(
            self.config.future_categorical_features
        )
        self.future_variable_selection = VariableSelectionNetworks(
            variable_num=future_variable_num,
            hidden_dim=self.config.hidden_dim,
            context_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
        )

        # static covariate encoding
        # c_s for time series features selection (as context)
        self.c_s_encoder = GatedResidualNetwork(
            self.config.hidden_dim,
            self.config.hidden_dim,
            self.config.hidden_dim,
            dropout=self.config.dropout,
        )
        # c_e for Temporal Fusion Decoder (as context)
        self.c_e_encoder = GatedResidualNetwork(
            self.config.hidden_dim,
            self.config.hidden_dim,
            self.config.hidden_dim,
            dropout=self.config.dropout,
        )
        # c_c for LSTM (as cell initial state)
        self.c_c_encoder = GatedResidualNetwork(
            self.config.hidden_dim,
            self.config.hidden_dim,
            self.config.hidden_dim,
            dropout=self.config.dropout,
        )
        # c_h for LSTM (as hidden initial state)
        self.c_h_encoder = GatedResidualNetwork(
            self.config.hidden_dim,
            self.config.hidden_dim,
            self.config.hidden_dim,
            dropout=self.config.dropout,
        )

        # LSTM Encoder (for past features)
        self.lstm_encoder = nn.LSTM(
            input_size=self.config.hidden_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # LSTM Decoder (for future features)
        self.lstm_decoder = nn.LSTM(
            input_size=self.config.hidden_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Feature process gate add & norm
        self.dropout = nn.Dropout(self.config.dropout)
        self.feature_process_glu = GatedLinearUnit(
            self.config.hidden_dim, self.config.hidden_dim
        )
        self.feature_process_layer_norm = nn.LayerNorm(self.config.hidden_dim)

        # Temporal Fusion Decoder
        self.temporal_fusion_decoder = TemporalFusionDecoder(
            self.config.hidden_dim, self.config.n_head, dropout=self.config.dropout
        )

        # Output
        # output gate add & norm
        self.output_glu = GatedLinearUnit(self.config.hidden_dim, self.config.hidden_dim)
        self.output_layer_norm = nn.LayerNorm(self.config.hidden_dim)
        # output projection
        self.output_projection = nn.Linear(
            self.config.hidden_dim, len(self.config.output_quantiles)
        )

    def _get_embedding(self, embedding, feature_data, mutil_hot=False):
        if not mutil_hot:  # tensor(batch_size,) or (batch_size, seq_len)
            return embedding(
                feature_data
            )  # (batch_size, hidden_dim) or (batch_size, seq_len, hidden_dim)
        else:  # tensor(batch_size, multi hot index) or (batch_size, seq_len, multi hot index)
            # mean pooling
            return (
                embedding(feature_data).sum(dim=-2)
                / (feature_data > 0).sum(dim=-1, keepdim=True).float()
            )  # (batch_size, hidden_dim) or (batch_size, seq_len, hidden_dim)

    def forward(
        self,
        static_numeric_data: dict = dict(),  # {feature_name: tensor(batch_size,)}
        static_categorical_data: dict = dict(),  # {feature_name: tensor(batch_size,)}
        past_numeric_data: dict = dict(),  # {feature_name: tensor(batch_size, past_seq_len)}
        past_categorical_data: dict = dict(),  # {feature_name: tensor(batch_size, past_seq_len)}
        future_numeric_data: dict = dict(),  # {feature_name: tensor(batch_size, future_seq_len)}
        future_categorical_data: dict = dict(),  # {feature_name: tensor(batch_size, future_seq_len)}
    ):

        # static numeric projection
        static_x = []
        if static_numeric_data:
            static_x.extend(
                [
                    self.static_numeric_projection[name](
                        static_numeric_data[name].unsqueeze(dim=-1)
                    )
                    for name in sorted(self.config.static_numeric_features)
                ]
            )
        if static_categorical_data:
            static_x.extend(
                [
                    self._get_embedding(
                        self.static_categorical_projection[name],
                        static_categorical_data[name],
                        name in self.config.multi_hot_data,
                    )
                    for name in sorted(self.config.static_categorical_features.keys())
                ]
            )

        # past numeric projection
        past_x = []
        if past_numeric_data:
            past_x.extend(
                [
                    self.past_numeric_projection[name](
                        past_numeric_data[name].unsqueeze(dim=-1)
                    )
                    for name in sorted(self.config.past_numeric_features)
                ]
            )
        if past_categorical_data:
            past_x.extend(
                [
                    self._get_embedding(
                        self.past_categorical_projection[name],
                        past_categorical_data[name],
                        name in self.config.multi_hot_data,
                    )
                    for name in sorted(self.config.past_categorical_features.keys())
                ]
            )

        # future numeric projection
        future_x = []
        if future_numeric_data:
            future_x.extend(
                [
                    self.future_numeric_projection[name](
                        future_numeric_data[name].unsqueeze(dim=-1)
                    )
                    for name in sorted(self.config.future_numeric_features)
                ]
            )
        if future_categorical_data:
            future_x.extend(
                [
                    self._get_embedding(
                        self.future_categorical_projection[name],
                        future_categorical_data[name],
                        name in self.config.multi_hot_data,
                    )
                    for name in sorted(self.config.future_categorical_features.keys())
                ]
            )

        # variable selection network
        static_x = self.static_variable_selection(*static_x)  # (batch_size, hidden_dim)
        c_s = self.c_s_encoder(static_x)  # (batch_size, hidden_dim)
        c_e = self.c_e_encoder(static_x)  # (batch_size, hidden_dim)
        c_c = self.c_c_encoder(static_x)  # (batch_size, hidden_dim)
        c_h = self.c_h_encoder(static_x)  # (batch_size, hidden_dim)

        past_x = self.past_variable_selection(
            *past_x, c=c_s
        )  # (batch_size, past_seq_len, hidden_dim)
        future_x = self.future_variable_selection(
            *future_x, c=c_s
        )  # (batch_size, future_seq_len, hidden_dim)

        # for batch lstm, the h_0, c_0 shape is (num_layers * num_directions, batch, hidden_size)
        c_h = c_h.unsqueeze(dim=0)  # (1, batch_size, hidden_dim)
        c_c = c_c.unsqueeze(dim=0)  # (1, batch_size, hidden_dim)

        # LSTM Encoder
        encoded_past_x, (hn, cn) = self.lstm_encoder(past_x, (c_h, c_c))
        # LSTM Decoder
        encoded_future_x, _ = self.lstm_decoder(future_x, (hn, cn))

        # Feature process gate add & norm
        x = torch.cat(
            [encoded_past_x, encoded_future_x], dim=1
        )  # (batch_size, seq_len, hidden_dim)
        gated_x = self.feature_process_glu(x)  # (batch_size, seq_len, hidden_dim)
        selected_x = torch.cat(
            [past_x, future_x], dim=1
        )  # (batch_size, seq_len, hidden_dim)
        x = self.feature_process_layer_norm(
            selected_x + gated_x
        )  # (batch_size, seq_len, hidden_dim)

        # Temporal Fusion Decoder
        encoded_x, attn = self.temporal_fusion_decoder(
            c_e, x
        )  # (batch_size, seq_len, hidden_dim)

        # Output
        gated_x = self.output_glu(encoded_x)  # (batch_size, seq_len, hidden_dim)
        x = self.output_layer_norm(x + gated_x)  # (batch_size, seq_len, hidden_dim)
        x = self.output_projection(x)  # (batch_size, seq_len, len(output_quantiles)

        # only output the future time steps
        x = x[
            :, past_x.size(1) :, :
        ]  # (batch_size, future_seq_len, len(output_quantiles))
        attn = attn[:, past_x.size(1) :, :]  # (batch_size, future_seq_len, seq_len)

        return x, attn
