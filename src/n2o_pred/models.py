import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class N2OPredictorRNN(nn.Module):
    """
    RNN model for N2O prediction

    Features:
        - Event-driven, each step corresponds to one observation
        - Input contains time interval features (time_delta, ferdur, sowdur)
        - Supports variable length sequences (using pack_padded_sequence)
    """

    def __init__(
        self,
        static_numeric_dim: int,
        static_categorical_dim: int,
        dynamic_numeric_dim: int,
        fertilization_numeric_dim: int,
        fertilization_categorical_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        rnn_type: str = 'LSTM',
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type

        # Static feature encoder (MLP)
        static_dim = static_numeric_dim + static_categorical_dim
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

        # Dynamic feature dimensions (including all numerical and categorical features)
        dynamic_dim = (
            dynamic_numeric_dim + fertilization_numeric_dim + fertilization_categorical_dim
        )

        # RNN input dimension = encoded static features + dynamic features
        rnn_input_dim = hidden_size + dynamic_dim

        # RNN layers
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                rnn_input_dim,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                rnn_input_dim,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
        else:
            raise ValueError(f'Unsupported RNN type: {rnn_type}')

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, static_features, dynamic_features, lengths=None):
        """
        Args:
            static_features: (batch, static_dim) - Static features (numerical + categorical encoded)
            dynamic_features: (batch, max_seq_len, dynamic_dim) - Dynamic features
            lengths: (batch,) - Actual length of each sequence

        Returns:
            predictions: (batch, max_seq_len) - Predictions for each step
        """
        _, max_seq_len = dynamic_features.shape[0], dynamic_features.shape[1]

        # Encoding static features
        static_encoded = self.static_encoder(static_features)  # (batch, hidden_size)

        # Copy static features to each time step # (batch, max_seq_len, hidden_size)
        static_encoded = static_encoded.unsqueeze(1).expand(-1, max_seq_len, -1)

        # Concatenate static and dynamic features # (batch, max_seq_len, rnn_input_dim)
        rnn_input = torch.cat([static_encoded, dynamic_features], dim=2)

        # RNN forward propagation (not using pack, because sequence lengths may be different)
        if lengths is not None:
            # Use pack to handle variable length sequences
            packed_input = pack_padded_sequence(
                rnn_input, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.rnn(packed_input)
            rnn_output, _ = pad_packed_sequence(
                packed_output, batch_first=True, total_length=max_seq_len
            )
        else:
            rnn_output, _ = self.rnn(rnn_input)

        # Output layer
        predictions = self.output_layer(rnn_output)
        predictions = predictions.squeeze(-1)

        return predictions



def count_parameters(model):
    """Count number of model parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
