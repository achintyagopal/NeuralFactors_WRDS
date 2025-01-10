import torch.nn as nn
import torch.nn.functional as F


class ResidBlock(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        activation="SiLU",
        dropout=0.0,
        use_bias=True,
        elementwise_affine=False,
    ):
        super().__init__()
        if isinstance(activation, str):
            activation = getattr(nn, activation)()
        else:
            activation = activation
        self.layers = nn.Sequential(
            nn.LayerNorm(input_size, elementwise_affine=elementwise_affine),
            nn.Linear(input_size, hidden_size, bias=use_bias),
            nn.Dropout(dropout),
            activation,
            nn.Linear(hidden_size, input_size, bias=use_bias),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.layers(x)


class RNNLayer(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        dropout=0,
        use_bias=True,
        rnn_type="GRU",
        elementwise_affine=False,
    ):
        super(RNNLayer, self).__init__()
        self.ln = nn.LayerNorm(input_size, elementwise_affine=elementwise_affine)

        if rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size,
                hidden_size,
                num_layers=1,
                bias=use_bias,
                batch_first=True,
                dropout=dropout,
                bidirectional=False,
            )
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size,
                hidden_size,
                num_layers=1,
                bias=use_bias,
                batch_first=True,
                dropout=dropout,
                bidirectional=False,
            )
        else:
            raise ValueError(f"Unrecognized {rnn_type=}")
        self.out_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        h, _ = self.rnn(self.ln(x))
        return x + self.out_layer(h)


class RNNModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_blocks=2,
        dropout=0,
        activation="SiLU",
        use_bias=True,
        elementwise_affine=False,
        rnn_type="GRU",
    ):
        super(RNNModel, self).__init__()
        layers = []
        for i in range(num_blocks):
            layers.append(
                RNNLayer(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    use_bias=use_bias,
                    elementwise_affine=elementwise_affine,
                    rnn_type=rnn_type,
                )
            )
            layers.append(
                ResidBlock(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    use_bias=use_bias,
                    elementwise_affine=elementwise_affine,
                    activation=activation,
                )
            )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
