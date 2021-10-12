import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, input_size, output_size, p, hidden_sizes):
        super().__init__()
        assert len(hidden_sizes) > 0
        layers = []

        layer1 = nn.Linear(input_size, hidden_sizes[0], bias=True)
        bn1 = nn.BatchNorm1d(hidden_sizes[0])
        dropout1 = nn.Dropout(p)

        last_layer = nn.Linear(hidden_sizes[-1], output_size, bias=True)

        layers.append(layer1)
        layers.append(bn1)
        layers.append(dropout1)
        layers.append(nn.ReLU())

        if len(hidden_sizes) > 1:
            for i in range(len(hidden_sizes) - 1):

                hidden_layer = nn.Linear(hidden_sizes[i], hidden_sizes[i + 1], bias=True)
                bn = nn.BatchNorm1d(hidden_sizes[i + 1])
                dropout = nn.Dropout(p)

                layers.append(hidden_layer)
                layers.append(bn)
                layers.append(dropout)
                layers.append(nn.ReLU())
        layers.append(last_layer)
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)
