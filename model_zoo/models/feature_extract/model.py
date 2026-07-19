import torch.nn as nn


class FeatureExtract(nn.Module):
    def __init__(self, ndims=2):
        super(FeatureExtract, self).__init__()
        assert ndims in [2, 3], "ndims should be 2 or 3. found: %d" % ndims
        Conv = getattr(nn, "Conv%dd" % ndims)

        self.layer1 = nn.Sequential(
            Conv(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            Conv(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
        )

        self.layer2 = nn.Sequential(
            Conv(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=False),
            Conv(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
        )

        self.layer3 = nn.Sequential(
            Conv(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=False),
            Conv(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        out = self.layer3(layer2)
        return [layer1, layer2, out]
