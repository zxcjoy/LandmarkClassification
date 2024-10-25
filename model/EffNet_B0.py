import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class EffNet_B0(nn.Module):
    def __init__(
        self, coarse_classes_num, fine_classes_num, pretrained=True, dropout=0
    ):
        super(EffNet_B0, self).__init__()

        if pretrained:
            self.model = EfficientNet.from_pretrained("efficientnet-b0")
        else:
            self.model = EfficientNet.from_name("efficientnet-b0")

        last_channels = self.model._blocks_args[-1].output_filters * 4

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.coarse_head = nn.Sequential(
            nn.Linear(last_channels, 128),
            nn.Dropout(dropout),
            nn.Linear(128, coarse_classes_num),
        )

        self.fine_head = nn.Sequential(
            nn.Linear(last_channels+128, 128),
            nn.Dropout(dropout),
            nn.Linear(128, fine_classes_num),
        )

    def forward(self, x):
        self.model.eval()
        with torch.no_grad():
            x = self.model.extract_features(x)
            x = self.pool(x)
        hidden = self.coarse_head[0](x)
        coarse = self.coarse_head(x)
        x = torch.concat([x,hidden], dim = -1)
        fine = self.fine_head(x)
        return coarse, fine


if __name__ == "__main__":
    _input = torch.randn(size=(2, 3, 256, 256))
    model = EffNet_B0(5, 10, pretrained=True, dropout=0)
    output = model(_input)
    print(output[0].shape)
    print(output[1].shape)
    print(nn.Softmax(1)(output[0]))
    print(nn.Softmax(1)(output[1]))
