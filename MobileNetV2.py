import torch
import torch.nn as nn
from torchvision.models import shufflenet_v2_x0_5

# --- Helper Functions ---
def channel_split(x, ratio=0.5):
    c = x.size(1)
    c1 = int(c * ratio)
    return x[:, :c1, :, :], x[:, c1:, :, :]

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    return x.view(batchsize, -1, height, width)

# --- InvertedResidual Block ---
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super().__init__()
        self.stride = stride
        branch_features = oup // 2

        assert self.stride in [1, 2]
        if self.stride == 1:
            assert inp == oup

        if self.stride == 2:
            self.branch1 = nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride=2, padding=1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if self.stride > 1 else branch_features, branch_features, 1, 1, 0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_features, branch_features, 3, stride=self.stride, padding=1, groups=branch_features, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, 1, 1, 0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = channel_split(x)
            out = torch.cat((x1, self.branch2(x2)), 1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), 1)
        return channel_shuffle(out, 2)

# --- Custom ShuffleNetV2 Backbone ---
class ShuffleNetV2Custom(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, 3, 2, 1, bias=False),  # 320 → 160
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(3, 2, 1)  # 160 → 80

        self.stage2 = self._make_stage(24, 32, 3)  # 80 → 40
        self.stage3 = self._make_stage(32, 96, 7)  # 40 → 20
        self.stage4 = self._make_stage(96, 320, 3) # 20 → 10

    def _make_stage(self, inp, oup, repeat):
        layers = [InvertedResidual(inp, oup, 2)]
        for _ in range(repeat - 1):
            layers.append(InvertedResidual(oup, oup, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)         # [1, 24, 160, 160]
        x = self.maxpool(x)       # [1, 24, 80, 80]
        rgb1 = x                  # RGB1

        x = self.stage2(x)        # [1, 32, 40, 40]
        rgb2 = x                  # RGB2

        x = self.stage3(x)        # [1, 96, 20, 20]
        rgb3 = x                  # RGB3

        x = self.stage4(x)        # [1, 320, 10, 10]
        rgb4 = x                  # RGB4

        return rgb1, rgb2, rgb3, rgb4

# --- Load Pretrained Weights ---
def load_custom_shufflenet_weights(custom_model, pretrained_model):
    custom_dict = custom_model.state_dict()
    pretrained_dict = pretrained_model.state_dict()

    matched_dict = {k: v for k, v in pretrained_dict.items() if k in custom_dict and v.shape == custom_dict[k].shape}
    custom_dict.update(matched_dict)
    custom_model.load_state_dict(custom_dict)
    print(f"Loaded {len(matched_dict)} layers from pretrained ShuffleNetV2 0.5x.")


