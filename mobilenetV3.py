import torch.nn as nn
import torch.nn.functional as F


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

class H_Sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(H_Sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.
    
class H_Swish(nn.Module):
    def __init__(self, inplace=True):
        super(H_Swish, self).__init__()
        self.inplace = inplace
        self.sigmoid = H_Sigmoid(inplace=inplace)

    def forward(self, x):
        return self.sigmoid(x) * x

class SqueezeExciteBlock(nn.Module):
    def __init__(self, exp_size):
        super(SqueezeExciteBlock, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(exp_size, exp_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(exp_size // 4, exp_size),
            H_Sigmoid()
        )

    def forward(self, x):
        batch, channels, height, width = x.size()
        out = F.avg_pool2d(x, kernel_size=[height, width]).view(batch, -1)
        out = self.dense(out)
        out = out.view(batch, channels, 1, 1)
        return out * x

class BaseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, non_linear, se, exp_size):
        super(BaseBlock, self).__init__()
        self.use_connect = stride == 1 and in_channels == out_channels
        self.se = se

        if non_linear == "RE":
            activation = nn.ReLU
        else:
            activation = H_Swish

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, exp_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(exp_size),
            activation(inplace=True)
        )
        self.depth_conv = nn.Sequential(
            nn.Conv2d(exp_size, exp_size, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, groups=exp_size),
            nn.BatchNorm2d(exp_size),
        )

        if self.se:
            self.squeeze_excite_block = SqueezeExciteBlock(exp_size)

        self.point_conv = nn.Sequential(
            nn.Conv2d(exp_size, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.depth_conv(out)

        if self.se:
            out = self.squeeze_excite_block(out)

        out = self.point_conv(out)

        if self.use_connect:
            return x + out
        else:
            return out
    
def divide_8(value):
    new_value = int(value + 8 / 2) // 8 * 8
    if new_value < 0.9 * value:
        new_value += 8
    return new_value        

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        m.bias.data.zero_()

class MobileNetV3Model(nn.Module):
    def __init__(self, model_mode="large", num_classes=1000, multiplier=1.0, dropout_rate=0.0):
        super(MobileNetV3Model, self).__init__()
        self.num_classes = num_classes

        if model_mode == "large":
            layers = [
                [16, 16, 3, 1, "RE", False, 16],
                [16, 24, 3, 2, "RE", False, 64],
                [24, 24, 3, 1, "RE", False, 72],
                [24, 40, 5, 2, "RE", True, 72],
                [40, 40, 5, 1, "RE", True, 120],

                [40, 40, 5, 1, "RE", True, 120],
                [40, 80, 3, 2, "HS", False, 240],
                [80, 80, 3, 1, "HS", False, 200],
                [80, 80, 3, 1, "HS", False, 184],
                [80, 80, 3, 1, "HS", False, 184],

                [80, 112, 3, 1, "HS", True, 480],
                [112, 112, 3, 1, "HS", True, 672],
                [112, 160, 5, 1, "HS", True, 672],
                [160, 160, 5, 2, "HS", True, 672],
                [160, 160, 5, 1, "HS", True, 960],
            ]

            init_conv_out = divide_8(16 * multiplier)
            self.init_conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=init_conv_out, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(init_conv_out),
                H_Swish(inplace=True),
            )

            self.block = nn.Sequential(
                *[BaseBlock(*params) for params in layers]
            )
            out_conv1_in = divide_8(160 * multiplier)
            out_conv1_out = divide_8(960 * multiplier)
            self.out_conv1 = nn.Sequential(
                nn.Conv2d(out_conv1_in, out_conv1_out, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_conv1_out),
                H_Swish(inplace=True),
            )

            out_conv2_in = divide_8(960 * multiplier)
            out_conv2_out = divide_8(1280 * multiplier)
            self.out_conv2 = nn.Sequential(
                nn.Conv2d(out_conv2_in, out_conv2_out, kernel_size=1, stride=1),
                H_Swish(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Conv2d(out_conv2_out, self.num_classes, kernel_size=1, stride=1),
            )

        elif model_mode == "small":
            layers = [
                [16, 16, 3, 2, "RE", True, 16],
                [16, 24, 3, 2, "RE", False, 72],
                [24, 24, 3, 1, "RE", False, 88],
                [24, 40, 5, 2, "RE", True, 96],
                [40, 40, 5, 1, "RE", True, 240],
                [40, 40, 5, 1, "RE", True, 240],
                [40, 48, 5, 1, "HS", True, 120],
                [48, 48, 5, 1, "HS", True, 144],
                [48, 96, 5, 2, "HS", True, 288],
                [96, 96, 5, 1, "HS", True, 576],
                [96, 96, 5, 1, "HS", True, 576],
            ]

            init_conv_out = divide_8(16 * multiplier)
            self.init_conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=init_conv_out, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(init_conv_out),
                H_Swish(inplace=True),
            )

            self.block = nn.Sequential(
                *[BaseBlock(*params) for params in layers]
            )

            out_conv1_in = divide_8(96 * multiplier)
            out_conv1_out = divide_8(576 * multiplier)

            self.out_conv1 = nn.Sequential(
                nn.Conv2d(out_conv1_in, out_conv1_out, kernel_size=1, stride=1),
                SqueezeExciteBlock(out_conv1_out),
                nn.BatchNorm2d(out_conv1_out),
                H_Swish(inplace=True),
            )

            out_conv2_in = divide_8(576 * multiplier)
            out_conv2_out = divide_8(1280 * multiplier)
            self.out_conv2 = nn.Sequential(
                nn.Conv2d(out_conv2_in, out_conv2_out, kernel_size=1, stride=1),
                H_Swish(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Conv2d(out_conv2_out, self.num_classes, kernel_size=1, stride=1),
            )

        self.apply(init_weights)

    def forward(self, x):
        out_conv = self.init_conv(x)
        out_conv = self.block(out_conv)
        out_conv = self.out_conv1(out_conv)
        batch, channels, height, width = out_conv.size()
        out_conv = F.avg_pool2d(out_conv, kernel_size=[height, width])
        out_conv = self.out_conv2(out_conv).view(batch, -1)
        return out_conv
