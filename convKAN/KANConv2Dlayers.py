import torch.nn as nn
import torch.nn.functional as F
from KANlayers import *

class KANConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(KANConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kanlayer = KANLinear(in_channels * kernel_size * kernel_size, out_channels)

    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        assert in_channels == self.in_channels

        # Apply unfold to get sliding local blocks
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        x_unfold = x_unfold.transpose(1, 2)
        x_unfold = x_unfold.reshape(batch_size * x_unfold.size(1), -1)

        out_unfold = self.kanlayer(x_unfold)

        # Reshape and transpose to get the final output
        out_unfold = out_unfold.reshape(batch_size, -1, out_unfold.size(1))
        out = out_unfold.transpose(1, 2)
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out.reshape(batch_size, self.out_channels, out_height, out_width)

        return out
    

class ChebyKANConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, degree=4):
        super(ChebyKANConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kanlayer = ChebyKANLayer(in_channels * kernel_size * kernel_size, out_channels, degree=degree)

    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        assert in_channels == self.in_channels

        # Apply unfold to get sliding local blocks
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        x_unfold = x_unfold.transpose(1, 2)
        x_unfold = x_unfold.reshape(batch_size * x_unfold.size(1), -1)

        out_unfold = self.kanlayer(x_unfold)

        # Reshape and transpose to get the final output
        out_unfold = out_unfold.reshape(batch_size, -1, out_unfold.size(1))
        out = out_unfold.transpose(1, 2)
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out.reshape(batch_size, self.out_channels, out_height, out_width)

        return out
    

class FastKANConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(FastKANConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kanlayer = FastKANLayer(in_channels * kernel_size * kernel_size, out_channels)

    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        assert in_channels == self.in_channels

        # Apply unfold to get sliding local blocks
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        x_unfold = x_unfold.transpose(1, 2)
        x_unfold = x_unfold.reshape(batch_size * x_unfold.size(1), -1)

        out_unfold = self.kanlayer(x_unfold)

        # Reshape and transpose to get the final output
        out_unfold = out_unfold.reshape(batch_size, -1, out_unfold.size(1))
        out = out_unfold.transpose(1, 2)
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out.reshape(batch_size, self.out_channels, out_height, out_width)

        return out
    


class GRAMKANConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(GRAMKANConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kanlayer = GRAMLayer(in_channels * kernel_size * kernel_size, out_channels)

    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        assert in_channels == self.in_channels

        # Apply unfold to get sliding local blocks
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        x_unfold = x_unfold.transpose(1, 2)
        x_unfold = x_unfold.reshape(batch_size * x_unfold.size(1), -1)

        out_unfold = self.kanlayer(x_unfold)

        # Reshape and transpose to get the final output
        out_unfold = out_unfold.reshape(batch_size, -1, out_unfold.size(1))
        out = out_unfold.transpose(1, 2)
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out.reshape(batch_size, self.out_channels, out_height, out_width)

        return out
    



class WavKANConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, wavelet_type='mexican_hat'):
        super(WavKANConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kanlayer = WavKANLayer(in_channels * kernel_size * kernel_size, out_channels, wavelet_type=wavelet_type)

    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        assert in_channels == self.in_channels

        # Apply unfold to get sliding local blocks
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        x_unfold = x_unfold.transpose(1, 2)
        x_unfold = x_unfold.reshape(batch_size * x_unfold.size(1), -1)

        out_unfold = self.kanlayer(x_unfold)

        # Reshape and transpose to get the final output
        out_unfold = out_unfold.reshape(batch_size, -1, out_unfold.size(1))
        out = out_unfold.transpose(1, 2)
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out.reshape(batch_size, self.out_channels, out_height, out_width)

        return out
    


class JacobiKANConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,degree=4):
        super(JacobiKANConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kanlayer = JacobiKANLayer(in_channels * kernel_size * kernel_size, out_channels, degree=degree)

    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        assert in_channels == self.in_channels

        # Apply unfold to get sliding local blocks
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        x_unfold = x_unfold.transpose(1, 2)
        x_unfold = x_unfold.reshape(batch_size * x_unfold.size(1), -1)

        out_unfold = self.kanlayer(x_unfold)

        # Reshape and transpose to get the final output
        out_unfold = out_unfold.reshape(batch_size, -1, out_unfold.size(1))
        out = out_unfold.transpose(1, 2)
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out.reshape(batch_size, self.out_channels, out_height, out_width)

        return out
    

    

class ReLUKANConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ReLUKANConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kanlayer = ReLUKANLayer(in_channels * kernel_size * kernel_size, 5, 3, out_channels)

    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        assert in_channels == self.in_channels

        # Apply unfold to get sliding local blocks
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        x_unfold = x_unfold.transpose(1, 2)
        x_unfold = x_unfold.reshape(batch_size * x_unfold.size(1), -1)

        out_unfold = self.kanlayer(x_unfold)

        # Reshape and transpose to get the final output
        out_unfold = out_unfold.reshape(batch_size, -1, out_unfold.size(1))
        out = out_unfold.transpose(1, 2)
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out.reshape(batch_size, self.out_channels, out_height, out_width)

        return out
    

    

class FasterKANConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(FasterKANConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kanlayer = FasterKANLayer(in_channels * kernel_size * kernel_size, out_channels)

    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        assert in_channels == self.in_channels

        # Apply unfold to get sliding local blocks
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        x_unfold = x_unfold.transpose(1, 2)
        x_unfold = x_unfold.reshape(batch_size * x_unfold.size(1), -1)

        out_unfold = self.kanlayer(x_unfold)

        # Reshape and transpose to get the final output
        out_unfold = out_unfold.reshape(batch_size, -1, out_unfold.size(1))
        out = out_unfold.transpose(1, 2)
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out.reshape(batch_size, self.out_channels, out_height, out_width)

        return out
    


class RBFKANConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(RBFKANConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kanlayer = RBFKANLayer(in_channels * kernel_size * kernel_size, out_channels)

    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        assert in_channels == self.in_channels

        # Apply unfold to get sliding local blocks
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        x_unfold = x_unfold.transpose(1, 2)
        x_unfold = x_unfold.reshape(batch_size * x_unfold.size(1), -1)

        out_unfold = self.kanlayer(x_unfold)

        # Reshape and transpose to get the final output
        out_unfold = out_unfold.reshape(batch_size, -1, out_unfold.size(1))
        out = out_unfold.transpose(1, 2)
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out.reshape(batch_size, self.out_channels, out_height, out_width)

        return out