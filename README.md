# KANConv2d
2D Convolutional KAN Layers with different types of activation functions.

## Overview
KANConv2D provides implementations of various Kolmogorov-Arnold Network (KAN) layers in Conv2D format, utilizing different types of activation functions.

## Available Layers
- **KANConv2d**: Convolutional version of the original B-Spline based KAN.
- **FastKANConv2d**: Uses Radial Basis Functions (RBF) as the activation function, a faster variant of `KANConv2D`.
- **FasterKANConv2d**: Employs the Reflectional Switch Activation Function (RSWAF).
- **ChebyKANConv2d**: Utilizes Chebyshev polynomials as the activation function.
- **GRAMKANConv2d**: Implements Gram polynomials as the activation function.
- **WavKANConv2d**: Uses Wavelet transforms as the activation function.
- **JacobiKANConv2d**: Employs Jacobi polynomials as the activation function.
- **ReLUKANConv2d**: Incorporates a modified version of ReLU as the activation function.
- **RBFKANConv2d**: Another implementation utilizing Radial Basis Functions (RBF) as the activation function.

## Usage
In principle, you can simply replace `nn.Conv2d` in your code with `KANConv2d` or any other activation function-based `KANConv2d` to integrate KAN convolution layers into your models.

```python
import torch
import torch.nn as nn
from KANConv2Dlayers import *

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = KANConv2d(1, 10, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = KANConv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(20 * 4 * 4, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.reshape(-1, 20 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```
### Notes:

- **No additional activation function is needed**: External activation functions like ReLU are not necessary after `KANConv2d`.

- **Polynomial degree in ChebyKANConv2d and JacobiKANConv2d**: For `ChebyKANConv2d` and `JacobiKANConv2d`, you can control the degree of the polynomial used in the activation by setting the `degree` parameter (default is 4). For example:

  ```python
  conv = ChebyKANConv2d(1, 10, kernel_size=5, degree=3)
  ```

- **Wavelet type in WavKANConv2d**: The type of wavelet used in `WavKANConv2d` can be adjusted via the wavelet_type parameter (default is `'mexican_hat'`). The available types include `'mexican_hat'`, `'morlet'`, `'dog'`, `'meyer'`, and `'shannon'`. For example:
  ```python
  conv = WavKANConv2d(1, 10, kernel_size=5, wavelet_type='dog')
  ```
With these flexible configurations, you can fine-tune the behavior of each `KANConv2d` variant to suit your needs.

## Acknowledgement
We would like to thank the open-source community for their invaluable contributions and support. Special thanks to researchers and developers who have explored various activation functions in Kolmogorov-Arnold Networks (KAN), including:
- The original implementation of KAN: [pyKAN](https://github.com/KindXiaoming/pykan)
- An Efficient Implementation of KAN: [Efficient-KAN](https://github.com/Blealtan/efficient-kan)
- [FastKAN](https://github.com/ZiyaoLi/fast-kan)
- [FasterKAN](https://github.com/AthanasiosDelis/faster-kan)
- [ChebyKAN](https://github.com/SynodicMonth/ChebyKAN)
- [GRAMKAN](https://github.com/Khochawongwat/GRAMKAN)
- [WavKAN](https://github.com/zavareh1/Wav-KAN)
- [JacobiKAN](https://github.com/SpaceLearner/JacobiKAN)
- [ReLUKAN](https://github.com/quiqi/relu_kan)
- [RBFKAN](https://github.com/Sid2690/RBF-KAN)

## Citation

If you find this work useful, please cite it as:

```bibtex
@article{yang2024activation,
  title={Activation Space Selectable Kolmogorov-Arnold Networks},
  author={Yang, Zhuoqin and Zhang, Jiansong and Luo, Xiaoling and Lu, Zheng and Shen, Linlin},
  journal={arXiv preprint arXiv:2408.08338},
  year={2024}
}
