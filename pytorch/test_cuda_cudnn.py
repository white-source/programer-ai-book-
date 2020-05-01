from __future__ import print_function
import torch
if __name__ == '__main__':
    print("Support CUDA ?:",torch.cuda.is_available())

    x = torch.tensor([10.0])
    x = x.cuda()
    print(x)

    y = torch.randn(2,3)
    y = y.cuda()
    print(y)

    z = x+y
    print(z)

    from torch.backends import cudnn
    print("Support cudnn ?:",cudnn.is_acceptable(x))