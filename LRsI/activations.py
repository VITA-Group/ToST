import torch
from torch import nn
from torch.nn import functional as F

class SwishParameteric(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()

    def forward(self, x, beta = 2):
        return x * torch.sigmoid(beta*x)
    
class GeLU(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return F.gelu(x)
    
class Mish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x*( torch.tanh(F.softplus(x)))

class LiSHT(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x*torch.tanh(x)


class BentID(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + ((torch.sqrt(torch.pow(x, 2) + 1) - 1) / 2)


class SmoothGradReLUnn(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return SmoothGradReLU.apply(x)
    

    
class SmoothGradReLU(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        parameter = 10
        curr_grad = 1/(1+ torch.exp(-parameter*input))
        grad_input = torch.mul(grad_input, curr_grad)
        return grad_input

    def __call__(self, input):
        return self.apply(input)

def SmoothReLU(x, alpha=200):
    x = x.clamp(min=0)
    return x - 1/alpha*torch.log(alpha*x + 1)


relu = nn.ReLU()
# leaky_relu = nn.LeakyReLU(negative_slope=0.2)
swish = lambda x: x*torch.sigmoid(x)
gelu = lambda x: F.gelu(x)
elu = lambda x: F.elu(x)
celu = lambda x: F.celu(x)
softplus_parameteric = lambda x: F.softplus(x, threshold=10)
softplus = lambda x: F.softplus(x, threshold=1)
mish = Mish()
bent_id = BentID()
lisht = lambda x: x*torch.tanh(x)
smoothgradrelu = SmoothGradReLUnn()
sigmoid = lambda x: F.sigmoid(x)
tanh = lambda x: F.tanh(x)


# TODO - Write a test case for smooth grad relu
# TODO - Maybe implement smoothrelu?
def test_smoothgradrelu():
    random_tensor = torch.randn((10000))
    random_tensor.requires_grad_(True)
    smoothgradrelu_out = smoothgradrelu(random_tensor)
    smoothgradrelu_out.sum().backward()
    smoothgradrelu_grad = random_tensor.grad.data
    random_tensor.grad.data.zero_()
    softplus_parameteric_out = softplus_parameteric(random_tensor)
    softplus_parameteric_out.sum().backward()
    softplus_parameteric_grad = random_tensor.grad.data
    assert (smoothgradrelu_grad-softplus_parameteric_grad).abs().sum() < 1e-10

if __name__ == "__main__":
    test_smoothgradrelu()