# Intro:
1- Tensors:             [numpy array]
    Tensor = matriu = torch.tensor([1, 2, 3], [2, 4, 5])

2 - GPU acceleration
    device = "cuda"
    tensor = tensor.to(device)
    x = torch.tensor(blabla).to(device)
    y = model(x)
    - Remind all "to device"

3 - Autograd
4 - Modular (nn.Module)
5 - Predefined layers & optimizers
6 - Dataloader
7 - load/save
8 - train/test loop

## Cost function
Co = cross entropy
Co = (yp-yr)^2 + minW + B~=0

### Entropy
TMDyn = mesura quantitat de desordre
H(x) = -Sumatorio(xEx) . p(x) . log(base 2)(p(x))
log(base 2) => bits información

#### Example:
x = 80%
O = 20%

H(---)=-( 0,2 log(base 2)(0,2) + 0,8 log(base 2)(0,8))
    = 0,46 + 0,26 = 0,72

## Cost function
Co = - Sumatorio de p(y) log(p(y))
softmax = e^x / Sumatorio de e^x

SCo / SYpred = (SCen / Ssoftmax) . (Ssoftmax / SYpred)
error = SCo = -2(yp - yr)
