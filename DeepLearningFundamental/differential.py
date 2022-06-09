from shutil import get_archive_formats
import torch

a = torch.tensor([2.0, 3.0], requires_grad=True)
b = torch.tensor([6.0, 4.0], requires_grad=True)

Q = 3 * a ** 3 - b ** 2

external_grad = torch.tensor([1.0, 1.0])
Q.backward(gradient=external_grad)

# check if collected gradients are correct
print(9 * a ** 2 == a.grad)
print(-2 * b == b.grad)
