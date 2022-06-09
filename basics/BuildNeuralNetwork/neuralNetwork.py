import os
from black import out
import torch
from torch import nn
from torch.utils.data import dataloader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_prob = nn.Softmax(dim=1)(logits)
y_pred = pred_prob.argmax(1)
print(f"Predicted Class: {y_pred}")

# illustration

# input 3 28*28-size image
input_image = torch.rand(3, 28, 28)
print(input_image.size())

# turns into 3 784-size imgae
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())


# nn.Linear
# The linear layer is a module that applies a linear transformation
# on the input using its stored weights and biases.
layer1 = nn.Linear(in_features=28 * 28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

# nn.ReLU
# In this model, we use nn.ReLU between our linear layers, but there’s
# other activations to introduce non-linearity in your model.
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

# nn.Sequential
# nn.Sequential is an ordered container of modules. The data is passed
# through all the modules in the same order as defined. You can use
# sequential containers to put together a quick network like seq_modules.
seq_modules = nn.Sequential(flatten, layer1, nn.ReLU(), nn.Linear(20, 10))

input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)

# nn.Softmax
# he logits are scaled to values [0, 1] representing the model’s predicted
# probabilities for each class. 'dim' parameter indicates the dimension along
# which the values must sum to 1
softmax = nn.Softmax(dim=1)
pred_prob = softmax(logits)

# Model parameters
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
