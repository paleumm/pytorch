import torch
import torchvision.models as models

model = models.vgg16(pretrained=True)

# save only weight
torch.save(model.state_dict(), "basics/saveLoadModel/model_weights.pth")

# we do not specify pretrained=True, i.e. do not load default weights
model = models.vgg16()

model.load_state_dict(torch.load("basics/saveLoadModel/model_weights.pth"))
model.eval()

# save the strructure too
torch.save(model, "basics/saveLoadModel/model.pth")

model = torch.load("basics/saveLoadModel/model.pth")
