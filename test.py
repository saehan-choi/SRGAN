import torch
import torchvision

# k = torch.randn((1,3,244,244))

# m = torch.ones(k)

# print(m.size())

model = torchvision.models.vgg19_bn(pretrained=True)

print(model.features[:7])