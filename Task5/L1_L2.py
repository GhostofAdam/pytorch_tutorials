import torch
import torchvision

model=torchvision.models.vgg16()

def Regularization(model):
    L1=0
    L2=0
    for param in model.parameters():
        L1+=torch.sum(torch.abs(param))
        L2+=torch.norm(param,2)
    return L1,L2
mnist=torchvision.datasets.mnist
#for epoch in range(100):
#train(model, device, train_loader, optimizer, epoch)
#test(model, device, test_loader)