import torch
from torch.autograd import Variable
x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.L1=torch.nn.Linear(1,1)
    def forward(self,x):
        return self.L1(x)

net=Net()
criterion=torch.nn.MSELoss(size_average=False)
opt=torch.optim.SGD(net.parameters(),lr=0.01)
for epoch in range(10):
    y_pred=net(x_data)
    loss=criterion(y_pred,y_data)
    print(epoch,loss.data)
    opt.zero_grad()
    loss.backward()
    opt.step()

hour_var = Variable(torch.Tensor([[4.0]]))
print("predict (after training)", 4, net.forward(hour_var).data[0][0])
