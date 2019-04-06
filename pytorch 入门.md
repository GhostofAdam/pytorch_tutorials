## pytorch 入门

* pytorch

  之前只用过tensorflow，对比来说，pytorch更适合用来做非常dynamic的研究加上对速度要求不高的产品。

  tensorflow版本迭代中出现了模块臃肿，实现复杂的问题，相比之下pytoch简洁的源码，方便动态图结构，封装很好的分布式方法等让我把手头的项目迁移到pytorch。（torchvision真好用）

* python环境

  Anaconda是一个python科学计算非常好用的application，包括了conda的包管理器，虚拟环境管理，jupyter notebook等非常好用的工具

  Anaconda下载

  <https://www.anaconda.com/distribution/>

  Anaconda安装

  *linux*

  ```shell
  bash Anaconda3-5.2.0-Linux-x86_64.sh
  ```

  最后会询问是否添加环境变量，建议选择yes

  *windows*

  按安装指示

* 安装pytorch

  <https://pytorch.org/get-started/locally/>

  ![1554539988483](C:\Users\Administrator.LAPTOP-RPD9KEH6\AppData\Roaming\Typora\typora-user-images\1554539988483.png)

  选择相应的操作系统，python版本，CUDA版本安装

* pytorch基础概念

  *tensor*

  很...普通

  ```python
  tensor=torch.Tensor(5, 3)#直接构造
  tensor=torch.Tensor(5, 3).uniform_(-1, 1)#均匀分布
  tensor=tensor.cuda()#放入GPU
  ```

  *AutoGrad*

  TensorFlow、Caffe 和 CNTK 等大多数框架都是使用的静态计算图，开发者必须建立或定义一个神经网络，并重复使用相同的结构来执行模型训练。改变网络的模式就意味着我们必须从头开始设计并定义相关的模块。

  但 PyTorch 使用的技术为自动微分（automatic differentiation）。在这种机制下，系统会有一个 Recorder 来记录我们执行的运算，然后再反向计算对应的梯度。这种技术在构建神经网络的过程中十分强大，因为我们可以通过计算前向传播过程中参数的微分来节省时间。

  ```python
  from torch.autograd import Variable
  
  x = Variable(train_x)#参数更新#参数更新
  y = Variable(train_y, requires_grad=False)#参数不更新
  ```

  

* 代码框架

  ```python
  def main():
      normalize=None
      trainset=torchvision.datasets.ImageFolder(train_dir,transforms.Compose([
              transforms.Resize(256),
              transforms.CenterCrop(224),
              transforms.ToTensor(),
              normalize,
          ]))
      train_loader = torch.utils.data.DataLoader(
          trainset, batch_size=batch_size*len(device_ids), shuffle=True,
          num_workers=workers, pin_memory=False)
      val_loader = torch.utils.data.DataLoader(
          torchvision.datasets.ImageFolder(val_dir, transforms.Compose([
              transforms.Resize(256),
              transforms.CenterCrop(224),
              transforms.ToTensor(),
              normalize,
          ])))
      model = torchvision.models.resnet152(pretrained=True)
      model = nn.DataParallel(model,device_ids=device_ids)
      model = model.cuda(device=device_ids[0])
      criterion = nn.CrossEntropyLoss()
      optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
  
      for epoch in range(train_epoch):
          adjust_learning_rate(optimizer, epoch, learning_rate=learning_rate)
          train(train_loader, model, criterion, optimizer, epoch)
          acc1 = validate(val_loader, model, criterion, val_gpu)
          is_best = acc1 > best_acc1
          best_acc1 = max(acc1, best_acc1)
          save_checkpoint({
              'epoch': epoch + 1,
              'arch': model_name,
              'state_dict': model.state_dict(),
              'best_acc1': best_acc1,
              'optimizer': optimizer.state_dict(),
          }, is_best)
  def train(train_loader, model, criterion, optimizer, epoch):
  def validation(val_loader, model, criterion, gpu, print_freq):
  ```

  