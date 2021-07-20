import torch
from torch import nn,optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from myoptim import myoptim
from deepnet import deepnet
from convnet import convnet

batch_size=1024
epochs=200
netname="conv"    #deep, conv
optimname="sgd"    #sgd, myoptim
lr=10

filename=netname+"+"+optimname+"+"+"lr="+str(lr)+".txt"
f=open(filename,"w")
train_data=datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)
test_data=datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)
train_dataloader=DataLoader(train_data,batch_size=batch_size)
test_dataloader=DataLoader(test_data,batch_size=batch_size)
batchnum=int((len(train_dataloader.dataset)+batch_size-1)/batch_size)
if netname=="deep":
    net=deepnet()
elif netname=="conv":
    net=convnet()
else:
    print("net not supported")
    sys.exit()
loss_f=nn.CrossEntropyLoss()
if optimname=="sgd":
    optimizer=optim.SGD(params=net.parameters(),lr=lr)
elif optimname=="myoptim":
    optimizer=myoptim(net,bnum=batchnum,lr=lr)
else:
    print("optimizer not supported")
    sys.exit()
bestacc=0
for i in range(epochs):
    print(f"Epoch {i+1}:")
    for batch,(x,y) in enumerate(train_dataloader):
        h=net(x)
        loss=loss_f(h,y)
        optimizer.zero_grad()
        loss.backward()
        if optimname=="sgd":
            optimizer.step()
        else:
            optimizer.step(batch)
    size=len(test_dataloader.dataset)
    net.eval()
    test_loss,correct=0,0
    with torch.no_grad():
        for x,y in test_dataloader:
            h=net(x)
            test_loss+=loss_f(h,y).item()
            correct+=(h.argmax(1)==y).type(torch.float).sum().item()
    test_loss/=((size+batch_size-1)/batch_size)
    correct/=size
    if bestacc<correct:
        bestacc=correct
        bestloss=test_loss
    print(f"Accuracy={(100*correct):>f}%({(100*bestacc):>f}%), Avg loss={test_loss:>8f}\n")
    f.write("{:d} {:f} {:f}\n".format(i,bestacc,bestloss))