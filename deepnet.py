from torch import nn

class deepnet(nn.Module):
    def __init__(self):
        super(deepnet,self).__init__()
        self.seq=nn.Sequential(
            nn.Flatten(),
            nn.Linear(784,89),
            nn.ReLU(),
            nn.Linear(89,10),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.seq(x)