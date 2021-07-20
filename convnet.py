from torch import nn

class convnet(nn.Module):
    def __init__(self):
        super(convnet,self).__init__()
        self.seq=nn.Sequential(
            nn.Conv2d(1,6,kernel_size=9,stride=4,padding=1),
            nn.ReLU(),
            nn.Conv2d(6,43,kernel_size=6,stride=1),
            nn.ReLU(),
            nn.Flatten(1,3),
            nn.Linear(43,10),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.seq(x)