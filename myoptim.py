import math
import numpy as np
from numpy import linalg
import torch
from torch import tensor
import copy

class pgmake():
    def __init__(self,net,zero=False,paramsize:int=0):
        super(pgmake,self).__init__()
        if zero:
            self.p=np.zeros(paramsize)
            self.g=np.zeros(paramsize)
        else:
            self.p=np.array([])
            self.g=np.array([])
            self.__init_sub__(net.children())
    def __init_sub__(self,layer):
        for sublayer in layer:
            self.__init_sub__(sublayer.children())
            for _,param in sublayer._parameters.items():
                if param is None:
                    continue
                if self.p.size==0:
                    self.p=np.array(param.data).flatten()
                    self.g=np.array(param.grad.data).flatten()
                else:
                    self.p=np.hstack((self.p,np.array(param.data).flatten()))
                    self.g=np.hstack((self.g,np.array(param.grad.data).flatten()))

class myoptim():
    def __init__(self,net,bnum:int=1,N:int=5,lr=0.01):
        super(myoptim,self).__init__()
        self.net=net
        self.paramsize=self.__paramcount__()
        self.N=N
        self.nmap=np.zeros(bnum,dtype=int)
        self.P1=0.25
        self.P2=0.9
        self.mask=-math.log(self.P2,2)/(N-2)
        self.lr=lr
        self.eps=1e-8
        pgzero=pgmake(self.net,True,self.paramsize)
        pglist={}
        for i in range(0,N):
            pglist[i]=copy.deepcopy(pgzero)
        self.pgmap={}
        for i in range(0,bnum):
            self.pgmap[i]=copy.deepcopy(pglist)
    def __f1__(self,x):
        return self.P1**linalg.norm(x)
    def __paramcount__(self):
        flag=0
        with torch.no_grad():
            flag=self.__paramcount_sub__(self.net.children(),flag)
        return flag
    def __paramcount_sub__(self,layer,flag):
        with torch.no_grad():
            for sublayer in layer:
                flag=self.__paramcount_sub__(sublayer.children(),flag)
                for _,param in sublayer._parameters.items():
                    if param is None:
                        continue
                    size=np.array(param).size
                    flag+=size
        return flag
    def __update__(self,new_param):
        with torch.no_grad():
            self.__update_sub__(self.net.children(),new_param,0)
    def __update_sub__(self,layer,new_param,flag):
        with torch.no_grad():
            for sublayer in layer:
                flag=self.__update_sub__(sublayer.children(),new_param,flag)
                for _,param in sublayer._parameters.items():
                    if param is None:
                        continue
                    size=np.array(param).size
                    param.set_(tensor(new_param[flag:flag+size].reshape(param.shape)))
                    flag=flag+size
        return flag
    def zero_grad(self):
        for _,param in self.net.named_parameters():
            if param.grad is not None:
                param.grad.data*=0
    def step(self,bid):
        pg=pgmake(self.net)
        pglist=self.pgmap[bid]
        n=self.nmap[bid]
        ########插入最新的pg########
        if n<self.N:
            n+=1
        for i in range(n-2,-1,-1):
            pglist[i+1]=pglist[i]
        pglist[0]=pg
        self.pgmap[bid]=pglist
        self.nmap[bid]=n
        if n==1:
            self.__update__(pg.p-pg.g*self.lr)    
            return
        ########计算a1########
        a1=np.zeros([n,n])
        for i in range(0,n):
            for j in range(i+1,n):
                dg=pglist[j].g-pglist[i].g
                dp=pglist[j].p-pglist[i].p
                if linalg.norm(dg)<self.eps:
                    a1[i,j]=self.lr
                else:
                    a1[i,j]=linalg.norm(dp)/linalg.norm(dg)
                a1[j,i]=a1[i,j]
        ########计算a2########
        a2=np.zeros(n)
        for i in range(0,n):
            for j in range(0,n):
                if i==j:
                    continue
                a2[i]+=self.__f1__(pglist[i].p-pglist[j].p)*a1[i,j]
        ########计算a3########
        a3=0
        for i in range(0,n):
            mulf=1
            for j in range(0,n):
                dij=pglist[i].p-pglist[j].p
                di0=pglist[i].p-pglist[0].p
                dj0=pglist[j].p-pglist[0].p
                lij=linalg.norm(dij)
                li0=linalg.norm(di0)
                lj0=linalg.norm(dj0)
                if lij<self.eps:
                    continue
                if li0<self.eps:
                    continue
                if lj0<self.eps:
                    continue
                if 2*lij*lj0<self.eps:
                    mulf*=(1/2)**self.mask
                else:
                    if (1+(lij**2+lj0**2-li0**2)/(2*lij*lj0))/2>0:
                        mulf*=((1+(lij**2+lj0**2-li0**2)/(2*lij*lj0))/2)**self.mask
                    else:
                        mulf=0
            a3+=self.__f1__(pglist[i].p-pglist[0].p)*mulf*a2[i]
        a3/=n*(n-1)*self.P2
        param_next=pg.p-pg.g*a3
        self.__update__(param_next)