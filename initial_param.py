#import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gym
import copy

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import optim

def init(module, gain):
    nn.init.orthogonal_(module.weight.data, gain=gain)
    nn.init.constant_(module.bias.data, 0)
    return module

class FlattenLayer(nn.Module):
    def forward(self,x):
        sizes = x.size()
        return x.view(sizes[0],-1)
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        def init_(module): return init(module, gain=nn.init.calculate_gain('relu'))
        self.conv=nn.Sequential(
        init_(nn.Conv2d(2, 32, 8,stride=4)),
        nn.ReLU(),
        init_(nn.Conv2d(32, 64, 4,stride=2)),
        nn.ReLU(),
        init_(nn.Conv2d(64, 64, 3,stride=1)),
        nn.ReLU(),
        FlattenLayer())
        self.shp=self.conv(torch.ones(1,2,100,100)).size()[-1]
        self.ln1=init_(nn.Linear(self.shp,512))
        def init_(module): return init(module, gain=1.0)
        self.critic=init_(nn.Linear(512,1))
        def init_(module): return init(module, gain=0.01)
        self.actor=init_(nn.Linear(512,3))

    def forward(self, x):
        tmp=self.conv(x)
        tmp2=F.relu(self.ln1(tmp))
        critic_output = self.critic(tmp2)
        actor_output = self.actor(tmp2)

        return critic_output, actor_output

net= Net()
net.load_state_dict(torch.load("./model_init.pth"))

std=net.state_dict()
print(std["conv.0.weight"])
print(std.keys())
x=std["conv.0.weight"].numpy()
print(x)
print(x.shape)
print(x.dtype)
print(std["conv.0.bias"].size())

np.savez("./initial_params",
                            std["conv.0.weight"].numpy(),
                            std["conv.0.bias"].numpy(),
                            std["conv.2.weight"].numpy(),
                            std["conv.2.bias"].numpy(),
                            std["conv.4.weight"].numpy(),
                            std["conv.4.bias"].numpy(),
                            std["ln1.weight"].T.numpy(),
                            std["ln1.bias"].T.numpy(),
                            std["critic.weight"].T.numpy(),
                            std["critic.bias"].T.numpy(),
                            std["actor.weight"].T.numpy(),
                            std["actor.bias"].T.numpy())

x=np.load("./initial_params.npz")
print(x["arr_10"].T.shape)
print(np.ones(32).shape)
