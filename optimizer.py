import torch
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt

LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1*torch.normal(torch.zeros(*x.size()))


# plt.scatter(x.numpy(), y.numpy())
# plt.show()

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)
        self.predict = torch.nn.Linear(20, 1)
        
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x
        
if __name__ == '__main__':
    net_SGD = Net()
    net_Momentum = Net()
    net_RMSprop = Net()
    net_Adam = Net()
    nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]
    
    # different optimizers
    opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
    opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR, momentum=0.8)
    opt_SGD = torch.optim.RMSprop(net_SGD.parameters(), lr=LR, alpha=0.9)
    opt_SGD = torch.optim.Adam(net_SGD.parameters(), lr=LR, betas=(0.9, 0.99))
    

    loss_func = torch.nn.MSELoss()
    losses_his = [[], [], [], []]   #recode loss
