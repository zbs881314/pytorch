import torch
from torch.autograd import Variable
from torch.nn import functional as F


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(128, 32)
        self.linear2 = torch.nn.Linear(32, 16)
        self.linear3 = torch.nn.Linear(16, 2)


    def forward(self, x):
        layer1_out = F.relu(self.linear1(x))
        layer2_out = F.relu(self.linear2(layer1_out))
        out = self.linear3(layer2_out)
        return out, layer1_out, layer2_out


def l1_penalty(var):
    return torch.abs(var).sum()

def l2_penalty(var):
    return torch.sqrt(torch.pow(var, 2).sum())

batchsize = 4
lambda1, lambda2 = 0.5, 0.01

model = MLP()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

inputs = Variable(torch.rand(batchsize, 128))
targets = Variable(torch.ones(batchsize).long())

optimizer.zero_grad()
outputs, layer1_out, layer2_out = model(inputs)
cross_entropy_loss = F.cross_entropy(outputs, targets)
l1_regularization = lambda1 * l1_penalty(layer1_out)
l2_regularization = lambda2 * l2_penalty(layer2_out)

# lambda1 = torch.tensor(0.5)
# lambda2 = torch.tensor(0.01)

# l1_regularization, l2_regularization = torch.tensor(0), torch.tensor(0)
# for param in model.parameters():
#     l1_regularization += torch.norm(param, 1)
#     l2_regularization += torch.norm(param, 2)

loss = cross_entropy_loss + l1_regularization + l2_regularization
loss.backward()
optimizer.step()



