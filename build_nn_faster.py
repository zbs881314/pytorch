import torch
import torch.nn.functional as F

# method1
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
        
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x
        
net1 = Net(n_feature=2, n_hidden=10, n_output=2)
print(net1)

# method2
net2 = torch.nn.Sequential(
    torch.nn.Linear(2, 10), 
    torch.nn.ReLu(), 
    torch.nn.Linear(10, 2) 
    )
    
print(net2)
