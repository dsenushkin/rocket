import torch
import rocket
import torch.nn.functional as F
import numpy as np
from torch import nn
from accelerate import Accelerator


import torchvision
from torchvision.datasets import MNIST

PATH = "./data"
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
mnist = MNIST(f"{PATH}/mnist", 
              download=True, 
              transform=transform, 
              target_transform=lambda x: torch.tensor(x))


class Accuracy(rocket.Metric):
    def __init__(self, accelerator: Accelerator = None, priority: int = 1000) -> None:
        super().__init__(accelerator, priority)
        self.positive = 0.0
        self.total = 0.0

    def launch(self, attrs: Attributes = None):
        batch = attrs.batch
        gt, pr = batch[1], batch[2]
        pr = pr.argmax(1)

        self.total += pr.numel()
        self.positive += torch.sum(gt == pr)
    
        attrs.looper.state.accuracy = (self.positive / self.total).item()


    def reset(self, attrs: Attributes = None):
        self.total = 0.0
        self.positive = 0.0
        
    
class LeNet(nn.Module):

    # network structure
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        '''
        One forward pass through the network.
        
        Args:
            x: input
        '''
        inp = x
        x = F.max_pool2d(F.relu(self.conv1(x[0])), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return inp[0], inp[1], x
        
    def num_flat_features(self, x):
        '''
        Get the number of features in a batch of tensors `x`.
        '''
        size = x.size()[1:]
        return np.prod(size)

net = LeNet()
opt =  torch.optim.AdamW(net.parameters())
sched = torch.optim.lr_scheduler.StepLR(opt, 100)


class CrossEntropy(torch.nn.Module):
    def forward(self, batch):
        return torch.nn.CrossEntropyLoss()(batch[2], batch[1])
loss = CrossEntropy()


accelerator = Accelerator(device_placement=False, gradient_accumulation_steps=2)

launcher = rocket.Launcher([
        rocket.Looper([
            rocket.Dataset(mnist, batch_size=1024),
            rocket.Module(net, capsules=[
                rocket.Loss(objective=loss),
                rocket.Optimizer(opt),
                rocket.Scheduler(sched)
            ]),
            rocket.Checkpointer(output_dir="./chkpt/",
                                overwrite=True,
                                save_every=50)
            
        ]),
    ],
    statefull=False,
    num_epochs=4,
    accelerator=accelerator
)
print(launcher)