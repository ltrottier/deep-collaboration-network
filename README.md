Deep Collaboration Network
============================

This is an implementation of Deep Collaboration Network (DCNet) in `pytorch` from Multi-Task Learning by Deep Collaboration and Application in Facial Landmark Detection (arXiv link available soon) by Trottier, et al. (2017).

## Requirements

1. [pytorch](http://pytorch.org/)
2. torchvision==0.1.8

## Example

Here is an example on how to use `DCNet` and train it with `MultiTaskCriterion`:

```python
from dcnet import DeepCollaborationNetwork, MultiTaskCriterion
from torch.autograd import Variable
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

# define (input, targets) pair
dims = [10, 5]
bs = 1
input = Variable(torch.randn(bs, 3, 112, 112))
targets = []
for i, dim in enumerate(dims):
    target = np.random.randint(0, dim, (bs, ))
    target = torch.from_numpy(target)
    target = Variable(target)
    targets.append(target)

# create training criterion
criterions = [nn.CrossEntropyLoss() for _ in dims]
weights = [1] * len(dims)
criterion = MultiTaskCriterion(criterions, weights)

# create network
net = DeepCollaborationNetwork('resnet18', dims, pretrained=True)

# optimize
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, nesterov=True)
optimizer.zero_grad()
output = net(input)
loss = criterion(output, targets)
loss.backward()
optimizer.step()

```


