import numpy as np
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskCriterion():
    def __init__(self, criterions, weights):
        self.criterions = criterions
        self.weights = weights
        self.n_criterions = len(self.criterions)

    def __call__(self, predictions, targets):
        self.criterions_loss = []
        self.criterions_weighted_loss = []
        self.loss = 0

        for i in range(self.n_criterions):
            cur_loss = self.criterions[i](predictions[i], targets[i])
            self.criterions_loss.append(cur_loss)
            cur_loss = cur_loss * self.weights[i]
            self.criterions_weighted_loss.append(cur_loss)
            self.loss = self.loss + cur_loss

        return self.loss


def network_as_series_of_blocks(name, pretrained):
    blocks = []

    if name == 'resnet18':
        net = torchvision.models.resnet18(pretrained=pretrained)
        n_features = [64, 64, 128, 256, 512]
        blocks.append(nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool))
        blocks.append(net.layer1)
        blocks.append(net.layer2)
        blocks.append(net.layer3)
        blocks.append(net.layer4)

    elif name == 'alexnet':
        net = torchvision.models.alexnet(pretrained=pretrained)
        features = net.features
        n_features = [64, 192, 256]

        blocks.append(nn.Sequential(*[features[i] for i in range(0, 3)]))
        blocks.append(nn.Sequential(*[features[i] for i in range(3, 6)]))
        blocks.append(nn.Sequential(*[features[i] for i in range(6, 13)]))

    model = nn.Sequential(*blocks)

    return model, n_features


class DeepCollaborationNetwork(nn.Module):
    def __init__(self, underlying_network_name, out_dims, pretrained):
        super(DeepCollaborationNetwork, self).__init__()

        self.out_dims = out_dims
        self.n_cols = len(out_dims)

        # Define network
        n_features = network_as_series_of_blocks(underlying_network_name, pretrained)[1]
        columns = [network_as_series_of_blocks(underlying_network_name, pretrained)[0] for _ in range(self.n_cols)]
        self.columns = nn.Sequential(*columns)
        self.n_blocks = len(self.columns[0]._modules)

        def gen_central_aggr_block(n_in, n_out):
            layer = nn.Sequential(
                nn.Conv2d(n_in, n_out, 1, 1, 0, bias=False), # 1x1 conv
                nn.BatchNorm2d(n_out),
                nn.ReLU(),
                nn.Conv2d(n_out, n_out, 3, 1, 1, bias=False), # 3x3 conv
                nn.BatchNorm2d(n_out),
                nn.ReLU())
            return layer

        def gen_task_aggr_block(n_in, n_out):
            layer = nn.Sequential(
                nn.Conv2d(n_in, n_out, 1, 1, 0, bias=False),
                nn.BatchNorm2d(n_out),
                nn.ReLU(),
                nn.Conv2d(n_out, n_out, 3, 1, 1, bias=False),
                nn.BatchNorm2d(n_out))
            return layer

        central_aggr = []
        for nf in n_features:
            aggr = gen_central_aggr_block(nf * self.n_cols, nf * self.n_cols // 4)
            central_aggr.append(aggr)
        self.central_aggr = nn.Sequential(*central_aggr)

        task_aggr = []
        for col in range(self.n_cols):
            col_aggr = []
            for nf in n_features:
                aggr = gen_task_aggr_block(nf + nf * self.n_cols // 4, nf)
                col_aggr.append(aggr)
            task_aggr.append(nn.Sequential(*col_aggr))
        self.task_aggr = nn.Sequential(*task_aggr)


        # create fc layers
        def fc_block(dim_in, dim_out):
            dim_h = (dim_in + dim_out) // 2
            block = nn.Sequential(
                nn.Linear(dim_in, dim_h),
                nn.BatchNorm1d(dim_h),
                nn.ReLU(),
                nn.Linear(dim_h, dim_out))
            return block
        self.fcs = []
        for out_dim in self.out_dims:
            self.fcs.append(fc_block(n_features[-1], out_dim))
        self.fcs = nn.Sequential(*self.fcs)


    def forward(self, x):

        block_inputs = [x] * self.n_cols
        for i in range(self.n_blocks):
            block_outputs = []
            for j, y in enumerate(block_inputs):
                block_outputs.append(self.columns[j][i](y))
            aggr = torch.cat(block_outputs, 1)
            aggr = self.central_aggr[i](aggr)

            block_inputs = []
            for j, y in enumerate(block_outputs):
                z = torch.cat([aggr, y], 1)
                block_inputs.append(y + self.task_aggr[j][i](z))

        outputs = []
        for j, y in enumerate(block_inputs):
            z = F.avg_pool2d(y, kernel_size=y.size()[2:])
            z = z.view(z.size(0), -1)
            outputs.append(self.fcs[j](z))

        return outputs

