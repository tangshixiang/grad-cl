# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from .common import MLP, ResNet18, ResNet18_TinyImagenet


class Net(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(Net, self).__init__()
        nl, nh = args.n_layers, args.n_hiddens
        n_tasks = 20

        # setup network
        self.is_cifar = (args.data_file == 'cifar100.pt'
                         or args.data_file == 'cifar10.pt'
                         or args.data_file == 'whole_cifar100.pt')

        self.is_imagenet = (args.data_file == 'tiny-imagenet-200.pt')
        if self.is_cifar:
            self.net = ResNet18(n_outputs)
        elif self.is_imagenet:
            self.net = ResNet18_TinyImagenet(n_outputs)
        else:
            self.net = MLP([n_inputs] + [nh] * nl + [n_outputs])

        # setup optimizer
        self.opt = torch.optim.SGD(self.parameters(), lr=args.lr)

        # setup losses
        self.bce = torch.nn.CrossEntropyLoss()

        if self.is_cifar or self.is_imagenet:
            self.nc_per_task = n_outputs / n_tasks
        else:
            self.nc_per_task = n_outputs
        self.n_outputs = n_outputs

    def compute_offsets(self, task):
        if self.is_cifar or self.is_imagenet:
            offset1 = task * self.nc_per_task
            offset2 = (task + 1) * self.nc_per_task
        else:
            offset1 = 0
            offset2 = self.n_outputs
        return int(offset1), int(offset2)

    def forward(self, x, t):
        output = self.net(x)
        if self.is_cifar or self.is_imagenet:
            # make sure we predict classes within the current task
            offset1, offset2 = self.compute_offsets(t)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output

    def observe(self, x, t, y, ep):
        # if t == 1:
        #     import pdb
        #     pdb.set_trace()

        self.train()
        self.zero_grad()

        if self.is_cifar or self.is_imagenet:
            losses = []
            for i in range(len(y)):
                offset1, offset2 = self.compute_offsets(y[i].item() // 5)
                loss = self.bce((self.net(x[i:i + 1])[:, offset1:offset2]),
                                y[i:i + 1] - offset1)
                losses.append(loss)

            losses = torch.stack(losses).mean()
            print("loss: {}".format(losses.item()))
            losses.backward()
        else:
            self.bce(self(x, t), y).backward()
        self.opt.step()
