# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.nn.functional import relu
import numpy as np
import random
import cv2

from .common import ResNet18


class Feature_Extractor(object):
    """ Classes for extracting activations and registering gradients from
        targetted intermediate layers"""

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        bsz = x.size(0)
        x = x.view(bsz, 3, 32, 32)
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            if name == 'linear':
                continue
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModuleOutputs(object):
    """ Class for making a forward pass, and getting:
    1. The network output
    2. activations from intermediate targetted layers
    3. gradients from intermediate targetted layers. """

    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = Feature_Extractor(self.model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.model.linear(output)
        return target_activations, output


class GradCam(object):
    def __init__(self, model, target_layer_names, use_cuda, mode):
        self.model = model
        if mode == 'train':
            self.model.train()
        elif mode == 'eval':
            self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.extractor = ModuleOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, offset1, offset2, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index is None:
            if offset1 > 0:
                index = np.argmax(
                    output.cpu().data.numpy()[:, :offset1], axis=1)
            else:
                index = np.argmax(output.cpu().data.numpy(), axis=1)

        one_hot = np.zeros((output.size()[0], output.size()[-1]),
                           dtype=np.float32)

        for i in range(len(one_hot)):
            one_hot[i][index[i]] = 1

        one_hot = torch.from_numpy(one_hot)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        self.model.linear.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1]

        target = features[-1]

        weights = torch.mean(grads_val, dim=2, keepdim=True)
        weights = torch.mean(weights, dim=3, keepdim=True)

        cam = weights * target
        cam = torch.sum(cam, dim=1)

        cam = relu(cam)

        cam = cam / (torch.norm(cam) + 1e-10)
        return output, cam


class Net(torch.nn.Module):
    # Re-implementation of Learning with Forgetting
    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(Net, self).__init__()
        self.nt = n_tasks
        self.reg = args.memory_strength
        self.n_feat = n_outputs
        self.n_classes = n_outputs
        self.samples_per_task = args.samples_per_task
        if self.samples_per_task <= 0:
            error("set explicitly args.samples_per_task")
        self.examples_seen = 0

        # setup network
        assert (args.data_file == 'cifar100.pt')

        self.net = ResNet18(n_outputs)
        self.cam_net = GradCam(self.net, 'layer4', args.cuda, mode='train')
        self.old_net = ResNet18(n_outputs)
        self.old_cam_net = GradCam(
            self.old_net, 'layer4', args.cuda, mode='eval')
        self.n_epochs = args.n_epochs

        # setup optimizer
        self.opt = torch.optim.SGD(self.net.parameters(), lr=args.lr)

        # setup losses
        self.bce = torch.nn.CrossEntropyLoss()
        self.kl = torch.nn.KLDivLoss()
        self.lsm = torch.nn.LogSoftmax(dim=1)
        self.sm = torch.nn.Softmax(dim=1)
        self.l1_norm = torch.nn.L1Loss()

        self.gpu = args.cuda
        self.is_cifar = (args.data_file == 'cifar100.pt')
        if self.is_cifar:
            self.nc_per_task = int(n_outputs / n_tasks)
        self.n_outputs = n_outputs

        # GAM-distillation

    def compute_offsets(self, task):
        offset1 = task * self.nc_per_task
        offset2 = (task + 1) * self.nc_per_task
        return int(offset1), int(offset2)

    def forward(self, x, t):
        output = self.net(x)
        if self.is_cifar:
            # make sure we predict classes within the current task
            offset1, offset2 = self.compute_offsets(t)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output

    def observe(self, x, t, y):
        self.net.train()
        self.net.zero_grad()
        self.old_net.zero_grad()

        self.examples_seen += x.size(0)

        offset1, offset2 = self.compute_offsets(t)
        logits, cam = self.cam_net(x, offset1, offset2)

        loss = self.bce(logits[:, offset1:offset2], y - offset1)

        if t > 0:
            # Add distillation loss
            old_logits, old_cam = self.old_cam_net(x, offset1, offset2)
            loss += self.reg * self.l1_norm(cam, old_cam)

            # for tt in range(t):
            #     offset1, offset2 = self.compute_offsets(tt)
            #     loss += self.reg * self.kl(
            #         self.lsm(logits[:, offset1:offset2]),
            #         self.sm(old_logits[:, offset1:offset2])) * self.nc_per_task

        # bprop and update
        self.net.zero_grad()
        loss.backward()
        self.opt.step()

        if self.examples_seen == self.samples_per_task * self.n_epochs:
            self.examples_seen = 0
            self.old_net.load_state_dict(self.net.state_dict())
