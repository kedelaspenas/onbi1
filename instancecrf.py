from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from convcrf.convcrf import ConvCRF

import os
import sys

import numpy as np
import scipy as scp
import math

import logging
import warnings
try:
    import pyinn as P
    has_pyinn = True
except ImportError:
    #  PyInn is required to use our cuda based message-passing implementation
    #  Torch 0.4 provides a im2col operation, which will be used instead.
    #  It is ~15% slower.
    has_pyinn = False
    pass

from utils import test_utils

import torch
import torch.nn as nn
from torch.nn import functional as nnfun
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import torch.nn.functional as F

import gc


class InstanceCRF(nn.Module):
 

    def __init__(self, conf, shape, nclasses=None):
        super(InstanceCRF, self).__init__()

        self.conf = conf
        self.shape = shape
        self.nclasses = nclasses

        self.trainable = conf['trainable']

        if not conf['trainable_bias']:
            self.register_buffer('mesh', self._create_mesh())
        else:
            self.register_parameter('mesh', Parameter(self._create_mesh()))

        if self.trainable:
            def register(name, tensor):
                self.register_parameter(name, Parameter(tensor))
        else:
            def register(name, tensor):
                self.register_buffer(name, Variable(tensor))

        register('pos_sdims', torch.Tensor([1 / conf['pos_feats']['sdims']]))

        if conf['col_feats']['use_bias']:
            register('col_sdims',
                     torch.Tensor([1 / conf['col_feats']['sdims']]))
        else:
            self.col_sdims = None

        register('col_schan', torch.Tensor([1 / conf['col_feats']['schan']]))
        register('col_compat', torch.Tensor([conf['col_feats']['compat']]))
        register('pos_compat', torch.Tensor([conf['pos_feats']['compat']]))
        register('weight_box', torch.Tensor([0.5]))
        register('weight_global', torch.Tensor([0.5]))

        self.CRF = ConvCRF(
            shape, nclasses, mode="col", conf=conf,
            use_gpu=True, filter_size=conf['filter_size'],
            norm=conf['norm'], blur=conf['blur'], trainable=conf['trainable'],
            convcomp=conf['convcomp'], weight=None,
            final_softmax=conf['final_softmax'],
            unary_weight=conf['unary_weight'],
            pyinn=conf['pyinn'],verbose=False)

        return

    def forward(self, unary_box, unary_global, img, num_iter=5):

        conf = self.conf

        bs, c, x, y = img.shape

        pos_feats = self.create_position_feats(sdims=self.pos_sdims, bs=bs)
        col_feats = self.create_colour_feats(
            img, sdims=self.col_sdims, schan=self.col_schan,
            bias=conf['col_feats']['use_bias'], bs=bs)

        compats = [self.pos_compat, self.col_compat]

        self.CRF.add_pairwise_energies([pos_feats, col_feats],
                                       compats, conf['merge'])

        unary = self.weight_box*unary_box + self.weight_global*unary_global
        prediction = self.CRF.inference(unary, num_iter=num_iter)

        self.CRF.clean_filters()
        return prediction

    def _create_mesh(self, requires_grad=False):
        hcord_range = [range(s) for s in self.shape]
        mesh = np.array(np.meshgrid(*hcord_range, indexing='ij'),
                        dtype=np.float32)

        return torch.from_numpy(mesh)

    def create_colour_feats(self, img, schan, sdims=0.0, bias=True, bs=1):
        norm_img = img * schan

        if bias:
            norm_mesh = self.create_position_feats(sdims=sdims, bs=bs)
            feats = torch.cat([norm_mesh, norm_img], dim=1)
        else:
            feats = norm_img
        return feats

    def create_position_feats(self, sdims, bs=1):
        if type(self.mesh) is Parameter:
            return torch.stack(bs * [self.mesh * sdims])
        else:
            return torch.stack(bs * [Variable(self.mesh) * sdims])