from __future__ import print_function

import torch
from torch.autograd import Variable
from my_resnext import create_resnext50_32x4d
from resnext_50_32x4d import resnext_50_32x4d as model
from utils import load_state_dict

# my model
my_model = create_resnext50_32x4d()
load_state_dict(my_model, torch.load('my_resnext_50_typeC_32x4d.pth'))

# model converted by https://github.com/clcarwin/convert_torch_to_pytorch
load_state_dict(model, torch.load('resnext_50_32x4d.pth'))

# pass the same data into the two models and compare the results
im = Variable(torch.rand(1, 3, 224, 224), volatile=True)
print('im.size(): ', im.size())
my_res = my_model(im)
res = model(im)
print('avg discrepancy: ', torch.mean(torch.abs(my_res - res)))
print('total discrepancy: ', torch.sum(torch.abs(my_res - res)))

print('\nmy result:\n', my_res, '\n')
print('result:\n', res)