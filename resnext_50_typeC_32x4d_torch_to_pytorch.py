from __future__ import print_function


"""The definition of the torch resnext_50_32x4d typeC model in the code [resnext.lua]

shortcut:
seq
    conv
    bn
OR
identity

resnext_bottleneck_C:
seq
    concat
        seq
            conv
            bn
            relu
            group conv
            bn
            relu
            conv
            bn
        shortcut
    add
    relu    

layer(n_blocks):
seq
    resnext_bottleneck_C
    ...
    resnext_bottleneck_C
    (totally n_blocks)
    
model:
seq
    conv
    bn
    relu
    max_pool
    layer(3)
    layer(4)
    layer(6)
    layer(3)
    avg_pool
    view
    linear
"""

"""The definition of the torch resnext_50_32x4d typeC model in the provided [resnext_50_32x4d.t7]
The difference to [resnext.lua] is in the bottleneck implementation

shortcut:
seq
    conv
    bn
OR
identity

resnext_bottleneck_C:
seq
    concat
        seq
            seq
                conv
                bn
                relu
                group conv
                bn
                relu
            conv
            bn
        shortcut
    add
    relu    

layer(n_blocks):
seq
    resnext_bottleneck_C
    ...
    resnext_bottleneck_C
    (totally n_blocks)

model:
seq
    conv
    bn
    relu
    max_pool
    layer(3)
    layer(4)
    layer(6)
    layer(3)
    avg_pool
    view
    linear
"""


import torch
from torch.utils.serialization import load_lua
from collections import namedtuple


# `unknown_classes=True` to ignore the error caused by unmanageable cudnn modules
torch_model = load_lua('resnext_50_32x4d.t7', unknown_classes=True)

# module_alias: just for better understanding, not used for indexing etc.
# module_type: one of the parametric module ['conv', 'bn', 'fc'], it's also not used eventually.
# pytorch_module_name: the module name in the pytorch state_dict.
# torch_module: an object, which is one sub-module of the loaded torch model.
ModuleMapping = namedtuple('ModuleMapping', ['module_alias', 'module_type', 'pytorch_module_name', 'torch_module'])

def get_block_mapping(torch_model, stage, block, with_shortcut_transform):
    """Get the mapping between torch and pytorch modules inside a block.
    Args:
        torch_model: the loaded torch model
        stage: one-based index, one of [1, 2, 3, 4]
        block: one-based index, one of [1, 2, 3, ...]
        with_shortcut_transform: False if identity shortcut, else True
    """
    mapping = [
        ModuleMapping('STAGE{}-BLOCK{}-MAINBRANCH-CONV1'.format(stage, block),      'conv',     'stage{}.{}.main_branch.0'.format(stage, block - 1),    torch_model.modules[stage + 3].modules[block - 1].modules[0].modules[0].modules[0].modules[0]),
        ModuleMapping('STAGE{}-BLOCK{}-MAINBRANCH-BN1'.format(stage, block),        'bn',       'stage{}.{}.main_branch.1'.format(stage, block - 1),    torch_model.modules[stage + 3].modules[block - 1].modules[0].modules[0].modules[0].modules[1]),
        ModuleMapping('STAGE{}-BLOCK{}-MAINBRANCH-CONV2GROUP'.format(stage, block), 'conv',     'stage{}.{}.main_branch.3'.format(stage, block - 1),    torch_model.modules[stage + 3].modules[block - 1].modules[0].modules[0].modules[0].modules[3]),
        ModuleMapping('STAGE{}-BLOCK{}-MAINBRANCH-BN2'.format(stage, block),        'bn',       'stage{}.{}.main_branch.4'.format(stage, block - 1),    torch_model.modules[stage + 3].modules[block - 1].modules[0].modules[0].modules[0].modules[4]),
        ModuleMapping('STAGE{}-BLOCK{}-MAINBRANCH-CONV3'.format(stage, block),      'conv',     'stage{}.{}.main_branch.6'.format(stage, block - 1),    torch_model.modules[stage + 3].modules[block - 1].modules[0].modules[0].modules[1]),
        ModuleMapping('STAGE{}-BLOCK{}-MAINBRANCH-BN3'.format(stage, block),        'bn',       'stage{}.{}.main_branch.7'.format(stage, block - 1),    torch_model.modules[stage + 3].modules[block - 1].modules[0].modules[0].modules[2]),
    ]
    if with_shortcut_transform:
        mapping.append(ModuleMapping('STAGE{}-BLOCK{}-SHORTCUT-CONV'.format(stage, block),  'conv', 'stage{}.{}.shortcut_transform.0'.format(stage, block - 1), torch_model.modules[stage + 3].modules[block - 1].modules[0].modules[1].modules[0]))
        mapping.append(ModuleMapping('STAGE{}-BLOCK{}-SHORTCUT-BN'.format(stage, block),    'bn',   'stage{}.{}.shortcut_transform.1'.format(stage, block - 1), torch_model.modules[stage + 3].modules[block - 1].modules[0].modules[1].modules[1]))
    return mapping

mapping = [
    ModuleMapping('CONV1', 'conv', 'conv1', torch_model.modules[0]),
    ModuleMapping('BN1', 'bn', 'bn1', torch_model.modules[1]),
]

for stage, blocks in zip([1, 2, 3, 4], [3, 4, 6, 3]):
    for block in range(1, blocks + 1):
        with_shortcut_transform = True if block == 1 else False
        mapping.extend(get_block_mapping(torch_model, stage, block, with_shortcut_transform))

mapping.append(ModuleMapping('FC', 'fc', 'fc', torch_model.modules[10]))

# A state_dict is what we need when initializing a pytorch model.
# It contains mapping from parameter name to parameter value.
def get_module_state_dict(mapping):
    state_dict = {}
    for weight_type in ['weight', 'bias', 'running_mean', 'running_var']:
        if hasattr(mapping.torch_module, weight_type) and getattr(mapping.torch_module, weight_type) is not None:
            state_dict[mapping.pytorch_module_name + '.' + weight_type] = getattr(mapping.torch_module, weight_type)
    return state_dict

state_dict = {}
for m in mapping:
    state_dict.update(get_module_state_dict(m))

torch.save(state_dict, 'my_resnext_50_typeC_32x4d.pth')
