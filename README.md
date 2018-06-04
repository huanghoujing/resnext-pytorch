This repository just provides ResNeXt definition and corresponding model weight, especially for ResNeXt-50-32x4d-typeC. 

# Requirement

- python 2.7
- pytorch 0.3
- No need of GPU

# Convert Torch Model Weight to Pytorch

Download the ImageNet model weight from this official url <https://s3.amazonaws.com/resnext/imagenet_models/resnext_50_32x4d.t7>. Place it in the current directory.

Run `python resnext_50_typeC_32x4d_torch_to_pytorch.py` to convert it to pytorch state_dict, which will be saved as `my_resnext_50_typeC_32x4d.pth` in the current directory. The conversion is simply done by:
- loading the torch model as a pytorch `torch.legacy.nn.Sequential.Sequential` object, using `torch.utils.serialization.load_lua`
- indexing and copying the model weights.

# Check the Correctness of Conversion

To check the correctness of conversion, I refer to another repository <https://github.com/clcarwin/convert_torch_to_pytorch>, which saves both the model definition and model weight from the official torch checkpoint.
- First, clone that repository and run `python convert_torch.py -m resnext_50_32x4d.t7`. Here `resnext_50_32x4d.t7` is the path to your downloaded official model. This step will generate two files, `resnext_50_32x4d.py` and `resnext_50_32x4d.pth`.
- Then, move these two files to the current directory.
- Run `python test_correctness.py`. The two results were identical in my run.

# Misc

When indexing the model loaded by `torch.utils.serialization.load_lua`, I found a minor mismatch between the definition of [`resnext.lua`](https://github.com/facebookresearch/ResNeXt/blob/3cf474fdffa9ba4ce11ad41c0278e38fcd47372f/models/resnext.lua) and the torch checkpoint. The difference is in `resnext_bottleneck_C`, whether to wrap the first two conv layers in an extra `nn.Sequential()`. I provide a new lua file that is consistent with the torch checkpoint. The [original](https://github.com/facebookresearch/ResNeXt/blob/3cf474fdffa9ba4ce11ad41c0278e38fcd47372f/models/resnext.lua#L108) and the [modified]() `resnext_bottleneck_C` definition.

# Reference

- [ResNeXt paper](https://arxiv.org/abs/1611.05431)
- [ResNeXt official code](https://github.com/facebookresearch/ResNeXt)
- [convert_torch_to_pytorch](https://github.com/clcarwin/convert_torch_to_pytorch)
