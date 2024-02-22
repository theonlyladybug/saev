import torch
import torch.nn as nn
import timm
import math
from typing import Callable
from contextlib import contextmanager
from typing import List, Union, Dict, Tuple
from functools import partial
  


# The Hook class does not currently only supports hooking on the following locations:
# 1 - residual stream post transformer block.
# 2 - mlp activations.
# More hooks can be added at a later date, but only post-module.
class Hook():
  def __init__(self, block_layer: int, module_name: str, hook_fn: Callable):
    self.path_dict = {
        'resid': '',
        'mlp': '.mlp.act',
    }
    assert module_name in self.path_dict.keys(), f'Module name \'{module_name}\' not recognised.'
    self.function = self.get_full_hook_fn(hook_fn, module_name)
    self.attr_path = self.get_attr_path(block_layer, module_name)

  def get_full_hook_fn(self, hook_fn: Callable, module_name: str):

    def full_hook_fn(module, module_input, module_output):
      return hook_fn(module_output)

    return full_hook_fn

  def get_attr_path(self, block_layer: int, module_name: str) -> str:
    attr_path = f'blocks[{block_layer}]'
    attr_path += self.path_dict[module_name]
    return attr_path
  
  def get_module(self, model):
    return self.get_nested_attr(model, self.attr_path)

  def get_nested_attr(self, model, attr_path):
    """
    Gets a nested attribute from an object using a dot-separated path.
    """
    attributes = attr_path.split(".")
    for attr in attributes:
        if '[' in attr:
            # Split at '[' and remove the trailing ']' from the index
            attr_name, index = attr[:-1].split('[')
            module = getattr(model, attr_name)[int(index)]
        else:
            module = getattr(model, attr)
    return module



class HookedVisionTransformer():
  def __init__(self, model_name: str, num_classes: int = -1):
    self.model = self.get_ViT(model_name, num_classes)

  def get_ViT(self, model_name, num_classes):
    model = timm.create_model(model_name, pretrained=True, num_classes = num_classes)
    return model

  def run_with_cache(self, input_batch, list_of_hook_locations: List[Tuple[int,str]], *args, **kwargs):
    cache_dict, list_of_hooks = self.get_caching_hooks(list_of_hook_locations)
    with self.hooks(list_of_hooks) as hooked_model:
      output = hooked_model(input_batch, *args, **kwargs)
    return output, cache_dict

  def get_caching_hooks(self, list_of_hook_locations: List[Tuple[int,str]]):
    """
    Note that the cache dictionary is index by the tuple (block_layer, module_name).
    """
    cache_dict = {}
    list_of_hooks=[]
    def save_activations(name, activations):
      cache_dict[name] = activations
    for (block_layer, module_name) in list_of_hook_locations:
      hook_fn = partial(save_activations, (block_layer, module_name))
      hook = Hook(block_layer, module_name, hook_fn)
      list_of_hooks.append(hook)
    return cache_dict, list_of_hooks

  def run_with_hooks(self, input_batch, list_of_hooks: List[Hook], *args, **kwargs):
    with self.hooks(list_of_hooks) as hooked_model:
      output = hooked_model(input_batch, *args, **kwargs)
    return output

  @contextmanager
  def hooks(self, hooks: List[Hook]):
    """

    This is a context manager for running a model with hooks. The funciton adds 
    forward hooks to the model, and then returns the hooked model to be run with 
    a foward pass. The funciton then cleans up by removing any hooks.

    Args:

      model VisionTransformer: The ViT that you want to run with the forward hook

      hooks List[Tuple[str, Callable]]: A list of forward hooks to add to the model. 
        Each hook is a tuple of the module name, and the hook funciton.

    """
    hook_handles = []
    try:
      for hook in hooks:
        # Create a full hook funciton, with all the argumnets needed to run nn.module.register_forward_hook().
        # The hook functions are added to the output of the module.
        module = hook.get_module(self.model)
        handle = module.register_forward_hook(hook.function)
        hook_handles.append(handle)
      yield self.model
    finally:
      for handle in hook_handles:
        handle.remove()

  def __call__(self, input, *args, **kwargs):
    return self.forward(input, *args, **kwargs)

  def forward(self, input, *args, **kwargs):
    return self.model(input, *args, **kwargs)