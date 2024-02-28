import gzip
import json
import os
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import einops
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import trange
from eindex import eindex
from IPython.display import HTML, display
from jaxtyping import Float, Int
from rich import print as rprint
from rich.table import Table
from torch import Tensor, topk
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from transformer_lens import utils
from transformer_lens.hook_points import HookPoint
from sae_training.hooked_vit import HookedVisionTransformer, Hook
from sae_training.sparse_autoencoder import SparseAutoencoder
from sae_training.config import ViTSAERunnerConfig

"""
To do:
    - Implement a funciton that returns the highest activating images for each feature specified in feature_idx. Plus the different quartiles.
    - Args:
        Batch of images
        ViT model
        SAE model
        Hook class instance for where to hook the ViT at
        number of quartile ranges
        number of images per quartile
        number of max activating images
    - Returns:
        dictionary indexed by SAE feature idx. The values are instances of a data class containing:
            > batch of images for highest activating examples
            > adictionary containing quanitle numbers and a batch of images for each of the quantiles
            > attributes containg metadata
            > a method for saving the images to a file with relevant subfolders
            > other functionality later on (eg text explanation, feature vis)?
            
            
To do next:
    - Hook the SAE hidden layer post relu.
    - Forward pass the images through the model (use the cfg.max...)
    - obtain a [batch, feature_idx] tensor
    - find top k batch indexes for each feature idx
    - use these indices to index the max activating images
    - repeat but for different quantiles
"""

@dataclass
class FeatureData():
    feature_idx: int
    max_activating_images: Tensor # [Batch, C, W, H]
    quantile_activating_images: Optional[Dict[int, Tensor]] = None
    text_explanation: Optional[List[str]] = None
    feature_visualisation: Optional[Tensor] = None
    diffusion_visualisation: Optional[Tensor] = None
    
    def __post_init__(self):
        if self.quantile_activating_images is not None:
            self.number_of_quantiles = len(self.quantile_activating_images)
        else:
            self.number_of_quantiles = None
        
    def save_image_data(self, directory_path: str = "dashboard"):
        directory_path += f'/{self.feature_idx}'
        # Loop through each image tensor and save it
        for i, img_tensor in enumerate(self.max_activating_images):
            if not os.path.exists(directory_path + f'/max_activating'):
                os.makedirs(directory_path + f'/max_activating')
            file_path = os.path.join(directory_path, f'max_activating/image_{i}.png')
            save_image(img_tensor, file_path)


def get_model_activations(image_batches, model, cfg):
    module_name = cfg.module_name
    block_layer = cfg.block_layer
    list_of_hook_locations = [(block_layer, module_name)]

    activations = model.run_with_cache(
        image_batches,
        list_of_hook_locations,
    )[1][(block_layer, module_name)]
    
    activations = activations[:,0,:]

    return activations

def get_all_model_activations(image_batches, model, cfg):
    max_batch_size = cfg.max_batch_size_for_vit_forward_pass
    number_of_mini_batches = image_batches.size()[0] // max_batch_size
    remainder = image_batches.size()[0] % max_batch_size
    sae_batches = []
    for mini_batch in trange(number_of_mini_batches, desc = "Dashboard: forward pass images through ViT"):
        sae_batches.append(get_model_activations(image_batches[mini_batch*max_batch_size : (mini_batch+1)*max_batch_size], model, cfg))
    
    if remainder>0:
        sae_batches.append(get_model_activations(image_batches[-remainder:], model, cfg))
        
    sae_batches = torch.cat(sae_batches, dim = 0)
    sae_batches = sae_batches.to(cfg.device)
    return sae_batches

def get_image_data(dataset_path, image_key, number_of_images: int, cfg: ViTSAERunnerConfig):
    dataset = iter(load_dataset(dataset_path, split="test", streaming=True))
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.Resize((cfg.image_width, cfg.image_height)),  # Resize the image to WxH pixels
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    ])
    device = cfg.device
    images=[]
    for batch in trange(number_of_images, desc = "Dashboard: loading images for evals"):
        next_image = next(dataset)[image_key]
        next_image = transform(next_image) # next_image is a torch tensor with size [C, W, H].
        images.append(next_image)
    batches = torch.stack(images, dim = 0) # batches has size [batch_size, C, W, H].
    batches = batches.to(device)
    return batches

def get_sae_activations(model_activaitons, sparse_autoencoder, feature_idx: Tensor):
    hook_name = "hook_hidden_post"
    max_batch_size = sparse_autoencoder.cfg.max_batch_size_for_vit_forward_pass # Use this for the SAE too
    number_of_mini_batches = model_activaitons.size()[0] // max_batch_size
    remainder = model_activaitons.size()[0] % max_batch_size
    sae_activations = []
    for mini_batch in trange(number_of_mini_batches, desc = "Dashboard: obtaining sae activations"):
        sae_activations.append(sparse_autoencoder.run_with_cache(model_activaitons[mini_batch*max_batch_size : (mini_batch+1)*max_batch_size])[1][hook_name][:,feature_idx])
    
    if remainder>0:
        sae_activations.append(sparse_autoencoder.run_with_cache(model_activaitons[-remainder:])[1][hook_name][:,feature_idx])
        
    sae_activations = torch.cat(sae_activations, dim = 0)
    sae_activations = sae_activations.to(sparse_autoencoder.cfg.device)
    return sae_activations

@torch.inference_mode()
def get_feature_data(
    sparse_autoencoder: SparseAutoencoder,
    model: HookedVisionTransformer,
    feature_idx: List[int],
    dataset_path: str = "imagenet-1k",
    image_key: str = "image",
    number_of_images: int = 32_768,
    n_groups: int = 5,
    number_of_max_activating_images: int = 10,
    number_of_quantile_images: int = 5,

) -> Dict[int, FeatureData]:
    '''
    Gets data that will be used to create the sequences in the HTML visualisation.

    Args:
        feature_idx: int
            The identity of the feature we're looking at (i.e. we slice the weights of the encoder). A list of
            features is accepted (the result will be a list of FeatureData objects).
        max_batch_size: Optional[int]
            Optionally used to chunk the tokens, if it's a large batch

        left_hand_k: int
            The number of items in the left-hand tables (by default they're all 3).
        buffer: Tuple[int, int]
            The number of tokens on either side of the feature, for the right-hand visualisation.

    Returns object of class FeatureData (see that class's docstring for more info).
    '''
    
    """
    To do next:
        - Calculate the model activations to obtain a tensor of size [batch, d_resid]. Will need to do this in minibatches
        - run the SAE with cache (inherits from HookedRootModule). Probably best to do this in mini batches too. Only calculate on feature_idx. returns tensor of size [batch, feature_idx].
        - find top k batch indexes for each feature idx
        - use these indices to index the max activating images
        - repeat but for different quantiles
    """
    return_data = {}
    images = get_image_data(dataset_path, image_key, number_of_images, sparse_autoencoder.cfg) # tensor of size [batch, C, W, H]
    model_activations = get_all_model_activations(images, model, sparse_autoencoder.cfg) # tensor of size [batch, d_resid]
    sae_activations = get_sae_activations(model_activations, sparse_autoencoder, torch.tensor(feature_idx)) # tensor of size [batch, feature_idx]
    del model_activations
    values, indices = topk(sae_activations, k = number_of_max_activating_images, dim = 0) 
    for feature in trange(len(feature_idx), desc = "Dashboard: getting and saving highest activating images"):
        # Find the top activating images...
        max_image_indicees = indices[:,feature]
        max_images = images[max_image_indicees] # size [number_of_max_activating_images, C, W, H]
        
        # Find the quantile images... yet to implement
        
        data = FeatureData(feature_idx[feature], max_images)
        data.save_image_data()
        return_data[feature_idx[feature]] = data
        
    return return_data