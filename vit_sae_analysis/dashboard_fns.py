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
from torchvision import transforms, datasets
from torchvision.utils import save_image
from tqdm import tqdm
from transformer_lens import utils
from transformer_lens.hook_points import HookPoint
from sae_training.hooked_vit import HookedVisionTransformer, Hook
from sae_training.sparse_autoencoder import SparseAutoencoder
from sae_training.config import ViTSAERunnerConfig
from sae_training.vit_activations_store import ViTActivationsStore

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
    max_image_values: Tensor # [Batch]
    feature_sparsity: float
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
        # Ensure the directory exists
        os.makedirs(directory_path, exist_ok=True)

        # Define the full path of the file
        file_path = os.path.join(directory_path, "sparsity.txt")

        # Open the file in write mode ('w') and write the string
        with open(file_path, 'w') as file:
            file.write(f"Log_10 feature sparsity: {torch.log10(self.feature_sparsity)}")
            
        # Loop through each image tensor and save it
        for i, img_tensor in enumerate(self.max_activating_images):
            if self.max_image_values[i]>0:
                if not os.path.exists(directory_path + f'/max_activating'):
                    os.makedirs(directory_path + f'/max_activating')
                file_path = os.path.join(directory_path, f'max_activating/{i}_{self.max_image_values[i]:.2g}.png')
                save_image(img_tensor, file_path)


def get_model_activations(model, inputs, cfg):
    module_name = cfg.module_name
    block_layer = cfg.block_layer
    list_of_hook_locations = [(block_layer, module_name)]

    activations = model.run_with_cache(
        list_of_hook_locations,
        **inputs,
    )[1][(block_layer, module_name)]
    
    activations = activations[:,0,:]

    return activations

def get_all_model_activations(model, images, cfg):
    max_batch_size = cfg.max_batch_size_for_vit_forward_pass
    number_of_mini_batches = len(images) // max_batch_size
    remainder = len(images) % max_batch_size
    sae_batches = []
    for mini_batch in trange(number_of_mini_batches, desc = "Dashboard: forward pass images through ViT"):
        image_batch = images[mini_batch*max_batch_size : (mini_batch+1)*max_batch_size]
        inputs = model.processor(images=image_batch, text = "", return_tensors="pt", padding = True).to(model.model.device)
        sae_batches.append(get_model_activations(model, inputs, cfg))
    
    if remainder>0:
        image_batch = images[-remainder:]
        inputs = model.processor(images=image_batch, text = "", return_tensors="pt", padding = True).to(model.model.device)
        sae_batches.append(get_model_activations(model, inputs, cfg))
        
    sae_batches = torch.cat(sae_batches, dim = 0)
    sae_batches = sae_batches.to(cfg.device)
    return sae_batches

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


def print_memory():
    # Check if CUDA is available
    if torch.cuda.is_available():
        # Get the current device
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        remaining_memory = total_memory-allocated_memory

        print(f"CUDA Device: {torch.cuda.get_device_name(device)}")
        print(f"Total Memory: {total_memory / (1024**3):.2f} GB")
        print(f"Allocated Memory: {allocated_memory / (1024**3):.2f} GB")
        print(f"Remaining Memory: {remaining_memory / (1024**3):.2f} GB")
    else:
        print("CUDA is not available.")


@torch.inference_mode()
def get_feature_data(
    sparse_autoencoder: SparseAutoencoder,
    model: HookedVisionTransformer,
    feature_idx: List[int],
    dataset_path: str = "imagenet-1k",
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
    torch.cuda.empty_cache()
    sparse_autoencoder.eval()
    
    dataset = load_dataset(sparse_autoencoder.cfg.dataset_path, split="train")
    
    if sparse_autoencoder.cfg.dataset_path=="cifar100": # Need to put this in the cfg
        image_key = 'img'
    else:
        image_key = 'image'
        
    dataset = dataset.shuffle(seed=42)
    iterable_dataset = iter(dataset)
    
    images = []
    for image in trange(number_of_images, desc = "Getting images for dashboard"):
        with torch.no_grad():
            try:
                images.append(next(iterable_dataset)[image_key])
            except StopIteration:
                iterable_dataset = iter(dataset.shuffle())
                images.append(next(iterable_dataset)[image_key])
    
    model_activations = get_all_model_activations(model, images, sparse_autoencoder.cfg) # tensor of size [batch, d_resid]
    sae_activations = get_sae_activations(model_activations, sparse_autoencoder, torch.tensor(feature_idx)) # tensor of size [batch, feature_idx]
    del model_activations
    values, indices = topk(sae_activations, k = number_of_max_activating_images, dim = 0)
    sparsity = (sae_activations>0).sum(dim = 0)/sae_activations.size()[0]
    for feature in trange(len(feature_idx), desc = "Dashboard: getting and saving highest activating images"):
        feature_sparsity = sparsity[feature]
        if feature_sparsity>0:
            # Find the top activating images...
            max_image_indicees = indices[:,feature]
            max_image_values = values[:, feature]
            feature_sparsity = sparsity[feature]
            max_images = images[max_image_indicees] # size [number_of_max_activating_images, C, W, H]
            
            # Find the quantile images... yet to implement
            
            data = FeatureData(feature_idx[feature], max_images, max_image_values, feature_sparsity)
            data.save_image_data()