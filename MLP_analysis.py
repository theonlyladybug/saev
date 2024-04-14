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
from sae_training.config import ViTSAERunnerConfig
import torchvision.transforms as transforms
from PIL import Image
import shutil


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

def get_new_top_k(first_values, first_indices, second_values, second_indices, k):
    total_values = torch.cat(first_values, second_values)
    total_indices = torch.cat(first_indices, second_indices)
    new_values, indices_of_indices = topk(total_values, k=k, dim=1)
    new_indices = torch.gather(total_indices, 1, indices_of_indices)
    return new_values, new_indices

@torch.inference_mode()
def get_feature_data(
    sae_config: ViTSAERunnerConfig, 
    model: HookedVisionTransformer,
    number_of_images: int = 32_768,
    number_of_max_activating_images: int = 10,
    max_number_of_images_per_iteration: int = 16_384,
):
    '''
    Need to do the following:
    save index and activation level of directions.
    '''
    torch.cuda.empty_cache()
    
    dataset = load_dataset(sae_config.dataset_path, split="train")
    
    if sae_config.dataset_path=="cifar100": # Need to put this in the cfg
        image_key = 'img'
    else:
        image_key = 'image'
    MLP_size = model.model.vision_model.config.intermediate_size
    MLP_out_weights = model.model.vision_model.encoder.layers[sae_config.block_layer].mlp.fc2.weight.detach() # size [resid_dimension, mlp_dim]
    mlp_out_weights /= torch.norm(mlp_out_weights, dim = 0, keepdim = True)
    #Create two tensors, both of size [MLP, num_max_act]. Stores index and activation value (or projected component).
    max_activating_image_indices = torch.zeros([MLP_size, number_of_max_activating_images]) - 1
    max_activating_image_values = torch.zeros([MLP_size, number_of_max_activating_images])
    # Get tensor of size [batch, colour, width, height]?
    
    # Get tesnor of size [batch, reisd]?
    
    # Get tensor of size [batch, MLP]? NOT by passing through the MLP encoder but instead by multiplying by the normalised decoder matrix.
    
    # Create a second tensor with the index values in them
    
    # Concatinate with the 'global' tensors
    
    # Sort the tensors and then delete the extra dimensions
    
    # Job done
    
    number_of_images_processed = 0
    while number_of_images_processed < number_of_images:
        torch.cuda.empty_cache()
        try:
            images = dataset[number_of_images_processed:number_of_images_processed + max_number_of_images_per_iteration][image_key]
        except StopIteration:
            print('All of the images in the dataset have been processed!')
            all_images_processed=True
            break
        
        model_activations = get_all_model_activations(model, images, sae_config) # tensor of size [batch, d_resid]
        MLP_basis_activations = torch.nn.Functional.relu(model_activations @ MLP_out_weights) # size [batch, d_resid]        
        del model_activations
        
        values, indices = topk(MLP_basis_activations, k = number_of_max_activating_images, dim = 1)
        indices += number_of_images_processed
        
        max_activating_image_values, max_activating_image_indices = get_new_top_k(max_activating_image_values, max_activating_image_indices, values, indices)
                
        number_of_images_processed += max_number_of_images_per_iteration
        
    directory = 'MLP'
    # Check if the directory exists
    if not os.path.exists(directory):
        # Create the directory if it does not exist
        os.makedirs(directory)
    
    torch.save(max_activating_image_indices, '{directory}/max_activating_image_indices.pt')
    torch.save(max_activating_image_values, '{directory}/max_activating_image_values.pt')
