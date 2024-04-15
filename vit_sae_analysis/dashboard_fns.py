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
import torchvision.transforms as transforms
from PIL import Image
import shutil

def load_images_and_convert_to_tensors(directory_path, device='cuda'):
    images_tensors = []
    activations = []
    supported_formats = '.png'  # Targeting PNG files
    
    for entry in os.listdir(directory_path):
        if entry.lower().endswith(supported_formats):
            img_path = os.path.join(directory_path, entry)
            img = Image.open(img_path)  # Ensure image is in RGB
            img_tensor = convert_images_to_tensor([img], device=device)
            images_tensors.append(img_tensor)
            activation = float(entry.split('_')[1].replace('.png',''))
            activations.append(torch.tensor([activation]))
    images_tensors = torch.concat(images_tensors, dim =0).to(device)
    activations = torch.concat(activations, dim =0).to(device)
    return images_tensors, activations

def convert_images_to_tensor(images, device='cuda'):
    """
    Convert a list of PIL images to a PyTorch tensor in RGB format with shape [B, C, H, W].

    Parameters:
    - images: List of PIL.Image objects.
    - device: The device to store the tensor on ('cpu' or 'cuda').

    Returns:
    - A PyTorch tensor with shape [B, C, H, W].
    """
    # Define a transform to convert PIL images (in RGB) to tensors
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),  # Convert image to RGB
        transforms.Resize((256, 256)),  # Resize the image to 224x224 pixels
        transforms.ToTensor(),  # Convert the image to a torch tensor
    ])

    # Ensure each image is in RGB format, apply the transform, and move to the specified device
    tensor_list = [transform(img).to(device) for img in images]
    tensor_output = torch.stack(tensor_list, dim=0)

    return tensor_output

def delete_files_in_directory(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


@dataclass
class FeatureData():
    feature_idx: int
    max_activating_images: Tensor # [Batch, C, W, H]
    max_image_values: Tensor # [Batch]
    feature_sparsity: float
    text_encoding: Optional[Tensor] = None
        
    def save_image_data(self, directory_path: str = "dashboard", device = 'cuda'):
        directory_path += f'/{self.feature_idx}'
        # Ensure the directory exists
        os.makedirs(directory_path, exist_ok=True)

        # Define the full path of the file
        file_path = os.path.join(directory_path, "sparsity.txt")

        # Open the file in write mode ('w') and write the string
        with open(file_path, 'a') as file:
            file.write(f"Log_10 feature sparsity: {torch.log10(self.feature_sparsity)}" + "\n")
        
        if self.text_encoding is not None:
            torch.save(self.text_encoding, directory_path + '/text_encoding.pt')
            
        # Check whether max activating examples exist!
        if os.path.exists(directory_path + f'/max_activating') and os.path.isdir(directory_path + f'/max_activating'):
            images_tensors, activations = load_images_and_convert_to_tensors(directory_path + f'/max_activating', device=device)
            delete_files_in_directory(directory_path + f'/max_activating')
            number_of_images = self.max_image_values.size()[0]
            # Merge the saved images and new images
            self.max_activating_images = torch.concat([self.max_activating_images,images_tensors], dim = 0).to(device)
            self.max_image_values = torch.concat([self.max_image_values,activations], dim = 0).to(device)
            # sort the merged tensors
            _, indices = self.max_image_values.sort(descending=True)
            self.max_image_values = self.max_image_values[indices]
            self.max_activating_images = self.max_activating_images[indices]
            # Take only the top n images
            self.max_image_values = self.max_image_values[:number_of_images]
            self.max_activating_images = self.max_activating_images[:number_of_images]
        
        # Loop through each image tensor and save it
        for i, img_tensor in enumerate(self.max_activating_images):
            if self.max_image_values[i]>0:
                if not os.path.exists(directory_path + f'/max_activating'):
                    os.makedirs(directory_path + f'/max_activating')
                file_path = os.path.join(directory_path, f'max_activating/{i}_{self.max_image_values[i]:.4g}.png')
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
    number_of_images: int = 32_768,
    number_of_max_activating_images: int = 10,
    max_number_of_images_per_iteration: int = 16_384,
    seed: int = 1,
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
    
    if sparse_autoencoder.cfg.model_name == "openai/clip-vit-large-patch14": # Need to include layernorm in the future! Also put this in a function!!
        visual_proj = model.model.visual_projection.weight.detach()
        text_proj = model.model.text_projection.weight.detach()
        inverse_text_proj = torch.inverse(text_proj)
        map_to_text = torch.matmul(inverse_text_proj, visual_proj)
        del visual_proj, text_proj, inverse_text_proj
    
    text_encoding = None
    
    dataset = load_dataset(sparse_autoencoder.cfg.dataset_path, split="train")
    
    if sparse_autoencoder.cfg.dataset_path=="cifar100": # Need to put this in the cfg
        image_key = 'img'
    else:
        image_key = 'image'
        
    dataset = dataset.shuffle()
    iterable_dataset = iter(dataset)
    
    number_of_images_processed = 0
    all_images_processed=False
    while number_of_images_processed < number_of_images:
        torch.cuda.empty_cache()
        images = []
        for image in trange(max_number_of_images_per_iteration, desc = "Getting images for dashboard"):
            with torch.no_grad():
                try:
                    images.append(next(iterable_dataset)[image_key])
                except StopIteration:
                    print('All of the images in the dataset have been processed!')
                    all_images_processed=True
                    break
        
        model_activations = get_all_model_activations(model, images, sparse_autoencoder.cfg) # tensor of size [batch, d_resid]
        sae_activations = get_sae_activations(model_activations, sparse_autoencoder, torch.tensor(feature_idx)) # tensor of size [batch, feature_idx]
        del model_activations
        
        # Convert the images list to a torch tensor
        images = convert_images_to_tensor(images)
        values, indices = topk(sae_activations, k = number_of_max_activating_images, dim = 0)
        sparsity = (sae_activations>0).sum(dim = 0)/sae_activations.size()[0]
        
        
        for feature in trange(len(feature_idx), desc = "Dashboard: getting and saving highest activating images"):
            feature_sparsity = sparsity[feature]
            if feature_sparsity>0:
                # Find the top activating images...
                max_image_indices = indices[:,feature]
                max_image_values = values[:, feature]
                feature_sparsity = sparsity[feature]
                max_images = images[max_image_indices] # size [number_of_max_activating_images, C, W, H]
                
                # Find the vector to condition stable difussion on
                if sparse_autoencoder.cfg.model_name == "openai/clip-vit-large-patch14":
                    encoder_vector = sparse_autoencoder.W_enc[:,feature_idx[feature]]
                    encoder_vector /= torch.norm(encoder_vector).pow(2)
                    residual_vector = sparse_autoencoder.b_dec + encoder_vector * (1-sparse_autoencoder.b_enc[feature_idx[feature]])
                    text_encoding = torch.matmul(map_to_text, residual_vector)
                
                data = FeatureData(feature_idx[feature], max_images, max_image_values, feature_sparsity, text_encoding = text_encoding)
                data.save_image_data()
                
        number_of_images_processed += max_number_of_images_per_iteration
        
        del images, values, indices, sparsity, sae_activations
        
        if all_images_processed:
            break
        
"""

def save_highest_activating_images(max_activating_image_indices, max_activating_image_values, directory, dataset, image_key):
    assert max_activating_image_values.size() == max_activating_image_indices.size(), "size of max activating image indices doesn't match the size of max activing values."
    number_of_neurons, number_of_max_activating_examples = max_activating_image_values.size()
    for neuron in trange(number_of_neurons):
        if not os.path.exists(f"{directory}/{neuron}"):
            os.makedirs(f"{directory}/{neuron}")
        for max_activaitng_image in range(number_of_max_activating_examples):
            image = dataset[int(max_activating_image_indices[neuron, max_activaitng_image].item())][image_key]
            image.save(f"{directory}/{neuron}/{max_activaitng_image}_{int(max_activating_image_indices[neuron, max_activaitng_image].item())}_{max_activating_image_values[neuron, max_activaitng_image].item():.4g}.png", "PNG")

def get_mean(dataset, model, image_key, sae_config, mean_size = 16_384):
    images = dataset[:mean_size][image_key]
    acts = get_all_model_activations(model, images, sae_config)
    mean_acts = torch.mean(acts, dim = 0)
    return mean_acts

@torch.inference_mode()
def get_feature_data(
    sae_config: ViTSAERunnerConfig, 
    model: HookedVisionTransformer,
    number_of_images: int = 32_768,
    number_of_max_activating_images: int = 15,
    max_number_of_images_per_iteration: int = 16_384,
    load_pretrained = True,
    seed = 1,
):
    '''
    Need to do the following:
    save index and activation level of directions.
    '''
    torch.cuda.empty_cache()
    
    dataset = load_dataset(sae_config.dataset_path, split="train")
    dataset = dataset.shuffle(seed = seed)
    directory = 'MLP'
    
    if sae_config.dataset_path=="cifar100": # Need to put this in the cfg
        image_key = 'img'
    else:
        image_key = 'image'
    MLP_size = model.model.vision_model.config.intermediate_size
    MLP_out_weights = model.model.vision_model.encoder.layers[sae_config.block_layer].mlp.fc2.weight.detach() # size [resid_dimension, mlp_dim]
    MLP_out_weights /= torch.norm(MLP_out_weights, dim = 0, keepdim = True)
    mean_acts = get_mean(dataset, model, image_key, sae_config)
    if load_pretrained:
        max_activating_image_indices = torch.load(f'{directory}/max_activating_image_indices.pt')
        max_activating_image_values = torch.load(f'{directory}/max_activating_image_values.pt')
    else:
        #Create two tensors, both of size [MLP, num_max_act]. Stores index and activation value (or projected component).
        max_activating_image_indices = torch.zeros([MLP_size, number_of_max_activating_images]).to(sae_config.device)
        max_activating_image_values = torch.zeros([MLP_size, number_of_max_activating_images]).to(sae_config.device)
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
            
            model_activations = get_all_model_activations(model, images, sae_config) - mean_acts # tensor of size [batch, d_resid], with mean removed
            MLP_basis_activations = torch.nn.functional.relu(model_activations @ MLP_out_weights).transpose(0,1) # size [MLP_dim, batch]        
            del model_activations
            
            values, indices = topk(MLP_basis_activations, k = number_of_max_activating_images, dim = 1)
            indices += number_of_images_processed
            
            max_activating_image_values, max_activating_image_indices = get_new_top_k(max_activating_image_values, max_activating_image_indices, values, indices, number_of_max_activating_images)
                    
            number_of_images_processed += max_number_of_images_per_iteration
            
        # Check if the directory exists
        if not os.path.exists(directory):
            # Create the directory if it does not exist
            os.makedirs(directory)
        
        torch.save(max_activating_image_indices, f'{directory}/max_activating_image_indices.pt')
        torch.save(max_activating_image_values, f'{directory}/max_activating_image_values.pt')
    
    save_highest_activating_images(max_activating_image_indices[:1000], max_activating_image_values[:1000], directory, dataset, image_key)

"""