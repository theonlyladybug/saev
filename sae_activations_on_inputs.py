import os
import sys
import torch
import wandb
import json
import plotly.express as px
from transformer_lens import utils
from datasets import load_dataset
from typing import  Dict
from pathlib import Path
from tqdm import tqdm
from functools import partial
from vit_sae_analysis.dashboard_fns import get_feature_data, FeatureData


sys.path.append("..")

from sae_training.utils import LMSparseAutoencoderSessionloader
from sae_analysis.visualizer import data_fns, html_fns
from sae_analysis.visualizer.data_fns import get_feature_data, FeatureData
from sae_training.config import ViTSAERunnerConfig
from sae_training.vit_runner import vision_transformer_sae_runner
from sae_training.train_sae_on_vision_transformer import train_sae_on_vision_transformer
from vit_sae_analysis.dashboard_fns import get_feature_data, FeatureData
from sae_training.sparse_autoencoder import SparseAutoencoder
from sae_training.utils import ViTSparseAutoencoderSessionloader

if torch.backends.mps.is_available():
    device = "mps" 
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
sae_path = "checkpoints/r9ompyb5/final_sparse_autoencoder_vit_base_patch32_clip_224_10_resid_49152.pt"

loaded_object = torch.load(sae_path)

cfg = loaded_object['cfg']

state_dict = loaded_object['state_dict']

sparse_autoencoder = SparseAutoencoder(cfg)

sparse_autoencoder.load_state_dict(state_dict)

sparse_autoencoder.eval()

loader = ViTSparseAutoencoderSessionloader(cfg)

model = loader.get_model(cfg.model_name)
model.to(cfg.device)

activations_loader = loader.get_activations_loader(cfg, model)

"""
Need to get a batch of images, store the images.
For each image do the following:
    Forward pass through the ViT, hook the activations.
    Get the model activations, pass them into the hooked SAE.
    get the SAE activations. Find which ones are positive, and only consider these indicies.
    Order the sae neurons by how strongly they're activated.
    Obtain an ordered list of tuples (neuron_idx, activation_score)
    create a directory titled with "Image_{image_number}". put the original image in this directory
    create a subdirectory for the sae neurons.
    create a subdirectory for each sae neuron within this. Title with f'{activation_score}_{neuron_idx}'
    in this directory, copy the sae directory from dashboard
"""