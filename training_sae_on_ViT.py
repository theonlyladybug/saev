import os
import sys
import torch
import wandb
import json
import pickle
import plotly.express as px
from transformer_lens import utils
from datasets import load_dataset
from typing import  Dict
from pathlib import Path
from tqdm import tqdm
from functools import partial


sys.path.append("..")

from sae_training.utils import LMSparseAutoencoderSessionloader
from sae_analysis.visualizer import data_fns, html_fns
from sae_analysis.visualizer.data_fns import get_feature_data, FeatureData
from sae_training.config import ViTSAERunnerConfig
from sae_training.vit_runner import vision_transformer_sae_runner
from sae_training.train_sae_on_vision_transformer import train_sae_on_vision_transformer

if torch.backends.mps.is_available():
    device = "mps" 
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

def imshow(x, **kwargs):
    x_numpy = utils.to_numpy(x)
    px.imshow(x_numpy, **kwargs).show()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB__SERVICE_WAIT"] = "300"

cfg = ViTSAERunnerConfig(
    
    # Data Generating Function (Model + Training Distibuion)
    class_token = True,
    image_width = 224,
    image_height = 224,
    image_key = None,
    model_name = "vit_base_patch32_clip_224",
    module_name = "resid",
    block_layer = 10,
    dataset_path = "imagenet-1k",
    use_cached_activations = False,
    cached_activations_path = None,
    d_in = 768,
    
    # SAE Parameters
    expansion_factor = 64,
    b_dec_init_method = "geometric_median",
    
    # Training Parameters
    lr = 0.0004,
    l1_coefficient = 0.00008,
    lr_scheduler_name="cosineannealingwarmup",
    batch_size = 512,
    lr_warm_up_steps=5000,
    total_training_tokens = 16_000,
    n_batches_in_store = 1,
    
    # Dead Neurons and Sparsity
    use_ghost_grads=True,
    feature_sampling_method = None,
    feature_sampling_window = 1000,
    dead_feature_window=5000,
    dead_feature_threshold = 1e-6,
    
    # WANDB
    log_to_wandb = True,
    wandb_project= "mats-hugo",
    wandb_entity = None,
    wandb_log_frequency=100,
    
    # Misc
    device = "cuda",
    seed = 42,
    n_checkpoints = 0,
    checkpoint_path = "checkpoints",
    dtype = torch.float32,
    )

sparse_autoencoder = vision_transformer_sae_runner(cfg)

print("*****Done*****")