from torch import nn
import torch
from vit_sae_analysis.dashboard_fns import get_model_activations, get_sae_activations, get_all_model_activations
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from leap_ie.vision import engine

"""
Need to load sae, initialise new_sae, change state_dict
"""
target_index=133


    
class partial_model(nn.Module):
    def __init__(self, sae_config, sae):
        super().__init__()
        self.clip_model, self.processor = self.get_ViT(sae_config.model_name)
        self.sae_config = sae_config
        self.sae = sae
    
    def get_ViT(self, model_name):
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        return model, processor
    
    def forward(self, x):
        x = self.clip_model.vision_model.embeddings(x)
        x = self.clip_model.vision_model.pre_layrnorm(x)
        for idx, layer in enumerate(self.clip_model.vision_model.encoder.layers):
            if idx<=self.sae_config.block_layer:
                x = layer(x)[0]
        x = self.sae(x[:,0,:])
        return x
    
class new_sae(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros((config.d_in, config.d_sae)), requires_grad=True)
        self.b_enc = nn.Parameter(torch.zeros((config.d_sae)), requires_grad=True)
        self.b_dec = nn.Parameter(torch.zeros((config.d_in)), requires_grad=True)
        
    def forward(self, x):
        x -= self.b_dec
        x = x @ self.W_enc.data
        x += self.b_enc
        x = nn.functional.relu(x)
        return x
    
    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)
        
        

sae_path = "checkpoints/pcy601zk/final_sparse_autoencoder_openai/clip-vit-large-patch14_-2_resid_65536.pt"
loaded_object = torch.load(sae_path)
cfg = loaded_object['cfg']
state_dict = loaded_object['state_dict']
sparse_autoencoder = new_sae(cfg)
sparse_autoencoder.load_my_state_dict(state_dict)
sparse_autoencoder = sparse_autoencoder.to(cfg.device)
partial_model_instance = partial_model(cfg, sparse_autoencoder)
partial_model_instance = partial_model_instance.to(cfg.device)

#Load images:
image = Image.open('dashboard/64081/max_activating/0_0.008484.png')
inputs = partial_model_instance.processor(images=[image], return_tensors="pt", padding = True).to(cfg.device)

output = partial_model_instance(inputs['pixel_values'])
print(inputs['pixel_values'])
input()
print(f'l1 value: {output.sum()}')
print(f'l0 value: {(output>0).sum()}')

#LEAP config
config = {"leap_api_key": "LEAPIE908A240083F2956D8A4CF8B8C0689EB"}

# res = engine.generate(
#     project_name="MATS",
#     model=partial_model_instance,
#     class_list=[str(i) for i in range(1000)],
#     config=config,
#     target_classes=[target_index],
#     samples=None,
#     device='cuda',
#     mode="pt",
# )

