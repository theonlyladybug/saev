import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from sae_training.hooked_vit import HookedVisionTransformer, Hook
from sae_training.config import ViTSAERunnerConfig
from tqdm import tqdm


class ViTActivationsStore:
    """
    Class for streaming tokens and generating and storing activations
    while training SAEs. 
    """
    def __init__(
        self, cfg: ViTSAERunnerConfig, model: HookedVisionTransformer, create_dataloader: bool = True,
    ):
        self.cfg = cfg
        self.model = model
        self.dataset = load_dataset(cfg.dataset_path, split="train", streaming=True)
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Resize((self.cfg.image_width, self.cfg.image_width)),  # Resize the image to WxH pixels
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        ])
        self.iterable_dataset = iter(self.dataset)

        assert self.cfg.image_key in next(self.iterable_dataset).keys(), f'The image key \'{self.cfg.image_key}\' is not valid for this dataset.'
        
        if self.cfg.use_cached_activations:
            """
            Need to implement this. loads stored activations from a file.
            """
            pass
        
        if create_dataloader:
            # fill buffer half a buffer, so we can mix it with a new buffer
            if self.cfg.class_token:
              print("Starting to create the data loader!!!")
              self.dataloader = self.get_data_loader()
              print("Data loader created!!!")
            else:
              """
              Need to implement a buffer for the image patch training.
              """
              pass

    def get_image_batches(self):
        """
        Streams a batch of tokens from a dataset. returns a generator for efficient memory usage.
        """

        batch_size = self.cfg.store_size
        max_batch_size = self.cfg.max_forward_pass_batch_size
        number_of_mini_batches = batch_size//max_batch_size
        remainder = batch_size - number_of_mini_batches*max_batch_size
        device = self.cfg.device
        for mini_batch in range(number_of_mini_batches):
            images=[]
            for batch in range(max_batch_size):
                next_image = next(self.iterable_dataset)[self.cfg.image_key]
                next_image = self.transform(next_image) # next_image is a torch tensor with size [C, W, H].
                images.append(next_image)
            batches = torch.stack(images, dim = 0) # batches has size [batch_size, C, W, H].
            del images
            batches = batches.to(device)
            yield batches
        if remainder>0:
            images=[]
            for batch in range(remainder):
                next_image = next(self.iterable_dataset)[self.cfg.image_key]
                next_image = self.transform(next_image) # next_image is a torch tensor with size [C, W, H].
                images.append(next_image)
            batches = torch.stack(images, dim = 0) # batches has size [batch_size, C, W, H].
            del images
            batches = batches.to(device)
            yield batches

    def get_activations(self, image_batches):
        
        module_name = self.cfg.module_name
        block_layer = self.cfg.block_layer
        list_of_hook_locations = [(block_layer, module_name)]

        activations = self.model.run_with_cache(
            image_batches,
            list_of_hook_locations,
        )[1][(block_layer, module_name)]
        
        if self.cfg.class_token:
          # Only keep the class token
          activations = activations[:,0,:] # See the forward(), foward_head() methods of the VisionTransformer class in timm. 
          # Eg "x = x[:, 0]  # class token" - the [:,0] indexes the batch dimension then the token dimension

        return activations
    
    def get_sae_batches(self):
        image_batches = self.get_image_batches()
        torch.cuda.empty_cache()
        
        sae_batches = []
        for mini_batch in tqdm(image_batches, desc = "Iterating over minibatches to create data loader."):
            self.print_memory()
            sae_batch = self.get_activations(mini_batch)
            torch.cuda.empty_cache()
            sae_batches.append(sae_batch)
            
        sae_batches = torch.cat(sae_batches, dim = 0)
        sae_batches = sae_batches.to(self.cfg.device)
        return sae_batches
        

    def get_data_loader(self) -> DataLoader:
        '''
        Return a torch.utils.dataloader which you can get batches from.
        
        Should automatically refill the buffer when it gets to n % full. 
        (better mixing if you refill and shuffle regularly).
        
        '''
        batch_size = self.cfg.batch_size
        
        sae_batches = self.get_sae_batches()
        
        dataloader = iter(DataLoader(sae_batches, batch_size=batch_size, shuffle=True))
        
        return dataloader
    
    
    def next_batch(self):
        """
        Get the next batch from the current DataLoader. 
        If the DataLoader is exhausted, refill the buffer and create a new DataLoader.
        """
        try:
            # Try to get the next batch
            return next(self.dataloader)
        except StopIteration:
            # If the DataLoader is exhausted, create a new one
            self.dataloader = self.get_data_loader()
            return next(self.dataloader)
        
    def print_memory(self):
        device = self.cfg.device
        total_mem = torch.cuda.get_device_properties(device).total_memory
        allocated_mem = torch.cuda.memory_allocated(device)
        cached_mem = torch.cuda.memory_reserved(device)
        available_mem = total_mem - allocated_mem

        print(f"Total GPU Memory: {total_mem / 1e9} GB")
        print(f"Allocated Memory: {allocated_mem / 1e9} GB")
        print(f"Cached Memory: {cached_mem / 1e9} GB")
        print(f"Available Memory: {available_mem / 1e9} GB")
