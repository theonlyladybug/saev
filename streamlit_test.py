import streamlit as st
import os
import random
import PIL.Image as Image
from load_sae_images_acts import load_random_images_and_activations
import numpy as np
import torch
import random
import plotly.express as px


def list_contents(path):
    """List directories and .png files in the given path"""
    try:
        # List directory contents
        dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        # List .png files
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.png')]
        dirs.sort()
        files.sort()
        return dirs, files
    except PermissionError:
        # Return empty lists if permission is denied
        return [], []

def app_navigation(dirs):
    """App navigation logic including displaying .png files in a grid"""
    if 'current_path' not in st.session_state:
        st.session_state.current_path = dirs[5]  # Default to first directory

    selected_path = st.selectbox("Select directory", options=dirs, index=dirs.index(st.session_state.current_path))
    st.session_state.current_path = selected_path

    subdirs, png_files = list_contents(main_directory + "/" + st.session_state.current_path + "/"+ sub_direcotry)

    # Display .png files in a grid
    if png_files:
        cols = st.columns(3)  # Adjust the number of columns for your grid
        for idx, file in enumerate(png_files):
            with cols[idx % 3]:  # Adjust the modulus for the number of columns
                img = Image.open(os.path.join(main_directory + "/" + st.session_state.current_path + "/" + sub_direcotry, file))
                st.image(img, caption=file, use_column_width=True)
                

main_directory = 'dashboard'
sub_direcotry = 'max_activating'
sae_path = ""
model_name = "openai/clip-vit-large-patch14"
layer = -2
location = "residual stream"
number_of_images_generated = 500
# Displays png files in dashboard/feature_idx/test directory

directories = list_contents(main_directory)[0]

if 'list_of_images_and_activations' not in st.session_state:
    st.session_state.list_of_images_and_activations = load_random_images_and_activations(sae_path, number_of_images_generated)

st.markdown("""
<style>
.custom-h1-style {
    font-family: "Inter", sans-serif;
    color: var(--text-color);
    text-align: center;
}
</style>

# <div class="custom-h1-style">ViT SAE features</div>
""", unsafe_allow_html=True)

st.text(f"Model name: {model_name}")
st.text(f"Layer: {layer}")
st.text(f"Location: {location}")

st.header('SAE Feature Navigator')

app_navigation(directories)

st.header('SAE Features On An Input Image')


# Function to convert a PyTorch tensor to a PIL Image
def tensor_to_pil(tensor):
    # Convert PyTorch tensor to numpy array
    # The tensor is in the shape [C, H, W] and needs to be converted to [H, W, C] for PIL
    # Also, ensure the tensor is on CPU and detach it from the computation graph
    np_image = tensor.cpu().detach().numpy()
    np_image = np.transpose(np_image, (1, 2, 0))
    
    # Handle grayscale images (C = 1)
    if np_image.shape[2] == 1:
        np_image = np_image.squeeze(axis=2)
    
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray((np_image * 255).astype(np.uint8))
    return pil_image

def load_images(image_paths):
    """Load images from the given list of image paths."""
    images = []
    for image_path in image_paths:
        image = Image.open(image_path)
        images.append(image)
    return images

def display_images_in_grid(images, num_columns=3):
    """Display images in a grid with the specified number of columns."""
    num_images = len(images)
    num_rows = num_images // num_columns + int(num_images % num_columns > 0)
    
    for i in range(num_rows):
        cols = st.columns(num_columns)
        for j in range(num_columns):
            index = i * num_columns + j
            if index < num_images:
                with cols[j]:
                    st.image(images[index], use_column_width=True)
                    
def display_dashboard(transformed_image, activations):
    # Display the image
    st.image(transformed_image, use_column_width=True)
    fig = px.line(
        activations.detach().cpu(),
    )
    st.plotly_chart(fig)
    vals, inds = torch.topk(activations.detach().cpu(), 5)
    for val, ind in zip(vals,inds):
        st.text(f"\nFeature {ind}:")
        feature_path = f"./dashboard/{ind}/max_activating"
        maes_file_names = os.listdir(feature_path)
        paths = [f"{feature_path}/{i}" for i in maes_file_names]
        images = load_images(paths)
        # Display images in a grid
        display_images_in_grid(images, num_columns=3)
        

if st.button('Generate random image'):
    random_index = random.randint(0, number_of_images_generated)
    image_tensor = st.session_state.list_of_images_and_activations[0][random_index]
    activations = st.session_state.list_of_images_and_activations[1][random_index]
    # Convert tensor to PIL Image
    transformed_image = tensor_to_pil(image_tensor)
    display_dashboard(transformed_image, activations)



    


