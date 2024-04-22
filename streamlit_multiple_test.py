import streamlit as st
import pandas as pd
import os
import pickle
import torch
import plotly.express as px
import random
from PIL import Image


expansion_factor = 64

def get_neuron_indices():
    directory = f"web_app_{expansion_factor}/neurons"
    indices=[]
    for name in os.listdir(directory):
        full_path = os.path.join(directory, name)
        # Check if it's a directory
        if os.path.isdir(full_path) and name.isdigit():
            indices.append(int(name))
    random.shuffle(indices)
    return indices

def set_selected_neuron():
    st.session_state.navigator_selected_neuron = st.session_state.navigator_selected_neuron_index
    set_navigator_meta_data()
    set_navigator_image_grid()
    set_navigator_mlp()
    
def set_navigator_meta_data():
    with open(f'web_app_{expansion_factor}/neurons/{st.session_state.navigator_selected_neuron}/meta_data.pkl', 'rb') as file:
        # Load the data from the file
        st.session_state.navigator_meta_data =  pd.DataFrame([pickle.load(file)])

def set_navigator_image_grid():
    st.session_state.navigator_image_grid = Image.open(f'web_app_{expansion_factor}/neurons/{st.session_state.navigator_selected_neuron}/highest_activating_images.png')
        
def set_navigator_mlp():
    tensor = torch.load(f'web_app_{expansion_factor}/neurons/{st.session_state.navigator_selected_neuron}/MLP.pt')
    df = pd.DataFrame({
        'X': range(len(tensor)),
        'Y': tensor.numpy()  # Convert tensor to numpy array
    })
    fig = px.line(df, x='X', y='Y', labels={
            'X': 'MLP index',  # Custom x-axis label
            'Y': 'Cosine similarity'  # Custom y-axis label
        })
    fig.update_layout(
        yaxis=dict(range=[-0.3, 0.6])  # Set the y-axis range
    )
    st.session_state.navigator_mlp = fig
    
    

    

# Define a function to render the home page
def home_page():
    st.title('Home Page')
    st.header('Welcome to the App!')

# Define a function to render Subpage 1
def navigator():
    st.title('Neuron navigator')
    
    if 'navigator_selected_neuron_index' not in st.session_state:
        st.session_state.navigator_selected_neuron_index = st.session_state.navigator_current_neuron_indices[0]
        st.session_state.navigator_selected_neuron = st.session_state.navigator_selected_neuron_index
    
    if 'navigator_meta_data' not in st.session_state:
        set_navigator_meta_data()
    
    if 'navigator_image_grid' not in st.session_state:
        set_navigator_image_grid()
    
    if 'navigator_mlp' not in st.session_state:
        set_navigator_mlp()
    
    selected_neuron = st.selectbox("Select a neuron:", st.session_state.navigator_current_neuron_indices, on_change=set_selected_neuron)
    col1, col2, col3, col4= st.columns(4, gap="small")
    if col1.button("Previous neuron", use_container_width=True): # Use on_click...?
        navigator_previous_neuron()
        
    if col2.button("Next neuron", use_container_width=True):
        navigator_next_neuron()
        
    if col3.button("Entropy > 0", use_container_width=True):
        navigator_positive_entropy()
        
    if col4.button("Reset", use_container_width=True):
        navigator_reset_entropy()
    
    
    # Simulated data for display
    st.header("Meta data")
    st.dataframe(st.session_state.navigator_meta_data, hide_index=True, use_container_width=True)
    st.header('Top 16 highest activating images')
    st.image(st.session_state.navigator_image_grid, use_column_width=True)
    st.header('Neuron alignment')
    st.plotly_chart(st.session_state.navigator_mlp)
    

# Define a function to render Subpage 2
def game():
    st.title('Guess the input image!')
    st.button("Next image")
    st.header('SAE activations')
    st.header('Top SAE features')

# A simple function to change the page state
def set_page(page_name):
    st.session_state.page = page_name

# Sidebar for navigation
with st.sidebar:
    if st.button("🏠 Home"):
        set_page('home')
    st.text("  ")  # Adding some space before subpage buttons
    st.text("  ")  # Adding some space before subpage buttons
    st.text("  ")  # Adding some space before subpage buttons
    if st.button(" 🔎 Neuron navigator"):
        st.session_state.navigator_all_neuron_indices = get_neuron_indices()
        st.session_state.navigator_current_neuron_indices = st.session_state.navigator_all_neuron_indices
        set_page('navigator')
    if st.button(" 🎮 Guess the input image"):
        set_page('game')

# Define a dictionary linking page names to function renderers
pages = {
    'home': home_page,
    'navigator': navigator,
    'game': game
}

# Initialize the session state for page if it's not already set
if 'page' not in st.session_state:
    st.session_state.page = 'home'
    
if 'entropy' not in st.session_state:
    st.session_state.entropy = torch.load(f'web_app_{expansion_factor}/neurons/entropy.pt')

if st.session_state.page == 'navigator' and 'navigator_all_neuron_indices' not in st.session_state: # Included as a santiy check. Should be set when the navigator button is pressed.
    st.session_state.navigator_all_neuron_indices = get_neuron_indices()
    st.session_state.navigator_current_neuron_indices = st.session_state.navigator_all_neuron_indices

# Render the current page
pages[st.session_state.page]()
