import streamlit as st
import pandas as pd

# Define a function to render the home page
def home_page():
    st.title('Home Page')
    st.header('Welcome to the App!')

# Define a function to render Subpage 1
def subpage1():
    st.title('Subpage 1')
    st.write("Select a neuron:")
    neurons = [101, 102, 103, 104, 105]  # Example neuron identifiers
    selected_neuron = st.selectbox("", neurons)
    
    # Simulated data for display
    data = pd.DataFrame({
        'Parameter': ['Param1', 'Param2', 'Param3'],
        'Value': [f'{selected_neuron}-1', f'{selected_neuron}-2', f'{selected_neuron}-3']
    })
    st.table(data)
    
    st.header('Additional Title 1')
    st.header('Additional Title 2')

# Define a function to render Subpage 2
def subpage2():
    st.title('Subpage 2')
    st.button("Next image")
    st.header('Additional Title 3')
    st.header('Additional Title 4')

# A simple function to change the page state
def set_page(page_name):
    st.session_state.page = page_name

# Sidebar for navigation
with st.sidebar:
    if st.button("🏠 Home"):
        set_page('home')
    st.text("  ")  # Adding some space before subpage buttons
    if st.button(" ➤ Subpage 1"):
        set_page('subpage1')
    if st.button(" ➤ Subpage 2"):
        set_page('subpage2')

# Define a dictionary linking page names to function renderers
pages = {
    'home': home_page,
    'subpage1': subpage1,
    'subpage2': subpage2
}

# Initialize the session state for page if it's not already set
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Render the current page
pages[st.session_state.page]()
