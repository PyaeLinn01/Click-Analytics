import os
from PIL import Image
import streamlit as st

def set_page_config():
    # Construct the path to the logo.png file in the assets folder
    logo_path = os.path.join("assets", "logo.png")
    page_icon = Image.open(logo_path)
    st.set_page_config(layout="centered", page_title="Click Analyst", page_icon=page_icon)

# Call the function
set_page_config()