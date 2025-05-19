import base64
from functools import lru_cache
from types import ModuleType

import streamlit as st
from streamlit.watcher import local_sources_watcher
from streamlit_card import card

st.set_page_config(page_title="MediaTools", page_icon="🛠", layout="wide", initial_sidebar_state="collapsed")


st.markdown("""
<style>
    /* Main theme colors and styling */
    :root {
        --primary-color: #a855f7;
        --primary-light: #c084fc;
        --dark-bg: #030712;
        --card-bg: #1f2937;
        --border-color: #374151;
        --text-color: #f3f4f6;
        --text-secondary: #9ca3af;
    }
    
    /* General styling */
    .stApp {
        background-color: var(--dark-bg);
        color: var(--text-color);
    }
    
    /* Header styling */
    .gradient-text {
        background-image: linear-gradient(to right, #a855f7, #d946ef, #f87171);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        color: transparent;
        font-weight: bold;
        display: inline-block;
    }
    
    /* Card styling */
    .tool-card {
        background-color: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 0;
        transition: all 0.3s ease;
        overflow: hidden;
        height: 100%;
    }
    .tool-card:hover {
        border-color: var(--primary-color);
        box-shadow: 0 0 15px rgba(168, 85, 247, 0.15);
    }
    .card-content {
        padding: 15px;
    }
    .card-image {
        width: 100%;
        aspect-ratio: 16/9;
        object-fit: cover;
    }
    .card-image-container {
        overflow: hidden;
        background-color: #374151;
    }
    .card-image:hover {
        transform: scale(1.05);
        transition: transform 0.3s ease;
    }
    
    /* Badge styling */
    .badge {
        background-color: #374151;
        color: var(--text-secondary);
        padding: 2px 8px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 500;
    }
    
    /* Button styling */
    .custom-button {
        background-color: var(--primary-color);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .custom-button:hover {
        background-color: var(--primary-light);
    }
    .ghost-button {
        background-color: transparent;
        color: var(--text-color);
        border: 1px solid var(--border-color);
    }
    .ghost-button:hover {
        background-color: rgba(255, 255, 255, 0.1);
        border-color: var(--text-secondary);
    }
    
    /* Search bar styling */
    .stTextInput > div > div > input {
        background-color: var(--card-bg);
        border-color: var(--border-color);
        color: var(--text-color);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: var(--card-bg);
        border-color: var(--border-color);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* User avatar styling */
    .user-avatar {
        width: 24px;
        height: 24px;
        border-radius: 50%;
        background-color: #4b5563;
        display: inline-block;
        margin-right: -8px;
        border: 2px solid var(--card-bg);
    }
    
    /* List view styling */
    .list-item {
        display: flex;
        background-color: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        overflow: hidden;
        transition: all 0.3s ease;
        margin-bottom: 12px;
    }
    .list-item:hover {
        border-color: var(--primary-color);
        box-shadow: 0 0 15px rgba(168, 85, 247, 0.15);
    }
    .list-image {
        width: 150px;
        height: 100%;
        object-fit: cover;
    }
    .list-content {
        padding: 15px;
        flex: 1;
    }
</style>
""", unsafe_allow_html=True)


@lru_cache
def patched_get_module_paths(module: ModuleType) -> set[str]:
    if module.__name__.startswith("torch"):
        return set([])
    return original_get_module_paths(module)


original_get_module_paths = local_sources_watcher.get_module_paths
local_sources_watcher.get_module_paths = patched_get_module_paths


@st.cache_data
def read_local_img(filepath: str):
    with open(filepath, "rb") as f:
        data = f.read()
        encoded = base64.b64encode(data)
        data = "data:image/png;base64," + encoded.decode("utf-8")
        return data


if __name__ == "__main__":
    
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.markdown('<h1 class="gradient-text">MediaTools</h1>', unsafe_allow_html=True)
    
    st.divider()
    
    is_upscaler_selected = card(
        title="☝ Image Upscaler",
        text="Upscale your images up to 4x!",
        image=read_local_img("./static/image-upscaler-card-img.png"),
        styles={
            "card": {},
        },
    )

    if is_upscaler_selected:
        st.switch_page("pages/Upscaler.py")
