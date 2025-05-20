import base64
from functools import lru_cache
from types import ModuleType

import streamlit as st
from streamlit.watcher import local_sources_watcher
import pandas as pd

tools = pd.read_json("tools.json")

st.set_page_config(page_title="MediaTools", page_icon="🛠", layout="wide", initial_sidebar_state="collapsed")

with open("styles/style.css") as css:
    st.html(f"<style>{css.read()}</style>")

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
    st.markdown('<h3 class="gradient-text" style="margin-left: calc(52% - 50vw);">MediaTools</h3>', unsafe_allow_html=True)
    st.markdown('<div class="full-width-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center;'>Discover the best tools for media creators</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: var(--text-secondary); font-size: 1.4rem;'>A curated collection of powerful tools to enhance your creative workflow</p>", unsafe_allow_html=True)
    
    search_col, filter_col = st.columns([2, 1])
    with search_col:
        search_query = st.text_input("Search", placeholder="Search for tools...", icon=":material/search:", label_visibility="hidden")
    with filter_col:
        category_filter = st.selectbox(
            "Categories",
            ["All Categories", "Image Editing", "Video Editing", "Audio Tools", "Design Tools", "AI Tools"],
            label_visibility="hidden"
        )
    
    if search_query:
        tools = tools[tools['name'].str.contains(search_query, case=False) | 
                        tools['description'].str.contains(search_query, case=False)]
        
    if category_filter != "All Categories":
        tools = tools[tools['category'] == category_filter]
    
    view_sort_cols = st.columns([2, 1], vertical_alignment="center")
    with view_sort_cols[0]:
        st.html("<h2 style='margin: 0; font-weight: 500'>Popular Tools</h2>")
    with view_sort_cols[1]:
        view_type = st.segmented_control("View", [":material/grid_on:", ":material/list:"], label_visibility="hidden", default=":material/grid_on:")
    
    if view_type == ":material/grid_on:":
        # Create a grid layout
        num_cols = 4  # Number of columns in the grid
        
        # Create rows with appropriate number of columns
        for i in range(0, len(tools), num_cols):
            cols = st.columns(num_cols)
            for j in range(num_cols):
                if i + j < len(tools):
                    tool = tools.iloc[i + j]
                    with cols[j]:
                        with st.container():
                            st.markdown(f"""
                            <div class="tool-card">
                                <div class="card-image-container">
                                    <img src="{tool['image']}" class="card-image" alt="{tool['name']}">
                                </div>
                                <div class="card-content">
                                    <span class="badge">{tool['category']}</span>
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px;">
                                        <h3>{tool['name']}</h3>
                                        <span style="font-size: 24px;">{tool['emoji']}</span>
                                    </div>
                                    <p style="color: var(--text-secondary); font-size: 14px; margin-top: 4px;">{tool['description']}</p>
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 12px;">
                                        <button class="custom-button">Try Tool</button>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
    else:  # List view
        for i, tool in tools.iterrows():
            st.markdown(f"""
            <div class="list-item">
                <img src="{tool['image']}" class="list-image" alt="{tool['name']}">
                <div class="list-content">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="display: flex; align-items: center;">
                            <span style="font-size: 24px; margin-right: 8px;">{tool['emoji']}</span>
                            <h3>{tool['name']}</h3>
                        </div>
                        <span class="badge">{tool['category']}</span>
                    </div>
                    <p style="color: var(--text-secondary); font-size: 14px; margin-top: 4px;">{tool['description']}</p>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 12px;">
                        <button class="custom-button">Try Tool</button>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)