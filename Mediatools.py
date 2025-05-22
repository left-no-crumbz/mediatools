import base64
from functools import lru_cache
from types import ModuleType

import streamlit as st
from streamlit.watcher import local_sources_watcher
import pandas as pd

@st.cache_data
def get_tools():
    return pd.read_json("tools.json")


@lru_cache(maxsize=1)
def get_css_content():
    with open("styles/style.css", "r") as f:
        return f.read()

def inject_css():
    st.html(f"<style>{get_css_content()}</style>")

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


def create_tool_card(tool: pd.Series) -> str:
    if pd.notna(tool.get('page')):
        return f"""
        <a href="/{tool['page']}" class="tool-card-link">
            <div class="tool-card">
                <div class="card-image-container">
                    <img src="{tool['image']}" class="card-image" alt="{tool['name']}">
                </div>
                <div class="card-content">
                    <div class="card-header">
                        <span class="badge">{tool['category']}</span>
                        <span class="card-emoji">{tool['emoji']}</span>
                    </div>
                    <div class="card-body">
                        <h3>{tool['name']}</h3>
                        <p>{tool['description']}</p>
                    </div>
                </div>
            </div>
        </a>
        """
    else:
        return f"""
        <div class="tool-card">
            <div class="card-image-container">
                <img src="{tool['image']}" class="card-image" alt="{tool['name']}">
            </div>
            <div class="card-content">
                <div class="card-header">
                    <span class="badge">{tool['category']}</span>
                    <span class="card-emoji">{tool['emoji']}</span>
                </div>
                <div class="card-body">
                    <h3>{tool['name']}</h3>
                    <p>{tool['description']}</p>
                </div>
            </div>
        </div>
        """

def create_list_item(tool: dict) -> str:
    if 'page' in tool and tool['page']:
        return f"""
        <a href="/{tool['page']}" class="list-item-link">
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
                </div>
            </div>
        </a>
        """
    else:
        return f"""
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
            </div>
        </div>
        """

def create_tools_grid(tools_df: pd.DataFrame) -> str:
    cards_html = "".join(create_tool_card(tool._asdict()) for tool in tools_df.itertuples())
    return f'<div class="tools-grid">{cards_html}</div>'

def create_tools_list(tools_df: pd.DataFrame) -> str:
    items_html = "".join(create_list_item(tool._asdict()) for tool in tools_df.itertuples())
    return f'<div class="tools-list">{items_html}</div>'

if __name__ == "__main__":
    st.set_page_config(page_title="MediaTools", page_icon="🛠", layout="wide", initial_sidebar_state="collapsed")
    
    tools = get_tools()
    inject_css()
    
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
        st.html(create_tools_grid(tools))
        is_clicked = st.button("Click Me!")
        if is_clicked:
            st.switch_page("pages/upscaler.py")
    else:
        st.html(create_tools_list(tools))