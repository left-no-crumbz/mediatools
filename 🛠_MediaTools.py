import base64
from functools import lru_cache
from types import ModuleType

import streamlit as st
from streamlit.watcher import local_sources_watcher
from streamlit_card import card

st.set_page_config(page_title="MediaTools", page_icon="🛠", layout="wide")


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
    is_upscaler_selected = card(
        title="☝ Image Upscaler",
        text="Upscale your images up to 4x!",
        image=read_local_img("./static/image-upscaler-card-img.png"),
        styles={
            "card": {},
        },
    )

    if is_upscaler_selected:
        st.switch_page("pages/1_☝_Upscaler.py")
