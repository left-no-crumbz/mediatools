import streamlit as st
from streamlit_card import card

st.set_page_config(page_title="FrameForge", page_icon="🔧")


if __name__ == "__main__":
    is_upscaler_selected = card(
        title="Upscaler",
        text="Upscale your images up to 4x!",
    )

    if is_upscaler_selected:
        st.switch_page("pages/1_☝_Upscaler.py")
