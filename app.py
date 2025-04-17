import os
from io import BytesIO
from types import ModuleType

import streamlit as st
import torch
from PIL import Image
from RealESRGAN import RealESRGAN
from streamlit.watcher import local_sources_watcher

original_get_module_paths = local_sources_watcher.get_module_paths


def patched_get_module_paths(module: ModuleType) -> set[str]:
    if module.__name__.startswith("torch"):
        return set([])

    return original_get_module_paths(module)


local_sources_watcher.get_module_paths = patched_get_module_paths


@st.cache_data
def load_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        WEIGHTS_PATH = "weights/RealESRGAN_x4.pth"
        download_weights = not os.path.exists(WEIGHTS_PATH)
        model = RealESRGAN(device, scale=4)
        model.load_weights(WEIGHTS_PATH, download=download_weights)
        return device, model
    except RuntimeError as e:
        print(f"{e}")
        if "CUDA" in str(e):
            device = torch.device("cpu")
            model = RealESRGAN(device, scale=4)
            model.load_weights(WEIGHTS_PATH, download=download_weights)
            return device, model
        raise e


@st.cache_data
def convert_img_to_bytes(img):
    st.cache_data.clear()
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


@st.cache_data
def ensure_rgb(_image):
    st.cache_data.clear()
    if _image.mode == "RGBA":
        background = Image.new("RGB", _image.size, (255, 255, 255))
        result = Image.alpha_composite(background.convert("RGBA"), _image)
        return result.convert("RGB")
    return _image.convert("RGB")


def main():
    st.title("LOCAL ESRGAN")

    uploaded_img = st.file_uploader(
        "Please upload a file you want to upscale.",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False,
    )

    device, model = load_model()

    st.info(f"Using device: {device}")

    if uploaded_img:
        try:
            col1, col2 = st.columns(2)
            original_img = Image.open(uploaded_img)

            with col1:
                st.header("Original Image")
                st.image(original_img)

            with st.spinner("Upscaling image..."):
                rgb_img = ensure_rgb(original_img)
                sr_img = model.predict(rgb_img)

            if sr_img:
                with col2:
                    st.header("Upscaled Image")
                    st.image(sr_img)

                buffer = convert_img_to_bytes(sr_img)

                st.download_button(
                    label="Download image",
                    data=buffer,
                    file_name="sr_image.png",
                    mime="image/png",
                    icon=":material/download:",
                    use_container_width=True,
                )
        except Exception as e:
            print(f"ERROR: {e}")
            st.error(f"Error processing image: {e}")


if __name__ == "__main__":
    main()
