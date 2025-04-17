from io import BytesIO
from types import ModuleType

import numpy as np
import onnxruntime as ort
import streamlit as st
import torch
from PIL import Image
from streamlit.watcher import local_sources_watcher

original_get_module_paths = local_sources_watcher.get_module_paths


def patched_get_module_paths(module: ModuleType) -> set[str]:
    if module.__name__.startswith("torch"):
        return set([])

    return original_get_module_paths(module)


local_sources_watcher.get_module_paths = patched_get_module_paths


# TODO: Range slider for upscale size
# TODO: Convert pytorch model to ONNX


@st.cache_resource
def load_model():
    try:
        if torch.cuda.is_available():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            device = torch.device("cuda")
        else:
            providers = ["CPUExecutionProvider"]
            device = torch.device("cpu")
        session = ort.InferenceSession("real-esrgan.onnx", providers=providers)

        return device, session
    except RuntimeError as e:
        print(f"{e}")
        if "CUDA" in str(e):
            device = torch.device("cpu")
            session = ort.InferenceSession("real-esrgan.onnx")
            return device, session
        raise e


@st.cache_data
def preprocess(img, target_size=(256, 256)):
    orig_size = img.size
    img = img.convert("RGB").resize(target_size, resample=Image.Resampling.LANCZOS)
    arr = np.array(img).astype(np.float32)
    arr = arr.transpose(2, 0, 1)
    arr = arr[np.newaxis, ...] / 255.0
    return arr, orig_size


def postprocess(output, orig_size):
    arr = output.squeeze().transpose(1, 2, 0)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    img = img.resize(orig_size, resample=Image.Resampling.LANCZOS)
    return img


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
                input_arr, orig_size = preprocess(rgb_img)
                input_name = model.get_inputs()[0].name
                output = model.run(None, {input_name: input_arr})[0]
                sr_img = postprocess(output, orig_size)

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
