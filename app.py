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
def preprocess(_img, target_size=(256, 256)):
    orig_size = _img.size
    img = _img.resize(target_size, resample=Image.Resampling.LANCZOS)
    arr = np.array(img).astype(np.float32)
    arr = arr.transpose(2, 0, 1)
    arr = arr[np.newaxis, ...] / 255.0
    return arr, orig_size


@st.cache_data
def postprocess(output, orig_size):
    arr = output.squeeze().transpose(1, 2, 0)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    img = img.resize(orig_size, resample=Image.Resampling.LANCZOS)
    return img


@st.cache_data
def convert_img_to_bytes(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


@st.cache_data
def preprocess_rgba(_img, target_size=(256, 256)):
    orig_size = _img.size
    rgb_img = _img.convert("RGB").resize(target_size, Image.Resampling.LANCZOS)
    alpha = _img.split()[-1].resize(target_size, Image.Resampling.LANCZOS)
    arr = (
        np.array(rgb_img).astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...] / 255.0
    )
    alpha_arr = np.array(alpha).astype(np.float32)[np.newaxis, np.newaxis, ...] / 255.0
    return arr, alpha_arr, orig_size


@st.cache_data
def postprocess_rgba(output, alpha_out, orig_size):
    arr = output.squeeze().transpose(1, 2, 0)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr).resize(orig_size, Image.Resampling.LANCZOS)
    alpha = alpha_out.squeeze()
    alpha = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    alpha_img = Image.fromarray(alpha[0] if alpha.ndim == 3 else alpha).resize(
        orig_size, Image.Resampling.LANCZOS
    )
    img.putalpha(alpha_img)

    return img


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
            input_name = model.get_inputs()[0].name

            with col1:
                st.header("Original Image")
                st.image(original_img)

            with st.spinner("Upscaling image..."):
                if original_img.mode == "RGBA":
                    input_arr, alpha_arr, orig_size = preprocess_rgba(original_img)  # type: ignore
                    output = model.run(None, {input_name: input_arr})[0]
                    alpha_out = alpha_arr
                    sr_img = postprocess_rgba(output, alpha_out, orig_size)
                else:
                    input_arr, orig_size = preprocess(original_img)
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
