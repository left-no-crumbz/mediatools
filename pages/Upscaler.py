import sys
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from time import perf_counter
from types import ModuleType
from typing import cast

import streamlit as st
import torch
from PIL import Image
from streamlit.watcher import local_sources_watcher
from streamlit_image_comparison import image_comparison

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)
import onnxruntime as ort # noqa: E402
from utils.strategy import RGBAStrategy, RGBStrategy  # noqa: E402

def profiler(func):
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        elapsed = perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f} seconds to execute")
        return result

    return wrapper


# TODO: Multi-pages for different features like:
# PNG to SVG
# Background remover
# TODO: Pad the image instead of resizing it
# TODO: Allow to download in different formats
# TODO: Slider for model


@lru_cache
def patched_get_module_paths(module: ModuleType) -> set[str]:
    if module.__name__.startswith("torch"):
        return set([])
    return original_get_module_paths(module)


original_get_module_paths = local_sources_watcher.get_module_paths
local_sources_watcher.get_module_paths = patched_get_module_paths


@profiler
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


@profiler
def convert_img_to_bytes(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


@profiler
def main():
    st.set_page_config(page_title="Image Upscaler", page_icon="☝")
    style_heading = "text-align: center; font-size: 4rem;"

    st.html(f"<h1 style='{style_heading}'>☝ Image Upscaler</h1>")

    uploaded_img = st.file_uploader(
        "Please upload a file you want to upscale.",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False,
    )

    do_retain_size = st.checkbox("Retain original size")

    device, model = load_model()

    st.info(f"Using device: {device}")

    if uploaded_img:
        img_bytes = uploaded_img.read()
        original_img = Image.open(BytesIO(img_bytes))
        input_name = model.get_inputs()[0].name

        print(f"type of input_name: {type(input_name)}")

        with st.status("Upscaling image...", expanded=True) as status:
            strategy = (
                RGBStrategy(img_bytes)
                if original_img.mode != "RGBA"
                else RGBAStrategy(img_bytes)
            )

            print(f"Image mode: {original_img.mode}")
            print(
                f"Selected strategy: {'RGBAStrategy' if original_img.mode == 'RGBA' else 'RGBStrategy'}"
            )

            if original_img.mode == "RGBA":
                strategy = cast(RGBAStrategy, strategy)
                sr_img = strategy.run_pipeline(model, input_name, do_retain_size)
            else:
                strategy = cast(RGBStrategy, strategy)
                sr_img = strategy.run_pipeline(model, input_name, do_retain_size)
            status.update(label="🏁 Finished!", state="complete", expanded=False)

        if sr_img:
            image_comparison(
                img1=original_img,  # type: ignore
                img2=sr_img,  # type: ignore
                label1="Original image",
                label2="Upscaled image",
                show_labels=True,
                in_memory=True,
                make_responsive=True,
                starting_position=50,
            )

            col1, col2 = st.columns(2)

            with col1:
                st.write(
                    f"Original dimensions: {original_img.width} x {original_img.height}"
                )
            with col2:
                st.write(f"Upscaled dimensions: {sr_img.width} x {sr_img.height}")

            buffer = convert_img_to_bytes(sr_img)

            st.download_button(
                label="Download image",
                data=buffer,
                file_name=f"{uploaded_img.name.split(".")[0]}-upscaled.png",
                mime="image/png",
                icon=":material/download:",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
