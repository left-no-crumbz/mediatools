from functools import lru_cache
from io import BytesIO
from time import perf_counter
from types import ModuleType
from typing import cast

import numpy as np
import onnxruntime as ort
import streamlit as st
import torch
from PIL import Image
from streamlit.watcher import local_sources_watcher

from strategy import RGBAStrategy, RGBStrategy

original_get_module_paths = local_sources_watcher.get_module_paths


def profiler(func):
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        elapsed = perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f} seconds to execute")
        return result

    return wrapper


# TODO: PNG to SVG
# FIXME: Rectangular images not being processed
# FIXME: Bug where the entire program is re-run if a button or the checkbox is pressed
# TODO: Pad the image instead of resizing it


@lru_cache
def patched_get_module_paths(module: ModuleType) -> set[str]:
    if module.__name__.startswith("torch"):
        return set([])

    return original_get_module_paths(module)


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
    st.image(
        "assets/realesrgan_logo.png",
        use_container_width=True,
    )
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
        col1, col2 = st.columns(2)
        original_img = Image.open(BytesIO(img_bytes))
        input_name = model.get_inputs()[0].name

        print(f"type of input_name: {type(input_name)}")

        with col1:
            st.header("Original Image")
            st.image(original_img)
            st.write(f"Dimensions: {original_img.width} x {original_img.height}")

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

                st.write("📐 Preprocessing the image...")
                # input_arr, alpha_arr, orig_size = preprocess_rgba(img_bytes)  # type: ignore

                input_arr, alpha_arr, orig_size, was_reshaped = cast(
                    tuple[np.ndarray, np.ndarray, tuple[int, int], bool],
                    strategy.preprocess(),
                )

                st.write("🏃 Running the model...")
                # output = model.run(None, {input_name: input_arr})[0]
                output = strategy.run_model(model, input_name, input_arr)

                print(f"Output: {type(output)}")

                alpha_out = alpha_arr
                st.write("✨ Postprocessing the image...")

                sr_img = cast(
                    Image.Image,
                    strategy.postprocess(
                        output,
                        alpha_out,
                        orig_size,
                        was_reshaped,
                        do_retain_size,
                    ),
                )

                status.update(label="🏁 Finished!", state="complete", expanded=False)

            else:
                strategy = cast(RGBStrategy, strategy)

                st.write("📐 Preprocessing the image...")

                input_arr, orig_size, was_reshaped = cast(
                    tuple[np.ndarray, tuple[int, int], bool],
                    strategy.preprocess(),
                )

                st.write("🏃‍♀️ Running the model...")
                # output = model.run(None, {input_name: input_arr})[0]
                output = strategy.run_model(model, input_name, input_arr)

                print(f"Output: {type(output)}")

                st.write("✨ Postprocessing the image...")

                sr_img = strategy.postprocess(
                    output, orig_size, was_reshaped, do_retain_size
                )

                # sr_img = postprocess(output, orig_size)
                status.update(label="🏁 Finished!", state="complete", expanded=False)

        if sr_img:
            with col2:
                st.header("Upscaled Image")
                st.image(sr_img)
                st.write(f"Dimensions: {sr_img.width} x {sr_img.height}")

            buffer = convert_img_to_bytes(sr_img)

            st.download_button(
                label="Download image",
                data=buffer,
                file_name="sr_image.png",
                mime="image/png",
                icon=":material/download:",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
