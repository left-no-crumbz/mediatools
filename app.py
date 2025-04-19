import time
from io import BytesIO
from types import ModuleType
from typing import Literal

import numpy as np
import onnxruntime as ort
import streamlit as st
import torch
from PIL import Image
from streamlit.watcher import local_sources_watcher

original_get_module_paths = local_sources_watcher.get_module_paths


def profiler(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f} seconds to execute")
        return result

    return wrapper


def patched_get_module_paths(module: ModuleType) -> set[str]:
    if module.__name__.startswith("torch"):
        return set([])

    return original_get_module_paths(module)


local_sources_watcher.get_module_paths = patched_get_module_paths


class Strategy:
    def __init__(
        self,
        img_bytes: np.ndarray,
        target_size: tuple[int, int] = (256, 256),
    ) -> None:
        self._img_bytes = img_bytes
        self._target_size = target_size

    def get_resample_method(
        self, size1: tuple[int, int], size2: tuple[int, int]
    ) -> Literal[Image.Resampling.LANCZOS, Image.Resampling.BILINEAR]:
        if size1[0] > size2[0] or size1[1] > size2[1]:
            resample_method = Image.Resampling.LANCZOS
        else:
            resample_method = Image.Resampling.BILINEAR

        return resample_method

    # to be abstracted
    def preprocesss(self):
        pass

    # to be abstracted
    def postprocess(self):
        pass


class RGBStrategy(Strategy):
    def preprocesss(self) -> tuple[np.ndarray, tuple[int, int]]:
        _img = Image.open(BytesIO(self._img_bytes))
        orig_size = _img.size

        resample_method = self.get_resample_method(self._target_size, orig_size)

        img = _img.resize(self._target_size, resample_method)
        arr = np.array(img).astype(np.float32)
        arr = arr.transpose(2, 0, 1)
        arr = arr[np.newaxis, ...] / 255.0
        return arr, orig_size

    def postprocess(
        self, orig_size: tuple[int, int], do_retain_size: bool
    ) -> Image.Image:
        arr = self._img_bytes.squeeze().transpose(1, 2, 0)
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8, copy=False)
        out_size = (arr.shape[1], arr.shape[0])

        resample_method = self.get_resample_method(orig_size, out_size)

        if do_retain_size:
            img = Image.fromarray(arr)
            return img

        img = Image.fromarray(arr).resize(orig_size, resample_method)
        return img


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
@st.cache_data
def preprocess(img_bytes, target_size=(256, 256)):
    _img = Image.open(BytesIO(img_bytes))
    orig_size = _img.size

    if target_size[0] > orig_size[0] or target_size[1] > orig_size[1]:
        resample_method = Image.Resampling.LANCZOS
    else:
        resample_method = Image.Resampling.BILINEAR

    img = _img.resize(target_size, resample_method)
    arr = np.array(img).astype(np.float32)
    arr = arr.transpose(2, 0, 1)
    arr = arr[np.newaxis, ...] / 255.0
    return arr, orig_size


@profiler
@st.cache_data
def postprocess(output, orig_size):
    arr = output.squeeze().transpose(1, 2, 0)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8, copy=False)
    out_size = (arr.shape[1], arr.shape[0])

    if orig_size[0] > out_size[0] or orig_size[1] > out_size[1]:
        resample_method = Image.Resampling.LANCZOS
    else:
        resample_method = Image.Resampling.BILINEAR

    img = Image.fromarray(arr).resize(orig_size, resample_method)
    return img


@profiler
def convert_img_to_bytes(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


@profiler
@st.cache_data
def preprocess_rgba(img_bytes, target_size=(256, 256)):
    _img = Image.open(BytesIO(img_bytes))
    orig_size = _img.size

    if target_size[0] > orig_size[0] or target_size[1] > orig_size[1]:
        resample_method = Image.Resampling.LANCZOS
    else:
        resample_method = Image.Resampling.BILINEAR

    rgb_img = _img.convert("RGB").resize(target_size, resample_method)
    alpha = _img.split()[-1].resize(target_size, resample_method)
    arr = (
        np.array(rgb_img).astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...] / 255.0
    )
    alpha_arr = np.array(alpha).astype(np.float32)[np.newaxis, np.newaxis, ...] / 255.0
    return arr, alpha_arr, orig_size


@profiler
@st.cache_data
def postprocess_rgba(output, alpha_out, orig_size):
    arr = output.squeeze().transpose(1, 2, 0)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    out_size = (arr.shape[1], arr.shape[0])

    if orig_size[0] > out_size[0] or orig_size[1] > out_size[1]:
        resample_method = Image.Resampling.LANCZOS
    else:
        resample_method = Image.Resampling.BILINEAR

    img = Image.fromarray(arr).resize(orig_size, resample_method)
    alpha = alpha_out.squeeze()
    alpha = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    alpha_img = Image.fromarray(alpha[0] if alpha.ndim == 3 else alpha).resize(
        orig_size, resample_method
    )
    img.putalpha(alpha_img)

    return img


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

    device, model = load_model()

    st.info(f"Using device: {device}")

    if uploaded_img:
        img_bytes = uploaded_img.read()
        try:
            col1, col2 = st.columns(2)
            original_img = Image.open(BytesIO(img_bytes))
            input_name = model.get_inputs()[0].name

            with col1:
                st.header("Original Image")
                st.image(original_img)

            info_placeholder = st.empty()
            info_placeholder.info("Preparing image for upscaling...")

            with st.spinner("Upscaling image..."):
                if original_img.mode == "RGBA":
                    info_placeholder.info("⚙ Preprocessing the image...")
                    input_arr, alpha_arr, orig_size = preprocess_rgba(img_bytes)  # type: ignore
                    info_placeholder.info("🏃‍♀️ Running the model...")
                    output = model.run(None, {input_name: input_arr})[0]
                    alpha_out = alpha_arr
                    info_placeholder.info("⚙ Postprocessing the image...")
                    sr_img = postprocess_rgba(output, alpha_out, orig_size)
                    info_placeholder.info("🏁 Finished!")

                else:
                    info_placeholder.info("⚙ Preprocessing the image...")
                    input_arr, orig_size = preprocess(img_bytes)
                    info_placeholder.info("🏃‍♀️ Running the model...")
                    output = model.run(None, {input_name: input_arr})[0]
                    info_placeholder.info("⚙ Postprocessing the image...")
                    sr_img = postprocess(output, orig_size)
                    info_placeholder.info("🏁 Finished!")

            time.sleep(3)
            info_placeholder.empty()

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
