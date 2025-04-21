from io import BytesIO
from time import perf_counter
from types import ModuleType
from typing import Literal, cast

import numpy as np
import onnxruntime as ort
import streamlit as st
import torch
from PIL import Image
from streamlit.watcher import local_sources_watcher

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


def patched_get_module_paths(module: ModuleType) -> set[str]:
    if module.__name__.startswith("torch"):
        return set([])

    return original_get_module_paths(module)


local_sources_watcher.get_module_paths = patched_get_module_paths


class Strategy:
    def __init__(
        self,
        img_bytes: bytes,
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
    def preprocess(self):
        pass

    # to be abstracted
    def postprocess(self):
        pass


class RGBStrategy(Strategy):
    def preprocess(self) -> tuple[np.ndarray, tuple[int, int], bool]:
        _img = Image.open(BytesIO(self._img_bytes)).convert("RGB")
        orig_size = _img.size

        aspect_ratio = orig_size[0] / orig_size[1]
        is_square = abs(aspect_ratio - 1.0) < 0.01

        if not is_square:
            _img = _img.resize(
                self._target_size,
                self.get_resample_method(self._target_size, orig_size),
            )

        arr = np.array(_img).astype(np.float32)
        arr = arr.transpose(2, 0, 1)
        arr = arr[np.newaxis, ...] / 255.0
        return arr, orig_size, (not is_square)

    def postprocess(
        self,
        model_output: np.ndarray,
        orig_size: tuple[int, int],
        was_reshaped: bool = False,
        do_retain_size: bool = False,
    ) -> Image.Image:
        arr = model_output.squeeze().transpose(1, 2, 0)
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8, copy=False)
        img = Image.fromarray(arr)

        if do_retain_size:
            resample_method = self.get_resample_method(
                (img.width, img.height), orig_size
            )
            img = img.resize(orig_size, resample_method)

        elif was_reshaped:
            upscale_factor = img.width / self._target_size[0]
            upscaled_width = int(orig_size[0] * upscale_factor)
            upscaled_height = int(orig_size[1] * upscale_factor)
            upscaled_size = (upscaled_width, upscaled_height)

            resample_method = self.get_resample_method(
                (img.width, img.height), upscaled_size
            )
            img = img.resize(upscaled_size, resample_method)

            MAX_PIXELS = 178_956_970
            current_pixels = img.width * img.height
            if current_pixels > MAX_PIXELS:
                st.warning("Upscaled image too big. Resizing to a reasonable size.")
                scale = (MAX_PIXELS / current_pixels) ** 0.5
                new_size = (int(img.width * scale), int(img.height * scale))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

        print(f"Original size: {orig_size}, Output image size: {img.size}")
        return img


class RGBAStrategy(Strategy):
    def preprocess(self) -> tuple[np.ndarray, np.ndarray, tuple[int, int], bool]:
        _img = Image.open(BytesIO(self._img_bytes))
        orig_size = _img.size

        aspect_ratio = orig_size[0] / orig_size[1]
        is_square = abs(aspect_ratio - 1.0) < 0.01

        if not is_square:
            resample_method = self.get_resample_method(self._target_size, orig_size)
            rgb_img = _img.convert("RGB").resize(self._target_size, resample_method)
            alpha = _img.split()[-1].resize(self._target_size, resample_method)
        else:
            rgb_img = _img.convert("RGB")
            alpha = _img.split()[-1]

        arr = (
            np.array(rgb_img).astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
            / 255.0
        )
        alpha_arr = (
            np.array(alpha).astype(np.float32)[np.newaxis, np.newaxis, ...] / 255.0
        )
        return arr, alpha_arr, orig_size, (not is_square)

    def postprocess(
        self,
        model_output: np.ndarray,
        alpha_out: np.ndarray,
        orig_size: tuple[int, int],
        was_reshaped: bool = False,
        do_retain_size: bool = False,
    ) -> Image.Image:
        arr = model_output.squeeze().transpose(1, 2, 0)
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

        alpha = alpha_out.squeeze()
        alpha = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
        alpha_img = Image.fromarray(alpha[0] if alpha.ndim == 3 else alpha)

        if do_retain_size:
            resample_method = self.get_resample_method(
                (img.width, img.height), orig_size
            )
            img = img.resize(orig_size, resample_method)
            alpha_img = alpha_img.resize(orig_size, resample_method)
        elif was_reshaped:
            upscale_factor = img.width / self._target_size[0]
            upscaled_width = int(orig_size[0] * upscale_factor)
            upscaled_height = int(orig_size[1] * upscale_factor)
            upscaled_size = (upscaled_width, upscaled_height)

            resample_method = self.get_resample_method(
                (img.width, img.height), upscaled_size
            )

            img = img.resize(upscaled_size, resample_method)
            alpha_img = alpha_img.resize(upscaled_size, resample_method)
        else:
            upscale_factor = img.width / orig_size[0]
            new_alpha_size = (
                int(alpha_img.width * upscale_factor),
                int(alpha_img.height * upscale_factor),
            )
            alpha_img = alpha_img.resize(new_alpha_size, Image.Resampling.LANCZOS)

        MAX_PIXELS = 178_956_970
        current_pixels = img.width * img.height

        if current_pixels > MAX_PIXELS:
            st.warning("Upscaled image too big. Resizing to a reasonable size.")
            scale = (MAX_PIXELS / current_pixels) ** 0.5
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            alpha_img = alpha_img.resize(new_size, Image.Resampling.LANCZOS)

        img.putalpha(alpha_img)
        print(f"Original size: {orig_size}, Output image size: {img.size}")

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

        with col1:
            st.header("Original Image")
            st.image(original_img)
            st.write(f"Dimensions: {original_img.width} x {original_img.height}")

        info_placeholder = st.empty()
        info_placeholder.info("Preparing image for upscaling...")

        with st.spinner("Upscaling image..."):
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

                info_placeholder.info("⚙ Preprocessing the image...")
                # input_arr, alpha_arr, orig_size = preprocess_rgba(img_bytes)  # type: ignore

                input_arr, alpha_arr, orig_size, was_reshaped = cast(
                    tuple[np.ndarray, np.ndarray, tuple[int, int], bool],
                    strategy.preprocess(),
                )

                info_placeholder.info("🏃‍♀️ Running the model...")
                output = model.run(None, {input_name: input_arr})[0]

                print(f"Output: {type(output)}")

                alpha_out = alpha_arr
                info_placeholder.info("⚙ Postprocessing the image...")

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

                info_placeholder.info("🏁 Finished!")

            else:
                strategy = cast(RGBStrategy, strategy)

                info_placeholder.info("⚙ Preprocessing the image...")

                input_arr, orig_size, was_reshaped = cast(
                    tuple[np.ndarray, tuple[int, int], bool],
                    strategy.preprocess(),
                )

                info_placeholder.info("🏃‍♀️ Running the model...")
                output = model.run(None, {input_name: input_arr})[0]

                print(f"Output: {type(output)}")

                info_placeholder.info("⚙ Postprocessing the image...")

                sr_img = strategy.postprocess(
                    output, orig_size, was_reshaped, do_retain_size
                )

                # sr_img = postprocess(output, orig_size)
                info_placeholder.info("🏁 Finished!")

        info_placeholder.empty()

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
