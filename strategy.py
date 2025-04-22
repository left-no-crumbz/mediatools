from io import BytesIO
from typing import Literal

import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image


class Strategy:
    def __init__(
        self,
        img_bytes: bytes,
        target_size: tuple[int, int] = (256, 256),
    ) -> None:
        self._img_bytes = img_bytes
        self._target_size = target_size

    @st.cache_data
    def get_resample_method(
        _self, size1: tuple[int, int], size2: tuple[int, int]
    ) -> Literal[Image.Resampling.LANCZOS, Image.Resampling.BILINEAR]:
        if size1[0] > size2[0] or size1[1] > size2[1]:
            resample_method = Image.Resampling.LANCZOS
        else:
            resample_method = Image.Resampling.BILINEAR

        return resample_method

    @st.cache_data(show_spinner=False)
    def run_model(
        _self, _model: ort.InferenceSession, input_name: str, input_arr: np.ndarray
    ):
        return _model.run(None, {input_name: input_arr})[0]

    # to be abstracted
    def preprocess(self):
        pass

    # to be abstracted
    def postprocess(self):
        pass

    # to be abstracted
    def run_pipeline(self):
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

    def run_pipeline(
        self,
        model: ort.InferenceSession,
        input_name: str,
        do_retain_size: bool = False,
    ) -> Image.Image:
        st.write("📐 Preprocessing the image...")
        input_arr, orig_size, was_reshaped = self.preprocess()
        st.write("🏃‍♀️ Running the model...")
        output = self.run_model(model, input_name, input_arr)
        st.write("✨ Postprocessing the image...")
        img = self.postprocess(output, orig_size, was_reshaped, do_retain_size)
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

    def run_pipeline(
        self,
        model: ort.InferenceSession,
        input_name: str,
        do_retain_size: bool = False,
    ) -> Image.Image:
        st.write("📐 Preprocessing the image...")
        input_arr, alpha_arr, orig_size, was_reshaped = self.preprocess()
        st.write("🏃‍♀️ Running the model...")
        output = self.run_model(model, input_name, input_arr)
        st.write("✨ Postprocessing the image...")
        img = self.postprocess(
            output, alpha_arr, orig_size, was_reshaped, do_retain_size
        )
        return img
