import argparse

import torch
import torch.onnx
from basicsr.archs.rrdbnet_arch import RRDBNet


def main(args):
    # An instance of the model
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
    )

    KEYNAME = "params_ema"

    model.load_state_dict(torch.load(args.input)[KEYNAME])
    # set the train mode to false since we will only run the forward pass.
    model.train(False)
    model.cpu().eval()

    # An example input
    x = torch.rand(1, 3, 256, 256)
    # Export the model
    with torch.no_grad():
        torch_out = model(x)

        torch.onnx.export(
            model=model,
            args=(x,),
            f="real-esrgan.onnx",
            opset_version=11,
            do_constant_folding=True,
            optimize=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {2: "height", 3: "width"},
                "output": {2: "height_out", 3: "width_out"},
            },
        )

    import numpy as np
    import onnxruntime as ort

    test_input = np.random.rand(1, 3, 64, 64).astype(np.float32)
    session = ort.InferenceSession("real-esrgan.onnx")
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: test_input})[0]

    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")

    print(torch_out.shape)


if __name__ == "__main__":
    """Convert pytorch model to onnx models"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="experiments/pretrained_models/RealESRGAN_x4plus.pth",
        help="Input model path",
    )
    parser.add_argument(
        "--output", type=str, default="realesrgan-x4.onnx", help="Output onnx path"
    )
    parser.add_argument(
        "--params", action="store_false", help="Use params instead of params_ema"
    )
    args = parser.parse_args()

    main(args)
