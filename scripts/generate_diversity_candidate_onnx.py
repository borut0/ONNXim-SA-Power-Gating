#!/usr/bin/env python3
"""Generate strictly auditable ONNX workload-diversity candidates.

No pretrained weights are downloaded: topology and tensor shapes, rather than
trained values, determine ONNXim's architectural execution.
"""

import argparse
import json
import os
from pathlib import Path

import onnxruntime as ort
import torch
import torchvision.models as tv_models


class GeluMlp(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(1024, 4096),
            torch.nn.GELU(),
            torch.nn.Linear(4096, 4096),
            torch.nn.GELU(),
            # A bias-free output projection avoids a standalone ONNX Add;
            # hidden affine layers and both nonlinearities remain intact.
            torch.nn.Linear(4096, 1024, bias=False),
        )

    def forward(self, value):
        return self.layers(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("vit_b_16", "mlp_gelu"))
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("ONNXIM_HOME", Path.cwd())),
    )
    args = parser.parse_args()

    if args.model == "vit_b_16":
        # PyTorch's inference fast path lowers MultiheadAttention to the
        # non-exportable aten::_native_multi_head_attention operator.  Turning
        # off only that optimization exposes the equivalent primitive graph to
        # the ONNX exporter; it does not change the model architecture.
        torch.backends.mha.set_fastpath_enabled(False)
        model = tv_models.vit_b_16(weights=None).eval()
        sample = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    else:
        model = GeluMlp().eval()
        sample = torch.zeros((1, 32, 1024), dtype=torch.float32)

    model_dir = args.root / "models" / args.model
    model_dir.mkdir(parents=True, exist_ok=True)
    raw_path = model_dir / f"{args.model}.raw.onnx"
    output_path = model_dir / f"{args.model}.onnx"

    with torch.no_grad():
        torch.onnx.export(
            model,
            sample,
            raw_path,
            export_params=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=17,
        )

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.optimized_model_filepath = str(output_path)
    ort.InferenceSession(str(raw_path), sess_options=options)
    raw_path.unlink()

    model_lists = args.root / "model_lists"
    model_lists.mkdir(parents=True, exist_ok=True)
    model_list = {
        "models": [{"name": args.model, "batch_size": 1, "request_time": 0}]
    }
    (model_lists / f"{args.model}_1.json").write_text(
        json.dumps(model_list, indent=2) + "\n"
    )
    print(output_path)


if __name__ == "__main__":
    main()
