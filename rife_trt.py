"""RIFE の TensorRT バックエンド

PyTorch の eager 実行より速い。ただしエンジンは
「解像度・timestep 固定」でビルドされるため、条件が合わないときは
呼び出し側で PyTorch にフォールバックする。

エンジンの作り方:
    python -c "import rife_trt; rife_trt.export_onnx(1088, 1920, 0.5)"
    trtexec --onnx=models/rife/rife425_1088x1920.onnx --fp16 \
            --saveEngine=models/rife/rife425_fp16.engine \
            --memPoolSize=workspace:512 --builderOptimizationLevel=1
"""

import sys
from pathlib import Path

_DIR = Path(__file__).parent / "models" / "rife"
_ctx = None
_engine = None
_names = None
_shape = None
_in_shape = None


def engine_path(h=1088, w=1920):
    return _DIR / f"rife425_fp16.engine"


def available(h, w):
    """このサイズのエンジンが使えるか (形状が違えば False)"""
    try:
        import tensorrt  # noqa: F401
    except Exception:
        return False
    if not engine_path(h, w).exists():
        return False
    try:
        _load(h, w)
        return tuple(_in_shape[-2:]) == (h, w)
    except Exception:
        return False


def export_onnx(h=1088, w=1920, timestep=0.5, scale=1.0):
    """IFNet を固定形状の ONNX に出す (TRT ビルドの入力にする)"""
    import torch
    import torch.nn as nn
    sys.path.insert(0, str(_DIR))
    from train_log.IFNet_HDv3 import IFNet

    net = IFNet()
    sd = torch.load(_DIR / "train_log" / "flownet.pkl", map_location="cpu",
                    weights_only=True)
    net.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()},
                        strict=False)
    net.eval()

    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net
            self.sl = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]
            self.t = timestep

        def forward(self, x):
            return self.net(x, self.t, self.sl)[2][-1]

    out = _DIR / f"rife425_{h}x{w}.onnx"
    torch.onnx.export(Wrap().eval(), (torch.randn(1, 6, h, w),), str(out),
                      input_names=["x"], output_names=["y"], opset_version=17,
                      dynamo=False)
    return out


def _load(h, w):
    global _ctx, _engine, _names, _shape
    if _ctx is not None and _shape == (h, w):
        return
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    with open(engine_path(h, w), "rb") as f:
        _engine = runtime.deserialize_cuda_engine(f.read())
    _ctx = _engine.create_execution_context()
    ins, outs = [], []
    for i in range(_engine.num_io_tensors):
        n = _engine.get_tensor_name(i)
        (ins if _engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
         else outs).append(n)
    _names = (ins[0], outs[0])
    global _in_shape
    _in_shape = tuple(_engine.get_tensor_shape(ins[0]))
    _shape = (h, w)


def infer(x):
    """x: (1,6,H,W) float32 の CUDA テンソル → (1,3,H,W) float32"""
    import torch
    h, w = x.shape[-2:]
    _load(h, w)
    x = x.contiguous()
    y = torch.empty((1, 3, h, w), dtype=torch.float32, device=x.device)
    _ctx.set_tensor_address(_names[0], x.data_ptr())
    _ctx.set_tensor_address(_names[1], y.data_ptr())
    _ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    return y
