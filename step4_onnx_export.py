"""
Step 4 – ONNX Export + CPU Inference Benchmark
===============================================
Exports best.pt → best.onnx and measures inference time per image.
"""

import time, cv2, numpy as np
from pathlib import Path
from ultralytics import YOLO

BASE    = Path(__file__).parent
BEST_PT = BASE / "runs" / "yolov8n_320" / "weights" / "best.pt"
ONNX_OUT= BASE / "runs" / "yolov8n_320" / "weights" / "best.onnx"
IMG_DIR = BASE / "metal_nut" / "test" / "bent"   # sample class for timing
IMGSZ   = 320
N_WARMUP= 5
N_BENCH = 50

def main():
    # 1. Export
    print("Exporting to ONNX...")
    model = YOLO(str(BEST_PT))
    export_path = model.export(format="onnx", imgsz=IMGSZ, simplify=True, opset=12)
    print(f"Exported to: {export_path}\n")

    # 2. Benchmark with onnxruntime
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1   # single-thread = realistic edge scenario
    sess = ort.InferenceSession(str(export_path),
                                sess_options=sess_opts,
                                providers=["CPUExecutionProvider"])

    inp_name = sess.get_inputs()[0].name
    print(f"ONNX input : {inp_name}  shape={sess.get_inputs()[0].shape}")
    print(f"Image size : {IMGSZ}x{IMGSZ}")
    print(f"Threads    : 1 (single-thread, edge parity)\n")

    # Load a real image for timing
    imgs = sorted(IMG_DIR.glob("*.png"))
    raw  = cv2.imread(str(imgs[0]))
    raw  = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    inp  = cv2.resize(raw, (IMGSZ, IMGSZ)).astype(np.float32) / 255.0
    inp  = inp.transpose(2,0,1)[None]   # NCHW

    # Warm-up
    for _ in range(N_WARMUP):
        sess.run(None, {inp_name: inp})

    # Timed runs
    times = []
    for _ in range(N_BENCH):
        t0 = time.perf_counter()
        sess.run(None, {inp_name: inp})
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    print("=" * 45)
    print(f"CPU Inference Benchmark ({N_BENCH} runs, 1 thread)")
    print(f"  Hardware : Apple M1 (CPU only)")
    print(f"  Image sz : {IMGSZ}x{IMGSZ}  (3-channel RGB float32)")
    print("-" * 45)
    print(f"  Mean     : {times.mean():.2f} ms")
    print(f"  Median   : {np.median(times):.2f} ms")
    print(f"  Std      : {times.std():.2f} ms")
    print(f"  Min      : {times.min():.2f} ms")
    print(f"  Max      : {times.max():.2f} ms")
    print(f"  FPS      : {1000/times.mean():.1f}")
    print("=" * 45)
    print(f"\nONNX file size: {Path(export_path).stat().st_size/1e6:.1f} MB")

if __name__ == "__main__":
    main()
