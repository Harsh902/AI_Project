import time
from ultralytics import YOLO

# === Paths ===
image_path = '/home/dll1305/Documents/ai-project/tf_models/object_detection/quantization/AI_Project/test/munich/munich_000000_000019_leftImg8bit.png'
model_fp32_path = '/home/dll1305/Documents/ai-project/tf_models/object_detection/quantization/AI_Project/train15/weights/best.onnx'
model_quant_path = '/home/dll1305/Documents/ai-project/tf_models/object_detection/quantization/AI_Project/quantized_models/best_quant_QUInt8.onnx'

import onnxruntime as ort
model_quant = YOLO(model_quant_path, task="detect")
_ = model_quant.predict(source=image_path, save=False, imgsz=640, verbose=False)

session = ort.InferenceSession(model_quant_path, providers=["CPUExecutionProvider"])
model_quant.predictor.session = session

# === Load models ===
model_fp32 = YOLO(model_fp32_path, task="detect")
# model_quant = YOLO(model_quant_path, task="detect")
# model_quant = YOLO(model_quant_path, task="detect", providers=["CPUExecutionProvider"])  # Force CPU

# === Inference timing function ===
def time_inference(model, image_path, warmup=1, repeat=10):
    # Warm-up
    for _ in range(warmup):
        _ = model(image_path)

    # Timed runs
    times = []
    for _ in range(repeat):
        start = time.time()
        _ = model(image_path)
        end = time.time()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    return avg_time

# === Run and compare ===
fp32_time = time_inference(model_fp32, image_path)
quant_time = time_inference(model_quant, image_path)

print(f"[FP32]   Avg Inference Time : {fp32_time * 1000:.2f} ms")
print(f"[QUInt8] Avg Inference Time : {quant_time * 1000:.2f} ms")
print(f"Speedup: {fp32_time / quant_time:.2f}x")
