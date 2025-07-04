import time
import cv2
import numpy as np
import onnxruntime as ort

# === Paths ===
image_path = '/home/dll1305/Documents/ai-project/tf_models/object_detection/quantization/AI_Project/test/munich/munich_000000_000019_leftImg8bit.png'
model_fp32_path = '/home/dll1305/Documents/ai-project/tf_models/object_detection/quantization/AI_Project/train15/weights/best.onnx'
model_quant_path = '/home/dll1305/Documents/ai-project/tf_models/object_detection/quantization/AI_Project/quantized_models/best_quant_QUInt8.onnx'

# === Preprocess function ===
def preprocess(img_path, size=1280):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0  # CHW and normalize
    img = np.expand_dims(img, axis=0)
    return img

input_tensor = preprocess(image_path)

# === Timing helper ===
def run_inference(session, input_tensor, input_name, warmup=1, repeat=10):
    for _ in range(warmup):
        _ = session.run(None, {input_name: input_tensor})
    times = []
    for _ in range(repeat):
        start = time.time()
        _ = session.run(None, {input_name: input_tensor})
        times.append(time.time() - start)
    return sum(times) / len(times)

# === FP32 ONNX (force CPU) ===
session_fp32 = ort.InferenceSession(model_fp32_path, providers=["CPUExecutionProvider"])
input_name_fp32 = session_fp32.get_inputs()[0].name
fp32_time = run_inference(session_fp32, input_tensor, input_name_fp32)

# === QUInt8 ONNX (force CPU) ===
session_quant = ort.InferenceSession(model_quant_path, providers=["CPUExecutionProvider"])
input_name_quant = session_quant.get_inputs()[0].name
quant_time = run_inference(session_quant, input_tensor, input_name_quant)

print(f"[FP32]   Avg Inference Time : {fp32_time * 1000:.2f} ms")
print(f"[QUInt8] Avg Inference Time : {quant_time * 1000:.2f} ms")
print(f"Speedup: {fp32_time / quant_time:.2f}x")
