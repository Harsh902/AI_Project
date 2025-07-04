import time
import cv2
import numpy as np
from openvino.runtime import Core

# === Paths ===
image_path = '/home/dll1305/Documents/ai-project/tf_models/object_detection/quantization/AI_Project/test/munich/munich_000000_000019_leftImg8bit.png'
model_fp32_path = '/home/dll1305/Documents/ai-project/tf_models/object_detection/quantization/openvino_fp32/best.xml'
model_quant_path = '/home/dll1305/Documents/ai-project/tf_models/object_detection/quantization/openvino_int8/best_quant_QUInt8.xml'

# === Preprocess ===
def preprocess(img_path, size=1280):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

input_tensor = preprocess(image_path)

# === Inference helper ===
def run_inference(model_path, input_tensor, repeat=10):
    core = Core()
    model = core.read_model(model_path)
    compiled_model = core.compile_model(model, "CPU")
    input_layer = compiled_model.input(0)

    # Warm-up
    _ = compiled_model([input_tensor])

    # Timed runs
    times = []
    for _ in range(repeat):
        start = time.time()
        _ = compiled_model([input_tensor])
        times.append(time.time() - start)

    return sum(times) / len(times)

# === Measure inference time ===
fp32_time = run_inference(model_fp32_path, input_tensor)
quant_time = run_inference(model_quant_path, input_tensor)

# === Print results ===
print(f"[FP32-OpenVINO]   Avg Inference Time : {fp32_time * 1000:.2f} ms")
print(f"[INT8-OpenVINO]   Avg Inference Time : {quant_time * 1000:.2f} ms")
print(f"Speedup: {fp32_time / quant_time:.2f}x")
