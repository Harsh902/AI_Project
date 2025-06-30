from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("./runs/detect/train65/weights/best.pt")

# Define path to directory containing images and videos for inference
source = "./dataset/images/test/"

# Run batched inference on a list of images
results = model(source, save_txt=True, save_conf=True, stream=True)  # return a list of Results objects

# Process results list
for result in results:
    print(result)

