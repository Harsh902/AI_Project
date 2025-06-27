from ultralytics import YOLO

model=YOLO("./runs/detect/train15/weights/best.pt") # or last

# Exporting the default model to TensorRT for citypersons dataset
model.export(format="engine", data="./data.yaml", half=True)

# Instructions to run the model on the validation set will be given once you run the script
# For my model, the code is something like this:
#  yolo val task=detect model=runs/detect/train15/weights/best.engine imgsz=1280 data=./data.yaml half