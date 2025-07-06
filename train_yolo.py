from ultralytics import YOLO

model = YOLO("yolo11m.pt")
# Train the model on the COCO8 example dataset for 100 epochs
results12l = model.train(data="data.yaml", epochs=100,
                      imgsz=1280, batch=2,
                      single_cls=True, patience=10, hsv_h=0.05, conf=0.2,hsv_s= 0.8, mixup= 0.2, cutmix= 0.2,
                      hsv_v= 0.7)

