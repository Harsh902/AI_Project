import torch
import detectron2
import cv2
import os
import json

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

def main():
    cfg = get_cfg()

    # Load config and model weights
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.WEIGHTS = "./output/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    test_dir = "./Data/gtFine/test"
    prediction_results = []

    for img_file in os.listdir(test_dir):
        if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(test_dir, img_file)
            image = cv2.imread(img_path)
            outputs = predictor(image)

            instances = outputs["instances"].to("cpu")
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()

            print(f"{img_file} -> {len(instances)} detections")
            for i in range(len(boxes)):
                box = boxes[i]
                prediction_results.append({
                    "image": img_file,
                    "class_id": int(classes[i]),
                    "confidence": float(scores[i]),
                    "bbox": {
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3])
                    }
                })
    print(prediction_results)
        # Save predictions to JSON
    with open("detectron2_predictions.json", "w") as f:
        json.dump(prediction_results, f, indent=4)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()