import torch
import detectron2
import cv2
import os
import json

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# try training on dataset
from detectron2.data.datasets import register_coco_instances
register_coco_instances("train", {}, "Data/train/_annotations.coco.json", "Data/train")
register_coco_instances("val", {}, "Data/valid/_annotations.coco.json", "Data/valid")
register_coco_instances("test", {}, "Data/test/_annotations.coco.json", "Data/test")

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

def main():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("Cityscapes/mask_rcnn_R_50_FPN.yaml"))
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ("test",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Cityscapes/mask_rcnn_R_50_FPN.yaml")
    cfg.MODEL.DEVICE = 'cpu'

    cfg.SOLVER.IMS_PER_BATCH = 15  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final2.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("test", output_dir="./output")
    val_loader = build_detection_test_loader(cfg, "test")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

    test_dir = "Data/test"
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

    # Write to JSON
    with open("detectron2_predictions.json", "w") as f:
        json.dump(prediction_results, f, indent=4)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()



