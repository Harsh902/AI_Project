import torch
import detectron2
import os

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer

# Register datasets
register_coco_instances("train", {}, "Data/train/_annotations.coco.json", "Data/train")
register_coco_instances("val", {}, "Data/valid/_annotations.coco.json", "Data/valid")

def main():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("Cityscapes/mask_rcnn_R_50_FPN.yaml"))
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ()  # No evaluation or test
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Cityscapes/mask_rcnn_R_50_FPN.yaml")
    cfg.MODEL.DEVICE = 'cpu'

    cfg.SOLVER.IMS_PER_BATCH = 15
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 300
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.OUTPUT_DIR = "./output_custom"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Rename the final model checkpoint
    final_model_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    custom_model_path = os.path.join(cfg.OUTPUT_DIR, "model_final_custom.pth")
    if os.path.exists(final_model_path):
        os.rename(final_model_path, custom_model_path)
        print(f"Model saved as {custom_model_path}")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
