# AI_Project

# CityPersons YOLO Training Pipeline

This document outlines the full process of downloading the **CityPersons** dataset, converting its annotations to **YOLO format**, and training/validating the YOLO model.

---

## 📦 Dataset Overview

- **CityPersons**: A pedestrian detection dataset based on the Cityscapes dataset.
- **Annotations**: Includes ground-truth bounding boxes (GTBBox) for pedestrians and other categories.

---

## 📁 Step 1: Download CityPersons Dataset

CityPersons is not freely available. To download:

1. Register at: [https://www.cityscapes-dataset.com/login/](https://www.cityscapes-dataset.com/login/)
2. After login, go to: [https://www.cityscapes-dataset.com/dataset-overview/#citypersons](https://www.cityscapes-dataset.com/dataset-overview/#citypersons)
3. Download the following files:
   - `leftImg8bit_trainvaltest.zip` (Cityscapes images)
   - `gtBbox_cityPersons_trainval.zip` (CityPersons bounding boxes)

4. Extract both zip files:
   ```bash
   unzip leftImg8bit_trainvaltest.zip -d data/
   unzip gtBbox_cityPersons_trainval.zip -d data/

After downloading and extracting the following zip files:

- `leftImg8bit_trainvaltest.zip`
- `gtBbox_cityPersons_trainval.zip`

You should have a structure like: \
data/ \
├── leftImg8bit/ \
│ ├── train/ \
│ │ ├── aachen/ \
│ │ ├── bochum/ \
│ │ └── ... \
│ |── val/ \
│ | ├── frankfurt/ \
│ | ├── münster/ \
│ | └── ... \
│ |── test/ \
│ | ├── berlin/ \
│ | ├── lindau/ \
│ | └── ... \
├── gtBbox_cityPersons/ \
│ ├── train/ \
│ │ ├── aachen/ \
│ │ │ ├── aachen_000000_000019.txt \
│ │ │ └── ... \
│ │ └── ... \
│ |── val/ \
│ | ├── frankfurt/ \
│ | └── ... \

## 🖼️ Image Files

- **Location:** `leftImg8bit/train/<city>/` and `leftImg8bit/val/<city>/`
- **Format:** `.png`
- **Resolution:** `2048 × 1024`
- **Example filename:**

## 🏷️ Annotation Files

- **Location:** `gtBbox_cityPersons/train/<city>/` and `gtBbox_cityPersons/val/<city>/`
- **Format:** `.txt` (NOT YOLO yet)
- **Example filename:**


>[!Note] 
>The test annotations for CityPersons are not public. To evaluate your results, you must follow the instructions mentioned here: \
>https://github.com/cvgroup-njust/CityPersons

## Converting to Yolo format

Ultralytics is the organization responsible for Yolo's developement now. \
You can find more information about them here: https://docs.ultralytics.com/models/yolo11/

To train a Yolo model using the Ultralytics framework, the images must be moved from their respective city folders to the parent folder. \
Additionally, the leftImg8bit is renamed to simply images.
The structure of the repository should look like:
data/ \
├── images/ \
│ ├── train/ \
│ │ ├── aachen_000000_000019_leftImg8bit.png \
│ │ ├── ... \
│ │ ├── bochum_000000_000019_leftImg8bit.png/ \
│ │ └── ... \
│ |── val/ \
│ │ ├── frankfurt_000000_000019_leftImg8bit.png \
│ │ ├── ... \
│ │ ├── muenster_000000_000019_leftImg8bit.png/ \
│ │ └── ... \

We no longer keep the city folders, all images are in their parent folders.
The same process must be done for the labels.
data/ \
├── labels/ \
│ ├── train/ \
│ │ ├── aachen_000000_000019_leftImg8bit.txt \
│ │ ├── ... \
│ │ ├── bochum_000000_000019_leftImg8bit.txt/ \
│ │ └── ... \
│ |── val/ \
│ │ ├── frankfurt_000000_000019_leftImg8bit.txt \
│ │ ├── ... \
│ │ ├── muenster_000000_000019_leftImg8bit.txt/ \
│ │ └── ... \

The complete data directory should resemble this structure:
data/ \
├── images/ \
│ ├── train/ \
│ │ ├── aachen_000000_000019_leftImg8bit.png \
│ │ ├── ... \
│ |── val/ \
│ │ ├── frankfurt_000000_000019_leftImg8bit.png \
│ │ ├── ... \
├── labels/ \
│ ├── train/ \
│ │ ├── aachen_000000_000019_leftImg8bit.txt \
│ │ ├── ... \
│ |── val/ \
│ │ ├── frankfurt_000000_000019_leftImg8bit.txt \
│ │ ├── ... \

AI Project: Pedestrian detection using Yolov11m on CityPersons dataset.

You can install the required libraries through the requiremetns.txt and then export the models
from the yolo page.
To export the model, run the export_model.py

# Testing the model

To generate the predctions for a yolo model on the test set, you can run the following python file:

The file name is _yolo_predict.py_
 
Once you have generated the predictions, they weill be available in a directory like this:
_save_dir: 'runs/detect/predict16'_

They will be of the form:
predict16
-- labels
    -- berlin something something.txt

And each text file would contain the id (0 in this case for person), the relative bbox scores and the confidence 
scroes.

aYou need to convert this to the COCO format, as mentioned here : https://github.com/cvgroup-njust/CityPersons/blob/master/evaluation/readme.txt

and sent the results to Prof. Zhang using your student email.

The files:
detections_test_full.json etc are my files, but their format is not proper. You need to change this to get the proper 
results.

I use this script to convert the results from yolo to Coco:
File name: _convert_test_files.py_
