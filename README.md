# AI_Project

# CityPersons YOLO Training Pipeline

This document outlines the full process of downloading the **CityPersons** dataset, converting its annotations to **YOLO format**, and training/validating the YOLO model.
License: Apache License, Version 2.0
---

## Dataset Overview

- **CityPersons**: A pedestrian detection dataset based on the Cityscapes dataset.
- **Annotations**: Includes ground-truth bounding boxes (GTBBox) for pedestrians and other categories.

---

## Step 0: Basic Setup

1. Python required: >=3.11.11
2. Create a new project in your IDE of choice.
3. Create a virtual environment in your project: https://docs.python.org/3/library/venv.html
4. Clone this project
5. Install the requirements from the requirements.txt : `pip install -r /path/to/requirements.txt`

## Step 1: Download CityPersons Dataset

To download Citypersons:

1. Register at: [https://www.cityscapes-dataset.com/login/](https://www.cityscapes-dataset.com/login/)
2. After login, go to: [https://www.cityscapes-dataset.com/dataset-overview/#citypersons](https://www.cityscapes-dataset.com/dataset-overview/#citypersons)
3. Download the following files:
   - `leftImg8bit_trainvaltest.zip` (Cityscapes images)
   - `gtBbox_cityPersons_trainval.zip` (CityPersons bounding boxes)

4. Extract both zip files (use the commands for your respective OS!):
   ```bash
   unzip leftImg8bit_trainvaltest.zip -d ./
   unzip gtBbox_cityPersons_trainval.zip -d .

After downloading and extracting the following zip files:

- `leftImg8bit_trainvaltest.zip`
- `gtBbox_cityPersons_trainval.zip`

You should have a structure like: \
project/ \
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
│ │ │ ├── aachen_000000_000019_gtBboxCityPersons.json \
│ │ │ └── ... \
│ │ └── ... \
│ |── val/ \
│ | ├── frankfurt/ \
│ | └── ... 

>[!Note] 
>The test annotations for CityPersons are not public. To evaluate your results, you must follow the instructions mentioned here: \
>https://github.com/cvgroup-njust/CityPersons

## Step 2: Converting to Yolo format

Ultralytics is the organization responsible for Yolo's developement now. \
You can find more information about them here: https://docs.ultralytics.com/models/yolo11/

To train a Yolo model using the Ultralytics framework, the images must be moved from their respective city folders to the parent folder. 
- Create a new directory called `data`
- Create 2 subdirectories in `data` called `images` and `labels`
- Create 2 subdirectories in `images`, `train` and `valid`
- Move the `train` and `valid` directories from _leftImg8bit_ folder to the _data/images_ folder
- Once this is done, you can run the following script: _move_resources.py_. 
- This will move the images and the labels to their parent folders

The structure of the repository should look like: \
data/ \
├── images/ \
│ ├── train/ \
│ │ ├── aachen_000000_000019_leftImg8bit.png \
│ │ ├── ... \
│ |── valid/ \
│ │ ├── frankfurt_000000_000019_leftImg8bit.png \
│ │ ├── ... \
├── labels/ \
│ ├── train/ \
│ │ ├── aachen_000000_000019_gtBboxCityPersons.json \
│ │ ├── ... \
│ |── valid/ \
│ │ ├── frankfurt_000000_000019_gtBboxCityPersons.json \
│ │ ├── ... 

Now, you need to convert the `json` file to `txt` format for yolo. \
For this, run the `convert_json_to_txt.py` file \
Now, your project should have a data directory that looks like: \
data/ \
├── images/ \
│ ├── train/ \
│ │ ├── aachen_000000_000019_leftImg8bit.png \
│ │ ├── ... \
│ |── valid/ \
│ │ ├── frankfurt_000000_000019_leftImg8bit.png \
│ │ ├── ... \
├── labels/ \
│ ├── train/ \
│ │ ├── aachen_000000_000019_gtBboxCityPersons.txt \
│ │ ├── ... \
│ |── valid/ \
│ │ ├── frankfurt_000000_000019_gtBboxCityPersons.txt \
│ │ ├── ... 

We still need to rename the labels to match the image names! \
For this, you can run the _rename_labels.py_ file \
Once this is alll done, the data directory would look like: \
data/ \
├── images/ \
│ ├── train/ \
│ │ ├── aachen_000000_000019_leftImg8bit.png \
│ │ ├── ... \
│ |── valid/ \
│ │ ├── frankfurt_000000_000019_leftImg8bit.png \
│ │ ├── ... \
├── labels/ \
│ ├── train/ \
│ │ ├── aachen_000000_000019_leftImg8bit.txt \
│ │ ├── ... \
│ |── valid/ \
│ │ ├── frankfurt_000000_000019_leftImg8bit.txt \
│ │ ├── ... 

Now, you must create a file called data.yaml, which should have the following structure:
```yaml
path: /abosolute/path/to/data
train: images/train
val: images/valid

nc: 1
names:
  0 : person
```

## Step 3: Training the model
Now you can train the yolo model by running the `train_yolo.py` script. \
This will train a Yolo11m model by default, but you can change it to any of the models mentioned below:
- yolo11n
- yolo11s
- yolo11l
- yolo12n
- yolo12s
- yolo12m
- yolo12l
- rt-detr-l
- rt-detr-x

Where: \
n - nano \
s - small \
m - medium \
l - large \
rt-detr: real time detection transformer \
rt-detr-l : real time detection transformer large \
rt-detr-x : real time detection transformer extra large 

The model and its results will be saved automatically in a folder called _runs_. \
This folder will be generated automatically when you run the training for the first time

## Step 3: Quantizing the model
Ultralytics has a built-in framework for exporting the models. \
You can read more about it here: https://docs.ultralytics.com/modes/export/ 

For quantization, this documentation can be used. \
Models can be exported from the `.pt` format to `tflite`, `onnx`, `tfedgetpu` etc. \
Exporting to these formats also performs quantization, because they are all optimized for edge devices.

To export the model you can run the _export_model.py_, which looks like this:

````python
from ultralytics import YOLO

# Add the location to whatever folder your model is in (will be given by the framework)
model=YOLO("./runs/detect/train15/weights/best.pt") # or last

# change the format according to your needs
model.export(format="engine", data="./data.yaml", half=True, imgsz=1280)
````

## Step 4: Running predictions
Ultralytics has a built in framework for running predictions from  the models. \
You can read more about it here: https://docs.ultralytics.com/modes/predict/ 

To generate predictions from the model you can run the _yolo_predict.py_, which looks like this:

````python
from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("./runs/detect/train65/weights/best.pt")

# Define path to directory containing images for inference
source = "./dataset/images/test/"

# Run batched inference on a list of images
results = model(source, save_txt=True, save_conf=True, stream=True)  # return a list of Results objects

# Process results list
for result in results:
    print(result)


````
Author:
Harsh Amit Doshi.
