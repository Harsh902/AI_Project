# AI Project Tiny ML 2

# CityPersons and INRIA YOLO Training Pipeline + Jetson Execution Instructions

This document outlines the full process of downloading the **CityPersons** and **INRIA** datasets, converting its annotations to **YOLO format**, and training/validating the YOLO model and how to get our results on the Jetson.
---

License: Apache Version 2.0
## Dataset Overview

- **CityPersons**: A pedestrian detection dataset based on the Cityscapes dataset.
- **INRIA**: The INRIA Person dataset is a widely used benchmark dataset for pedestrian detection, particularly for autonomous driving and related research.

## Detection Metrics
The object detection metrics from Yoloa are:
Box(P, R, mAP50, mAP50-95): This metric provides insights into the model's performance in detecting objects:

P (Precision): The accuracy of the detected objects, indicating how many detections were correct.

R (Recall): The ability of the model to identify all instances of objects in the images.

mAP50: Mean average precision calculated at an intersection over union (IoU) threshold of 0.50. It's a measure of the model's accuracy considering only the "easy" detections.

mAP50-95: The average of the mean average precision calculated at varying IoU thresholds, ranging from 0.50 to 0.95. It gives a comprehensive view of the model's performance across different levels of detection difficulty.

You can find more about it here: https://docs.ultralytics.com/guides/yolo-performance-metrics/#class-wise-metrics

---

## Step 0: Basic Setup

1. Python required: >=3.11.11
2. Create a new project in your IDE of choice.
3. Create a virtual environment in your project: https://docs.python.org/3/library/venv.html
4. Clone this project
5. Install the requirements from the requirements.txt : `pip install -r /path/to/requirements.txt`

> [!Note]
> For INRIA, create a separate project for clarity.

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

## INRIA

# Steps:

There are Yolo annotations available for INRIA out-of-the-box. \
To get them, create an account in Roboflow, and download the following dataset: \
https://universe.roboflow.com/pedestrian-u3qhb/inra/dataset/3

When downloading, select the Yolo11 annotations format.

This will download the dataset in the format required for Yolo, along with the labels and the data.yaml file. 

You can use the training scripts already provided. 
You can also simply create a python file with the following code:

````python
from ultralytics import YOLO

# Add the location to whatever folder your model is in (will be given by the framework)
model=YOLO("yolo11n.pt") # or last

# change the format according to your needs
results = model.train(data="./data.yaml")
````

To check your results on the validation set, you can use the following code. 

````python
from ultralytics import YOLO

# Add the location to whatever folder your model is in (will be given by the framework)
model=YOLO("yolo11n.pt") # or last

# change the format according to your needs
results = model.val(data="./data.yaml")
````

To get the metrics for the test set, you can use the same code as mentioned above, with one slight addition:

````python
from ultralytics import YOLO

# Add the location to whatever folder your model is in (will be given by the framework)
model=YOLO("yolo11n.pt") # or last

# change the format according to your needs
results = model.val(data="./data.yaml", split="test")
````

# Jetson
Lastly, to test and evaluate our models on Jetson, first it needs to be setup following the guide here: \
https://www.waveshare.com/wiki/JETSON-NANO-DEV-KIT

Once it's setup, the code to train, validate and test a model remains the same.
It would look something like this:

````python
from ultralytics import YOLO

# Add the location to whatever folder your model is in (will be given by the framework)
model=YOLO("yolo11n.pt") # or last

# change the format according to your needs
results = model.train(data="./data.yaml", epochs=100, imgsz=640)

trained_model = YOLO("runs/detect/trainXX/wegiths/best.pt") # or last.pt
results = trained_model.val(data="./data.yaml", split="test")

source = ["path/to/an/image.png", "path/to/another/image.png", ...]
trained_model = YOLO("runs/detect/trainXX/wegiths/best.pt") # or last.pt
results = trained_model(source)
for result in results:
   print(result)
````

To calculate the energy, keep the predict function of the model in a separate file, such as _yolo_predict.py_
Then, create a file called _calculate_power.py_, and add the following code to it:

````python
import time
import subprocess
import threading
import psutil
import os

power_readings = []
memory_readings = []

READINGS_LOG = "readings_log.txt"
SUMMARY_LOG = "summary_log.txt"

def read_current(channel=0):
    path = f"/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_current{channel}_input"
    try:
        with open(path, "r") as f:
            microamps = int(f.read().strip())  # µA
        return microamps / 1_000_000  # Convert to A
    except Exception as e:
        print(f"Error reading current: {e}")
        return None

# change the voltage, we had 5, but yours can be different!
def log_power_and_memory(voltage=5.0, interval=0.5, stop_event=None, log_file=READINGS_LOG):
    print("Logging power and memory usage...")

    header_needed = not os.path.exists(log_file)
    with open(log_file, "a") as f:
        if header_needed:
            f.write("Run_ID,Timestamp,Power_W,RAM_Used_MB\n")  # Header with run ID

        run_id = int(time.time())  # Unique ID per run, based on start time

        while not stop_event.is_set():
            current = read_current(0)
            timestamp = time.time()
            power = voltage * current if current is not None else None
            mem_used = psutil.virtual_memory().used / (1024 * 1024)

            if power is not None:
                power_readings.append((timestamp, power))
                memory_readings.append((timestamp, mem_used))
                f.write(f"{run_id},{timestamp:.2f},{power:.3f},{mem_used:.2f}\n")
            else:
                f.write(f"{run_id},{timestamp:.2f},N/A,{mem_used:.2f}\n")

            time.sleep(interval)

def run_yolo_and_monitor():
    global power_readings, memory_readings
    power_readings = []
    memory_readings = []

    stop_event = threading.Event()
    logger_thread = threading.Thread(target=log_power_and_memory, args=(5.0, 0.5, stop_event))

    start_time = time.time()
    logger_thread.start()

    subprocess.run(["python3", "yolo_predict.py"])

    stop_event.set()
    logger_thread.join()
    end_time = time.time()

    duration = end_time - start_time

    with open(SUMMARY_LOG, "a") as f:
        run_id = int(start_time)
        f.write(f"\n--- Run {run_id} Summary ---\n")
        f.write(f"Start Time (Unix): {start_time:.2f}\n")
        f.write(f"Duration: {duration:.2f} seconds\n")

        if power_readings:
            avg_power = sum(p for _, p in power_readings) / len(power_readings)
            energy = avg_power * duration
            f.write(f"Average Power: {avg_power:.3f} W\n")
            f.write(f"Estimated Energy Used: {energy:.2f} Joules\n")
        else:
            f.write("No power readings recorded.\n")

        if memory_readings:
            avg_mem = sum(m for _, m in memory_readings) / len(memory_readings)
            max_mem = max(m for _, m in memory_readings)
            f.write(f"Average RAM Used: {avg_mem:.2f} MB\n")
            f.write(f"Peak RAM Used: {max_mem:.2f} MB\n")
        else:
            f.write("No memory readings recorded.\n")

    print("\nRun complete. Appended to 'readings_log.txt' and 'summary_log.txt'.")

run_yolo_and_monitor()

````
The results are saved to the respective log files.

You have reached the end! Congratulations.
You should be able to:
- train a Yolo model on CityPersons and INRIA dataset
- evaluate its performance
- export it to different formats
- run it on Jetson and check its metrics such as power usage


Author:
Harsh Amit Doshi.
