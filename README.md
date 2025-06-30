# AI_Project
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