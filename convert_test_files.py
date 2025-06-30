import os
import json
from PIL import Image

input_folder = './runs/detect/predict13/labels'
image_folder = './dataset/images/test'  # Folder with original images
output_json = 'detection_mock.json'

dt_coco = []
ndt = 0

# Sort files to keep consistent image_id indexing
all_files = sorted(os.listdir(input_folder))
for image_id, filename in enumerate(all_files, start=1):
    file_path = os.path.join(input_folder, filename)

    if not os.path.isfile(file_path):
        print(file_path)
        continue

    # Get image size
    image_name = os.path.splitext(filename)[0]
    for ext in ['.jpg', '.png', '.jpeg']:
        image_path = os.path.join(image_folder, image_name + ext)
        if os.path.exists(image_path):
            with Image.open(image_path) as img:
                img_width, img_height = img.size
            break
    else:
        print(f"Image not found for label file: {filename}")
        continue

    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = list(map(float, line.split()))
        if len(parts) != 6:
            print(f"Skipping invalid line in {filename}: {line}")
            continue

        class_id, x_rel, y_rel, w_rel, h_rel, score = parts

        # Convert to absolute COCO format
        x_abs = x_rel * img_width
        y_abs = y_rel * img_height
        w_abs = w_rel * img_width
        h_abs = h_rel * img_height

        x_min = x_abs - w_abs / 2
        y_min = y_abs - h_abs / 2

        dt_coco.append({
            'image_id': image_id,
            'category_id': 1,
            'bbox': [x_min, y_min, w_abs, h_abs],
            'score': score
        })
        ndt += 1

# Save to JSON
with open(output_json, 'w') as fp:
    json.dump(dt_coco, fp)

print(f"Saved {ndt} detections to {output_json}")
