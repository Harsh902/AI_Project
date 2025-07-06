import os
import json

def convert_json_to_yolo(json_data, img_width, img_height):
    lines = []
    for obj in json_data.get('objects', []):
        if obj.get('label') != "pedestrian":
            continue

        class_id = 0
        x, y, w, h = obj['bbox']
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        w_norm = w / img_width
        h_norm = h / img_height
        line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
        lines.append(line)
    return lines

def match_and_convert(images_dir, json_dir, output_labels_dir):
    os.makedirs(output_labels_dir, exist_ok=True)
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_file in image_files:
        img_base = os.path.splitext(img_file)[0]  # e.g. frankfurt_000000_000294_leftImg8bit

        # Extract the first 3 parts of the image basename separated by "_"
        parts = img_base.split("_")
        shared_id = "_".join(parts[:3])  # frankfurt_000000_000294

        # Find JSON label file matching shared_id and ending with gtBboxCityPersons.json
        possible_labels = [f for f in os.listdir(json_dir)
                           if f.startswith(shared_id) and f.endswith('_gtBboxCityPersons.json')]

        if not possible_labels:
            print(f"No label found for image: {img_file}")
            continue

        json_file = possible_labels[0]
        json_path = os.path.join(json_dir, json_file)

        with open(json_path, 'r') as f:
            json_data = json.load(f)

        yolo_lines = convert_json_to_yolo(
            json_data,
            img_width=json_data['imgWidth'],
            img_height=json_data['imgHeight']
        )

        # Save label file with the full image basename + .txt extension
        label_name = img_base + ".txt"
        label_path = os.path.join(output_labels_dir, label_name)

        with open(label_path, "w") as out_file:
            out_file.write("\n".join(yolo_lines))  # Empty if no pedestrians

        print(f"✓ Created label: {label_name} from {json_file}")

match_and_convert(
    images_dir="./data/images/train",
    json_dir="./gtBbox_cityPersons/labels/train",
    output_labels_dir="./data/labels/train"
)

match_and_convert(
    images_dir="./data/images/valid",
    json_dir="./gtBbox_cityPersons/labels/val",
    output_labels_dir="./data/labels/valid"
)
