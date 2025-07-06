import os

def rename_labels_to_match_images(images_dir, labels_dir):
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]  # e.g. aachen_000001_leftImg8bit
        # Try to find corresponding label (any .txt file that starts with the same city + frame)
        for label_file in os.listdir(labels_dir):
            if label_file.endswith('.txt') and label_file.startswith(base_name[:19]):
                old_label_path = os.path.join(labels_dir, label_file)
                new_label_path = os.path.join(labels_dir, base_name + '.txt')

                os.rename(old_label_path, new_label_path)
                print(f"Renamed {label_file} -> {base_name}.txt")
                break

# Example usage:
rename_labels_to_match_images(
    "./data/images/train",
    "./data/labels/train"
)

rename_labels_to_match_images(
    "./data/images/valid",
    "./data/labels/val"
)
