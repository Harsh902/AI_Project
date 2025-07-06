import os
import shutil


def flatten_and_cleanup(train_dir):
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}

    for root, dirs, files in os.walk(train_dir):
        # Skip the root folder itself
        if root == train_dir:
            continue

        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                src_path = os.path.join(root, file)
                dst_path = os.path.join(train_dir, file)

                # Avoid overwriting files with the same name
                if os.path.exists(dst_path):
                    base, ext = os.path.splitext(file)
                    counter = 1
                    while True:
                        new_name = f"{base}_{counter}{ext}"
                        dst_path = os.path.join(train_dir, new_name)
                        if not os.path.exists(dst_path):
                            break
                        counter += 1

                shutil.move(src_path, dst_path)
                print(f"Moved: {src_path} -> {dst_path}")

    # Remove empty subdirectories
    for root, dirs, _ in os.walk(train_dir, topdown=False):
        for d in dirs:
            subfolder = os.path.join(root, d)
            try:
                os.rmdir(subfolder)
                print(f"Deleted empty folder: {subfolder}")
            except OSError:
                pass  # Directory not empty or not deletable


def flatten_and_cleanup_json(train_dir):
    image_extensions = {'.json'}

    for root, dirs, files in os.walk(train_dir):
        # Skip the root folder itself
        if root == train_dir:
            continue

        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                src_path = os.path.join(root, file)
                dst_path = os.path.join(train_dir, file)

                # Avoid overwriting files with the same name
                if os.path.exists(dst_path):
                    base, ext = os.path.splitext(file)
                    counter = 1
                    while True:
                        new_name = f"{base}_{counter}{ext}"
                        dst_path = os.path.join(train_dir, new_name)
                        if not os.path.exists(dst_path):
                            break
                        counter += 1

                shutil.move(src_path, dst_path)
                print(f"Moved: {src_path} -> {dst_path}")

    # Remove empty subdirectories
    for root, dirs, _ in os.walk(train_dir, topdown=False):
        for d in dirs:
            subfolder = os.path.join(root, d)
            try:
                os.rmdir(subfolder)
                print(f"Deleted empty folder: {subfolder}")
            except OSError:
                pass  # Directory not empty or not deletable


# For images
train_path = ("./data/images/train/")
flatten_and_cleanup(train_path)

test_path = ("./data/images/test/")
flatten_and_cleanup(test_path)

val_path = ("./data/images/val/")
flatten_and_cleanup(val_path)

# For labels - no need to move them, since we combine them first, and then convert them to txt and add them to the data/labels folder
train_path = ("./gtBbox_cityPersons/train/")
flatten_and_cleanup_json(train_path)

val_path = ("./gtBbox_cityPersons/val/")
flatten_and_cleanup_json(val_path)

