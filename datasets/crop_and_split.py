"""
Script to crop CUB-200-2011 images using bounding boxes and split into train/test sets.

Expected CUB-200-2011 dataset structure:
- images/
- images.txt
- bounding_boxes.txt
- train_test_split.txt
"""

import os
from PIL import Image
import shutil

def makedir(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def crop_and_split_images(dataset_root, output_root):
    """
    Crop images using bounding boxes and split into train/test sets.

    Args:
        dataset_root: Path to CUB_200_2011 dataset root
        output_root: Path to output directory (e.g., ./datasets/cub200_cropped/)
    """

    # Define paths
    images_txt = os.path.join(dataset_root, 'images.txt')
    bboxes_txt = os.path.join(dataset_root, 'bounding_boxes.txt')
    split_txt = os.path.join(dataset_root, 'train_test_split.txt')
    images_dir = os.path.join(dataset_root, 'images')

    # Check if required files exist
    for file_path in [images_txt, bboxes_txt, split_txt]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")

    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # Create output directories
    train_dir = os.path.join(output_root, 'train_cropped')
    test_dir = os.path.join(output_root, 'test_cropped')
    makedir(train_dir)
    makedir(test_dir)

    # Read images.txt: image_id image_path
    print("Reading images.txt...")
    image_paths = {}
    with open(images_txt, 'r') as f:
        for line in f:
            img_id, img_path = line.strip().split()
            image_paths[int(img_id)] = img_path

    # Read bounding_boxes.txt: image_id x y width height
    print("Reading bounding_boxes.txt...")
    bboxes = {}
    with open(bboxes_txt, 'r') as f:
        for line in f:
            parts = line.strip().split()
            img_id = int(parts[0])
            x, y, width, height = map(float, parts[1:5])
            bboxes[img_id] = (x, y, width, height)

    # Read train_test_split.txt: image_id is_training_image
    print("Reading train_test_split.txt...")
    split_info = {}
    with open(split_txt, 'r') as f:
        for line in f:
            img_id, is_train = line.strip().split()
            split_info[int(img_id)] = int(is_train)

    # Process each image
    print("Processing images...")
    num_train = 0
    num_test = 0
    num_errors = 0

    for img_id in sorted(image_paths.keys()):
        img_path = image_paths[img_id]
        bbox = bboxes[img_id]
        is_train = split_info[img_id]

        # Construct full image path
        full_img_path = os.path.join(images_dir, img_path)

        if not os.path.exists(full_img_path):
            print(f"Warning: Image not found: {full_img_path}")
            num_errors += 1
            continue

        try:
            # Load image
            img = Image.open(full_img_path)
            img = img.convert('RGB')  # Ensure RGB format

            # Crop using bounding box
            x, y, width, height = bbox
            x1 = int(x)
            y1 = int(y)
            x2 = int(x + width)
            y2 = int(y + height)

            # Ensure bbox is within image bounds
            img_width, img_height = img.size
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width, x2)
            y2 = min(img_height, y2)

            cropped_img = img.crop((x1, y1, x2, y2))

            # Determine output directory and create class subdirectory
            # Extract class name from image path (e.g., "001.Black_footed_Albatross/...")
            class_name = img_path.split('/')[0]

            if is_train == 1:
                output_dir = os.path.join(train_dir, class_name)
                num_train += 1
            else:
                output_dir = os.path.join(test_dir, class_name)
                num_test += 1

            makedir(output_dir)

            # Save cropped image
            img_filename = os.path.basename(img_path)
            output_path = os.path.join(output_dir, img_filename)
            cropped_img.save(output_path)

            if (num_train + num_test) % 500 == 0:
                print(f"Processed {num_train + num_test} images...")

        except Exception as e:
            print(f"Error processing {full_img_path}: {e}")
            num_errors += 1
            continue

    print("\n" + "="*60)
    print(f"Processing complete!")
    print(f"Training images: {num_train}")
    print(f"Test images: {num_test}")
    print(f"Errors: {num_errors}")
    print(f"Train output: {train_dir}")
    print(f"Test output: {test_dir}")
    print("="*60)

if __name__ == "__main__":
    # Default paths
    dataset_root = './datasets/CUB_200_2011'
    output_root = './datasets/cub200_cropped'

    print("CUB-200-2011 Image Cropping and Splitting")
    print("="*60)
    print(f"Dataset root: {dataset_root}")
    print(f"Output root: {output_root}")
    print("="*60 + "\n")

    # Allow user to override paths
    import sys
    if len(sys.argv) > 1:
        dataset_root = sys.argv[1]
    if len(sys.argv) > 2:
        output_root = sys.argv[2]

    crop_and_split_images(dataset_root, output_root)
    print("\nDone! You can now run img_aug.py to augment the training images.")

