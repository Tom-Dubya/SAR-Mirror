import os
import random
import shutil
from pathlib import Path


SOURCE_DIR = r"C:\Users\hmull\OneDrive\Documents\CSC2053\B_Correlated"
OUTPUT_DIR = r"C:\Users\hmull\OneDrive\Documents\CSC2053\RESEARCHTHREE"

TRAIN_RATIO = 0.8   # 80% train, 20% test
RANDOM_SEED = 42

def main():
    random.seed(RANDOM_SEED)

    source_path = Path(SOURCE_DIR)
    output_path = Path(OUTPUT_DIR)

    train_path = output_path / "train"
    test_path = output_path / "test"

    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    for class_dir in source_path.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        images = list(class_dir.glob("*.*"))

        random.shuffle(images)

        split_index = int(len(images) * TRAIN_RATIO)
        train_images = images[:split_index]
        test_images = images[split_index:]

        (train_path / class_name).mkdir(parents=True, exist_ok=True)
        (test_path / class_name).mkdir(parents=True, exist_ok=True)

        for img in train_images:
            shutil.copy(img, train_path / class_name / img.name)

        for img in test_images:
            shutil.copy(img, test_path / class_name / img.name)

        print(f"{class_name}: {len(train_images)} train, {len(test_images)} test")

    print("\nDataset successfully split into RESEARCHTHREE/train and RESEARCHTHREE/test")

if __name__ == "__main__":
    main()