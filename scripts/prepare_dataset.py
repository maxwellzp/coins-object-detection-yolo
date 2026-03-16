from pathlib import Path
import random
import shutil

RAW_IMAGES = Path("dataset/raw")
YOLO_EXPORT = Path("dataset/yolo_export")

PROCESSED_DATASET = Path("dataset/processed")

IMAGES_SRC = YOLO_EXPORT / "images"
LABELS_SRC = YOLO_EXPORT / "labels"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1


def create_dirs():
    for dir_name in ["train", "val", "test"]:
        images_dir = PROCESSED_DATASET / "images" / dir_name
        images_dir.mkdir(parents=True, exist_ok=True)

        labels_dir = PROCESSED_DATASET / "labels" / dir_name
        labels_dir.mkdir(parents=True, exist_ok=True)


def get_pairs():
    images = list(RAW_IMAGES.glob("*.jpg"))

    pairs = []
    for img in images:
        label_name = LABELS_SRC / (img.stem + ".txt")

        if label_name.exists():
            pairs.append((img, label_name))
        else:
            print(f"Missing label for {label_name}")

    return pairs


def split_dataset(pairs):
    random.shuffle(pairs)
    total = len(pairs)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train = pairs[:train_end]
    val = pairs[train_end:val_end]
    test = pairs[val_end:]

    return train, val, test


def copy_split(data, split):
    for img, label in data:
        shutil.copy(img, PROCESSED_DATASET / "images" / split / img.name)
        shutil.copy(label, PROCESSED_DATASET / "labels" / split / label.name)


def main():
    random.seed(42)
    print("Preparing dataset...")
    create_dirs()

    pairs = get_pairs()
    print(f"Pairs: {len(pairs)}")

    train, val, test = split_dataset(pairs)

    print("Train:", len(train))
    print("Val:", len(val))
    print("Test:", len(test))

    copy_split(train, "train")
    copy_split(val, "val")
    copy_split(test, "test")

    print("Done")


if __name__ == "__main__":
    main()
