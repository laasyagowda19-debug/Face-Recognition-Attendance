
import cv2
import json
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
STUDENTS_FILE = BASE_DIR / "students.json"

MODEL_FILE = MODELS_DIR / "face_model.yml"
LABELS_FILE = MODELS_DIR / "labels.json"


def main():
    if not STUDENTS_FILE.exists():
        print("Register students first.")
        return

    with STUDENTS_FILE.open("r", encoding="utf-8") as file:
        students = json.load(file)

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    training_images = []
    training_labels = []
    label_map = {}
    next_label = 0

    for reg_no in sorted(students.keys()):
        folder = DATASET_DIR / reg_no

        if not folder.exists():
            print(f"Skipping {reg_no}: image folder missing.")
            continue

        student_images = []

        for path in folder.iterdir():
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue

            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

            if image is None or image.size == 0:
                continue

            image = cv2.resize(image, (200, 200))
            image = cv2.equalizeHist(image)
            student_images.append(image)

        if not student_images:
            print(f"Skipping {reg_no}: no valid images.")
            continue

        label_map[str(next_label)] = reg_no

        for image in student_images:
            training_images.append(image)
            training_labels.append(next_label)

        print(
            f"Loaded {len(student_images)} images for "
            f"{students[reg_no]['name']} ({reg_no})"
        )
        next_label += 1

    if len(label_map) < 2:
        print("Register at least two students with valid face images.")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    recognizer.train(
        training_images,
        np.array(training_labels, dtype=np.int32)
    )
    recognizer.save(str(MODEL_FILE))

    with LABELS_FILE.open("w", encoding="utf-8") as file:
        json.dump(label_map, file, indent=4)

    print("\nTraining completed successfully.")
    print("Students trained:", len(label_map))
    print("Images trained:", len(training_images))
    print("Model saved to:", MODEL_FILE)
    print("Labels saved to:", LABELS_FILE)


if __name__ == "__main__":
    main()
