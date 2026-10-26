
import cv2
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "models" / "face_model.yml"
LABELS_FILE = BASE_DIR / "models" / "labels.json"

CASCADE_FILE = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)


class FaceEngine:
    def __init__(self, threshold=65):
        if not MODEL_FILE.exists() or not LABELS_FILE.exists():
            raise FileNotFoundError(
                "Trained model not found. Run python train_model.py first."
            )

        self.detector = cv2.CascadeClassifier(CASCADE_FILE)

        if self.detector.empty():
            raise RuntimeError("Could not load face detector.")

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(str(MODEL_FILE))
        self.threshold = threshold

        with LABELS_FILE.open("r", encoding="utf-8") as file:
            self.labels = json.load(file)

    def recognize_image(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        results = []

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            face = cv2.equalizeHist(face)

            label, confidence = self.recognizer.predict(face)

            reg_no = self.labels.get(str(label))
            recognized = (
                reg_no is not None
                and confidence <= self.threshold
            )

            results.append({
                "box": (x, y, w, h),
                "student_id": reg_no if recognized else None,
                "recognized": recognized,
                "confidence": float(confidence)
            })

        return results
