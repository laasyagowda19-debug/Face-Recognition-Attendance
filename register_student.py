
import cv2
import json
import shutil
import re
import numpy as np

from pathlib import Path


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"
STUDENTS_FILE = BASE_DIR / "students.json"
MODEL_DIR = BASE_DIR / "models"

MODEL_FILE = MODEL_DIR / "face_model.yml"
LABELS_FILE = MODEL_DIR / "labels.json"


# ==========================================
# SETTINGS
# ==========================================

MAX_PHOTOS = 10
FACE_MATCH_THRESHOLD = 65
FACE_SIZE = (200, 200)

CASCADE_FILE = (
    Path(cv2.data.haarcascades)
    / "haarcascade_frontalface_default.xml"
)


# ==========================================
# VERIFY FACE DETECTOR
# ==========================================

def load_face_detector():
    if not CASCADE_FILE.is_file():
        print("ERROR: Face detector XML file not found.")
        print("Expected location:", CASCADE_FILE)
        return None

    detector = cv2.CascadeClassifier(
        str(CASCADE_FILE)
    )

    if detector.empty():
        print("ERROR: Could not load face detector.")
        return None

    return detector


# ==========================================
# LOAD STUDENTS
# ==========================================

def load_students():
    if not STUDENTS_FILE.exists():
        return {}

    try:
        with STUDENTS_FILE.open(
            "r", encoding="utf-8"
        ) as file:
            students = json.load(file)

        if isinstance(students, dict):
            return students

        print("Invalid students.json format.")
        return {}

    except (json.JSONDecodeError, OSError) as error:
        print("Could not load students.json:", error)
        return {}


# ==========================================
# SAVE STUDENTS
# ==========================================

def save_students(students):
    STUDENTS_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    with STUDENTS_FILE.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            students,
            file,
            indent=4,
            ensure_ascii=False
        )


# ==========================================
# VALIDATE STUDENT NAME
# ==========================================

def get_student_name():
    while True:
        name = input(
            "Enter student name: "
        ).strip()

        if not name:
            print("Name cannot be empty.")
            continue

        if not re.fullmatch(r"[A-Za-z ]+", name):
            print(
                "Invalid name. Use English letters "
                "and spaces only."
            )
            continue

        return " ".join(name.split())


# ==========================================
# VALIDATE REGISTER NUMBER
# ==========================================

def get_register_number():
    while True:
        reg_no = input(
            "Enter register number: "
        ).strip()

        if not re.fullmatch(r"[0-9]+", reg_no):
            print(
                "Invalid register number. "
                "Enter numeric digits only."
            )
            continue

        if int(reg_no) == 0:
            print("Register number cannot be zero.")
            continue

        return reg_no


# ==========================================
# PREPROCESS FACE
# ==========================================

def preprocess_face(face):
    face = cv2.resize(face, FACE_SIZE)
    face = cv2.equalizeHist(face)
    return face


# ==========================================
# LOAD EXISTING TRAINED MODEL
# ==========================================

def load_existing_recognizer():
    if not hasattr(cv2, "face"):
        print(
            "OpenCV face module is unavailable. "
            "Install opencv-contrib-python."
        )
        return None, {}

    if not MODEL_FILE.is_file():
        return None, {}

    if not LABELS_FILE.is_file():
        return None, {}

    try:
        recognizer = (
            cv2.face.LBPHFaceRecognizer_create()
        )

        recognizer.read(str(MODEL_FILE))

        with LABELS_FILE.open(
            "r", encoding="utf-8"
        ) as file:
            labels = json.load(file)

        if not isinstance(labels, dict):
            print("Invalid labels.json format.")
            return None, {}

        return recognizer, labels

    except Exception as error:
        print("Could not load trained model:", error)
        return None, {}


# ==========================================
# FIND STUDENT BY MODEL LABEL
# ==========================================

def find_student_by_label(label, labels, students):
    value = labels.get(str(label))

    if value is None:
        return None

    if isinstance(value, (str, int)):
        reg_no = str(value)
        student = students.get(reg_no)

        if student:
            return {
                "name": student.get("name", "Unknown"),
                "reg_no": reg_no
            }

    elif isinstance(value, dict):
        raw_reg_no = value.get(
            "reg_no",
            value.get("register_number")
        )

        if raw_reg_no is None:
            return None

        reg_no = str(raw_reg_no)
        student = students.get(reg_no)

        if student:
            return {
                "name": student.get(
                    "name",
                    value.get("name", "Unknown")
                ),
                "reg_no": reg_no
            }

    return None


# ==========================================
# CHECK DUPLICATE FACE
# ==========================================

def find_existing_face(
    face, recognizer, labels, students
):
    if recognizer is None:
        return None

    try:
        processed_face = preprocess_face(face)

        label, distance = recognizer.predict(
            processed_face
        )

        print(
            f"Face comparison distance: {distance:.2f}"
        )

        if distance <= FACE_MATCH_THRESHOLD:
            return find_student_by_label(
                label, labels, students
            )

    except Exception as error:
        print("Face comparison failed:", error)

    return None


# ==========================================
# REGISTER STUDENT
# ==========================================

def main():
    print()
    print("STUDENT FACE REGISTRATION")
    print("=========================")

    name = get_student_name()
    reg_no = get_register_number()

    students = load_students()

    # Check duplicate register number
    if reg_no in students:
        print("\nRegister number already exists.")
        print("Student:", students[reg_no].get("name"))
        print("Register number:", reg_no)
        return

    # Load detector before creating folders
    detector = load_face_detector()

    if detector is None:
        print(
            "\nPlease fix the OpenCV cascade file "
            "before registering students."
        )
        return

    # Load existing model if available
    recognizer, labels = load_existing_recognizer()

    if recognizer is None:
        print(
            "\nWARNING: No trained face model found."
        )
        print(
            "Duplicate-face checking is unavailable "
            "until the model is trained."
        )

    # Prepare student photo folder
    student_dir = DATASET_DIR / reg_no

    if student_dir.exists() and any(
        student_dir.iterdir()
    ):
        print(
            "\nA non-empty photo folder already exists "
            "for this register number."
        )
        print("Check it before registering again.")
        return

    student_dir.mkdir(
        parents=True, exist_ok=True
    )

    # Open webcam
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print("ERROR: Cannot open webcam.")
        shutil.rmtree(
            student_dir, ignore_errors=True
        )
        return

    count = 0
    duplicate_student = None
    cancelled = False

    print("\nRegistration instructions:")
    print("- Look at the camera.")
    print("- Press SPACE to capture a photo.")
    print(f"- Capture {MAX_PHOTOS} photos.")
    print("- Press Q to cancel.")
    print("- Use good lighting and vary your angle.")

    try:
        while count < MAX_PHOTOS:
            ok, frame = camera.read()

            if not ok or frame is None:
                print("Could not read webcam.")
                cancelled = True
                break

            gray = cv2.cvtColor(
                frame, cv2.COLOR_BGR2GRAY
            )

            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80)
            )

            face_box = None

            if len(faces) > 0:
                face_box = max(
                    faces,
                    key=lambda box: box[2] * box[3]
                )

                x, y, w, h = face_box

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

            cv2.putText(
                frame,
                f"Photos: {count}/{MAX_PHOTOS}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imshow(
                "AIML Student Registration",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                cancelled = True
                break

            if key == 32:
                if face_box is None:
                    print(
                        "No face detected. "
                        "Look toward the camera and try again."
                    )
                    continue

                x, y, w, h = face_box

                face = gray[y:y + h, x:x + w]

                # Compare with previously trained students
                duplicate_student = find_existing_face(
                    face,
                    recognizer,
                    labels,
                    students
                )

                if duplicate_student:
                    print("\nPOSSIBLE DUPLICATE FACE!")
                    print(
                        "Matched student:",
                        duplicate_student["name"]
                    )
                    print(
                        "Register number:",
                        duplicate_student["reg_no"]
                    )
                    print("Registration cancelled.")
                    cancelled = True
                    break

                processed_face = preprocess_face(face)

                next_count = count + 1

                image_path = (
                    student_dir
                    / f"face_{next_count:03d}.jpg"
                )

                if cv2.imwrite(
                    str(image_path), processed_face
                ):
                    count = next_count
                    print(
                        f"Saved photo {count}/{MAX_PHOTOS}"
                    )
                else:
                    print("Could not save photo.")

    except KeyboardInterrupt:
        cancelled = True
        print("\nRegistration interrupted.")

    finally:
        camera.release()
        cv2.destroyAllWindows()

    # Handle duplicate face
    if duplicate_student is not None:
        shutil.rmtree(
            student_dir, ignore_errors=True
        )

        print("\nDUPLICATE REGISTRATION BLOCKED.")
        print(
            "Existing student:",
            duplicate_student["name"]
        )
        print(
            "Register number:",
            duplicate_student["reg_no"]
        )
        return

    # Save student only after all photos are captured
    if count == MAX_PHOTOS and not cancelled:
        students[reg_no] = {
            "name": name
        }

        try:
            save_students(students)
        except OSError as error:
            shutil.rmtree(
                student_dir, ignore_errors=True
            )
            print("Could not save student information:", error)
            return

        print("\nREGISTRATION SUCCESSFUL!")
        print("Student name:", name)
        print("Register number:", reg_no)
        print(f"Saved {MAX_PHOTOS} photos.")
        print("\nNext step: train the face recognition model.")
        print("Run the training script provided by your project.")

    else:
        shutil.rmtree(
            student_dir, ignore_errors=True
        )

        print(
            f"\nRegistration cancelled after {count} photos."
        )
        print("Incomplete photos were removed.")
        print("Student details were not saved.")


# ==========================================
# START PROGRAM
# ==========================================

if __name__ == "__main__":
    main()