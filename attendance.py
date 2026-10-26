import cv2
import csv
import json
from pathlib import Path
from datetime import datetime

from face_engine import FaceEngine

BASE_DIR = Path(__file__).resolve().parent
STUDENTS_FILE = BASE_DIR / "students.json"
ATTENDANCE_FILE = BASE_DIR / "attendance.csv"

FIELDS = [
    "name",
    "register_number",
    "date",
    "entry_time",
    "exit_time",
    "status",
]


def load_students():
    if not STUDENTS_FILE.exists():
        raise FileNotFoundError(
            "students.json not found. Register students first."
        )

    with STUDENTS_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_attendance():
    if not ATTENDANCE_FILE.exists():
        return []

    with ATTENDANCE_FILE.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames != FIELDS:
            raise ValueError(
                "Your attendance.csv has an older format. "
                "Back it up as attendance_old.csv and restart."
            )

        return list(reader)


def write_attendance(rows):
    with ATTENDANCE_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def mark_entry_or_exit(reg_no, students):
    student = students.get(str(reg_no))

    if student is None:
        print("Student details not found:", reg_no)
        return "STUDENT NOT FOUND"

    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    rows = read_attendance()

    record = next(
        (
            row for row in rows
            if row["register_number"] == str(reg_no)
            and row["date"] == today
        ),
        None
    )

    if record is None:
        rows.append({
            "name": student["name"],
            "register_number": str(reg_no),
            "date": today,
            "entry_time": current_time,
            "exit_time": "",
            "status": "Present",
        })

        write_attendance(rows)

        message = f"ENTRY | {student['name']} | Reg No: {reg_no}"
        print(f"{message} at {current_time}")
        return "ENTRY"

    elif not record["exit_time"]:
        record["exit_time"] = current_time
        record["status"] = "Completed"

        write_attendance(rows)

        message = f"EXIT | {student['name']} | Reg No: {reg_no}"
        print(f"{message} at {current_time}")
        return "EXIT"

    else:
        print(
            f"Attendance already completed today for "
            f"{student['name']} ({reg_no})."
        )
        return "COMPLETED TODAY"


def main():
    students = load_students()
    engine = FaceEngine()
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Cannot open webcam.")
        return

    seen_counts = {}
    triggered = set()
    face_status = {}

    print("Face Attendance System Started.")
    print("First appearance = ENTRY.")
    print("Next appearance after leaving view = EXIT.")
    print("Press Q to quit.")

    try:
        while True:
            ok, frame = camera.read()

            if not ok:
                print("Could not read webcam frame.")
                break

            results = engine.recognize_image(frame)
            visible_ids = set()

            for result in results:
                x, y, w, h = result["box"]
                reg_no = str(result["student_id"])

                if result["recognized"] and reg_no in students:
                    visible_ids.add(reg_no)
                    seen_counts[reg_no] = (
                        seen_counts.get(reg_no, 0) + 1
                    )

                    name = students[reg_no]["name"]
                    color = (0, 255, 0)

                    if (
                        seen_counts[reg_no] >= 5
                        and reg_no not in triggered
                    ):
                        status = mark_entry_or_exit(reg_no, students)
                        face_status[reg_no] = status
                        triggered.add(reg_no)

                    status = face_status.get(reg_no, "RECOGNIZED")

                    label = f"{name} | Reg No: {reg_no}"
                    status_label = f"Status: {status}"

                else:
                    name = "Unknown"
                    color = (0, 0, 255)
                    label = "Unknown Student"
                    status_label = "Status: Not Marked"

                cv2.rectangle(
                    frame, (x, y), (x + w, y + h), color, 2
                )

                cv2.putText(
                    frame,
                    label,
                    (x, max(y - 30, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

                cv2.putText(
                    frame,
                    status_label,
                    (x, max(y - 8, 45)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

            # Allow another event after the face leaves
            # the camera view for several frames.
            for reg_no in list(seen_counts):
                if reg_no not in visible_ids:
                    seen_counts[reg_no] = (
                        seen_counts.get(reg_no, 0) + 1
                    )

                    if seen_counts[reg_no] >= 10:
                        seen_counts[reg_no] = 0
                        triggered.discard(reg_no)

            # Display current date and time on the camera.
            now_text = datetime.now().strftime(
                "%d-%m-%Y  %H:%M:%S"
            )

            cv2.putText(
                frame,
                now_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            cv2.imshow("Student Face Attendance", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
