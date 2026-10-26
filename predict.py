
import cv2
from face_engine import FaceEngine


def main():
    engine = FaceEngine()
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Cannot open webcam.")
        return

    print("Recognition started. Press Q to quit.")

    try:
        while True:
            ok, frame = camera.read()

            if not ok:
                print("Could not read webcam.")
                break

            results = engine.recognize_image(frame)

            for result in results:
                x, y, w, h = result["box"]

                if result["recognized"]:
                    name = f"Register No: {result['student_id']}"
                    color = (0, 255, 0)
                else:
                    name = "Unknown"
                    color = (0, 0, 255)

                cv2.rectangle(
                    frame, (x, y), (x+w, y+h), color, 2
                )
                cv2.putText(
                    frame, name, (x, max(y-10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
                )

            cv2.imshow("Face Recognition Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
