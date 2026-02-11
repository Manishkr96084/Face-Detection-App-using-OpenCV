import cv2
import os

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_from_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not load image.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    print("Faces detected:", len(faces))

    cv2.imshow("Face Detection - Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_from_webcam():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot access webcam")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Face Detection - Webcam", frame)

        key = cv2.waitKey(1)

        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Face Detection App")
    print("1. Detect from Image")
    print("2. Detect from Webcam")

    choice = input("Enter your choice (1/2): ")

    if choice == "1":
        path = input("Enter image path: ")
        detect_from_image(path)
    elif choice == "2":
        detect_from_webcam()
    else:
        print("Invalid choice.")
