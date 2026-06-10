import cv2
from ultralytics import YOLO

model = YOLO("VirtualCamera/runs/detect/train-3/weights/best.pt")

source = "http://192.168.178.23:4747/video"
# source = 0  # normale Webcam

cap = cv2.VideoCapture(source)

cv2.namedWindow("YOLO Live Card Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Live Card Detection", 960, 540)

if not cap.isOpened():
    raise RuntimeError("Kamera/Stream konnte nicht geöffnet werden.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(
        frame,
        conf=0.5,
        imgsz=1280,
        device=0,
        verbose=False
    )

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Live Card Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()