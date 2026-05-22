import cv2
import os
import time

os.makedirs("camera_tests", exist_ok=True)

for i in range(5):
    print(f"\nTesting camera {i} with MSMF")

    cap = cv2.VideoCapture(i, cv2.CAP_MSMF)

    print("opened:", cap.isOpened())

    if not cap.isOpened():
        cap.release()
        continue

    time.sleep(1)

    ret, frame = cap.read()
    print("ret:", ret)

    if ret and frame is not None:
        print("shape:", frame.shape, "min/max:", frame.min(), frame.max())
        filename = f"VirtualCamera/camera_tests/camera_{i}_MSMF.png"
        cv2.imwrite(filename, frame)
        print("saved:", filename)

    cap.release()