import cv2
import numpy as np

for i in range(10):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"Camera {i}: nicht geöffnet")
        continue

    ret, frame = cap.read()

    if ret and frame is not None:
        mean = np.mean(frame)
        print(f"Camera {i}: OK | shape={frame.shape} | mean={mean:.2f}")

        cv2.imshow(f"Camera {i}", frame)
        #cv2.waitKey(1500)
        cv2.destroyAllWindows()
    else:
        print(f"Camera {i}: geöffnet, aber kein Frame")

    cap.release()