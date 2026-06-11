import cv2
import keyboard
from ultralytics import YOLO
from collections import defaultdict

from evaluator import best_poker_hand

def predict():
    frame_counter = 0
    prev_p = False
    model = YOLO("VirtualCamera/models/cards_train-3/weights/best.pt")

    source = 0
    cap = cv2.VideoCapture(source)

    cv2.namedWindow("YOLO Live Card Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Live Card Detection", 960, 540)

    if not cap.isOpened():
        raise RuntimeError("Kamera/Stream konnte nicht geöffnet werden.")

    SPLIT_Y = 300   # mit Pfeiltasten anpassen
    best_text = "Best Hand: press H"

    COLORS = {
        "Spieler":     (50,  180, 255),
        "Tischkarten": (0,   200, 120),
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            frame,
            conf=0.55,
            imgsz=960,
            device=0,
            verbose=False
        )

        annotated_frame = results[0].plot()

        h, w = annotated_frame.shape[:2]


        cv2.line(annotated_frame, (0, SPLIT_Y), (w, SPLIT_Y), (200, 200, 200), 1)
        cv2.putText(annotated_frame, "TISCHKARTEN", (10, SPLIT_Y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["Tischkarten"], 2)
        cv2.putText(annotated_frame, "SPIELER", (10, SPLIT_Y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["Spieler"], 2)

        detections = defaultdict(dict)  

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cy   = (y1 + y2) // 2
            zone = "Spieler" if cy > SPLIT_Y else "Tischkarten"
            card = model.names[int(box.cls)]
            conf = float(box.conf)

            # nur speichern wenn diese Karte noch nicht da ist
            # oder neue Confidence besser ist
            if card not in detections[zone] or conf > detections[zone][card]:
                detections[zone][card] = conf

        tisch   = "  ".join(detections["Tischkarten"].keys()) or "—"
        spieler = "  ".join(detections["Spieler"].keys())     or "—"

        player_cards = list(detections["Spieler"].keys())
        table_cards = list(detections["Tischkarten"].keys())

        all_cards = player_cards + table_cards

        key = cv2.waitKey(1) & 0xFF

        if key == ord("h"):
            if len(all_cards) >= 5:
                try:
                    best_hand_name, best_hand_cards = best_poker_hand(all_cards)
                    best_text = f"Best Hand: {best_hand_name} | {' '.join(best_hand_cards)}"
                    print(best_text)
                except ValueError as e:
                    best_text = "Best Hand: card format error"
                    print(e)
            else:
                best_text = f"Best Hand: need at least 5 cards ({len(all_cards)} detected)"
                print(best_text)

        if key == ord("q"):
            break

        cv2.putText(annotated_frame, f"Tisch:   {tisch}",   (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["Tischkarten"], 2)
        cv2.putText(annotated_frame, f"Spieler: {spieler}", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["Spieler"], 2)

        cv2.putText(
            annotated_frame,
            best_text,
            (10, h-70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )
        cv2.imshow("YOLO Live Card Detection", annotated_frame)

        
        if keyboard.is_pressed('up'):
            SPLIT_Y = max(0,  SPLIT_Y - 5)
        elif keyboard.is_pressed('down'):
            SPLIT_Y = min(h,  SPLIT_Y + 5)
        

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    predict()