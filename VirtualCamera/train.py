import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



from ultralytics import YOLO
from pathlib import Path
import torch
from multiprocessing import freeze_support


def main():
    BASE_DIR = Path(__file__).resolve().parent  # VirtualCamera

    DATA_YAML = BASE_DIR / "data" / "cards_combined_final" / "data.yaml"

    print("DATA_YAML:", DATA_YAML)
    print("Exists:", DATA_YAML.exists())

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"data.yaml nicht gefunden: {DATA_YAML}")

    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO("yolov8n.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=768,
        batch=24,
        device=0,
        workers=6,
        project=str(BASE_DIR / "models"),
        name="cards_train",
        pretrained=True,
        patience=10,
        save=True,
        plots=False,
        amp=True,
        deterministic=False
    )


if __name__ == "__main__":
    freeze_support()
    main()