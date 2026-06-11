from ultralytics import YOLO
from pathlib import Path


def main():
    BASE_DIR = Path(__file__).resolve().parent  # VirtualCamera

    WEIGHTS = BASE_DIR / "models" / "cards_train-3" / "weights" / "best.pt"
    DATA_YAML = BASE_DIR/ "data" / "cards_combined_final" / "data.yaml"

    print("DATA_YAML:", DATA_YAML)
    print("DATA_YAML exists:", DATA_YAML.exists())

    print("WEIGHTS:", WEIGHTS)
    print("WEIGHTS exists:", WEIGHTS.exists())

    model = YOLO(str(WEIGHTS))

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=768,
        batch=16,
        device=0,
        project=str(BASE_DIR / "metrics"),
        name="test_eval",
        plots=True
    )

    print("Precision:", metrics.box.mp)
    print("Recall:", metrics.box.mr)
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)

if __name__ == "__main__":
    main()