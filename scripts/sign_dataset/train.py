"""4-class(Left Sign/Right Sign/Red Light/Green Light) YOLO11n 재현 학습 스크립트.

dataset_v2/ (labels/*.cache가 있다면 반드시 삭제 후 실행 — 라벨 수정 후에도
Ultralytics가 캐시를 그대로 재사용해 예전 라벨로 학습되는 문제가 있었음)
을 기준으로 sign_light_best_v5b와 동일한 가중치를 재현한다.

사용법:
    rm -f scripts/sign_dataset/dataset_v2/labels/*.cache
    python3 scripts/sign_dataset/train.py
    # runs/detect/runs/sign_light_v5b/weights/best.onnx 생성됨
"""

from pathlib import Path

from ultralytics import YOLO

DATA_YAML = Path(__file__).parent / "dataset_v2" / "data.yaml"
RUN_NAME = "sign_light_v5b"


def main() -> None:
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(DATA_YAML),
        imgsz=416,
        epochs=150,
        batch=8,
        device="cpu",
        patience=0,
        project="runs",
        name=RUN_NAME,
    )
    best = Path("runs/detect/runs") / RUN_NAME / "weights" / "best.pt"
    YOLO(str(best)).export(format="onnx", opset=12, imgsz=416)


if __name__ == "__main__":
    main()
