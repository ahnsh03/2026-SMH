# 좌우 표지판 현장 조명 재학습 (v1)

대회측 사전 제공 영상으로 학습된 기존 `weights/sign_best.onnx`가 실제 대회장 조명에서
검출률이 떨어져서, 현장에서 새로 녹화한 bag 파일로 재학습한 1차 결과물이다.

이 폴더는 `direction_sign/detector.py`(담당: 장원정)를 건드리지 않고 완전히 독립적으로
작업한 데이터 준비·학습 산출물이다.

## 폴더 구성

| 경로 | 내용 |
|---|---|
| `extract_candidates.py` | bag → 후보 프레임 추출 + 규칙기반(파란원+흰화살표) 초안 bbox 라벨링 스크립트 |
| `dataset_v1/` | 사람이 직접 전수 검수해서 정제한 학습셋 (56장, YOLO 포맷) |
| `trained_model/sign_best_retrain_v1.onnx` | 재학습된 모델 (416×416, opset12, 기존 `detector.py`와 입출력 shape `[1,6,3549]` 완전 호환 확인됨) |
| `trained_model/sign_best_retrain_v1.pt` | 같은 모델의 PyTorch 체크포인트 (추가 파인튜닝용) |

## 데이터 출처

6개 bag(`bag_20260711_144948`, `bag_20260711_150234`, `bev_bag`, `bev_bag2`,
`camera_only_01`, `camera_only_02`)에서 `extract_candidates.py`로 118장의 후보를 뽑은 뒤,
118장 전체를 콘택트시트로 만들어 **사람이 직접 한 장 한 장 육안 검수**했다.

- 규칙기반 초안 라벨의 실제 오탐률이 매우 높았음(트랙 커브 경계, 노트북 화면, 휴대폰
  화면을 표지판으로 잘못 잡은 경우 다수) — 118장 중 62장을 오라벨로 판단해 제외
- 복수 박스(진짜 표지판 + 오탐 박스가 한 프레임에 같이 잡힌 경우) 6장도 라벨 오염
  가능성으로 제외
- 최종 56장(Left Sign 35 / Right Sign 21)만 학습에 사용 — **표지판이 카메라 앞에
  고정 거치된 채 촬영된 캘리브레이션성 구간(`camera_only_01/02`)이 대부분**이라, 실제
  주행 중 다양한 각도·거리 변화는 상대적으로 적게 반영되어 있음

## 학습

```bash
pip install ultralytics
python3 -c "
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
model.train(data='dataset_v1/data.yaml', imgsz=416, epochs=150, batch=8,
             device='cpu', patience=0)
"
```

- 1차 시도(epochs=60, `patience=20`)는 데이터셋이 8장짜리 검증셋이라 지표가 너무
  튀어서, early-stopping이 **사실상 학습이 거의 안 된 epoch 3을 "best"로 잘못 선택**함
  (raw confidence 0.007 수준 — 아무것도 못 잡는 상태). **이 결과물(v1)은 `patience=0`으로
  early-stopping을 끄고 150 epoch까지 다 돌린 버전**이다.
- 최종 지표(150 epoch): precision 0.85 / recall 0.9 / mAP50 0.94 / mAP50-95 0.90
  (단, 검증셋 자체가 8장뿐이라 이 숫자의 통계적 신뢰도는 낮음 — 아래 "실측 검증"이 더 의미있음)

## 실측 검증 (118장 전체, 학습에 안 쓰인 프레임 포함)

| | 기존 규칙기반 폴백 (`detect_turn_rule_based`) | 재학습 모델 v1 |
|---|---|---|
| 학습에 쓴 56장 정탐률 | (해당 없음 — 애초에 이 규칙기반으로 후보를 뽑았음) | **56/56 (100%)** |
| 오탐으로 제외한 62장 중 오탐 개수 | 62/62 (100%, 정의상 당연 — 이 규칙기반이 뽑은 후보라서) | **3/62 (5%)** |

즉 **기존 규칙기반이 잘못 잡던 오탐 사례의 95%를 재학습 모델이 걸러낸다.** 남은 3건
오탐(`bag_20260711_144948_00864`, `02124`, `bag_20260711_150234_01339`)은 전부 트랙
커브/분기 지점 — 여전히 커브 모양을 표지판으로 착각하는 잔여 취약점이 있다.

## ⚠️ 알려진 한계

- **학습 데이터가 56장으로 매우 작다.** 실전 배포 전 실제 주행 중 더 다양한 각도·거리·
  조명에서 추가 데이터를 모아 재학습하는 걸 강력 권장.
- 검증셋(8장)이 너무 작아 학습 중 리포트되는 mAP 지표는 신뢰도가 낮음 — 위 "실측 검증"
  (118장 전체 실측)이 훨씬 신뢰할 만한 근거.
- 트랙 커브/분기 구간에서의 잔여 오탐(3건)은 아직 미해결 — 다음 재학습 때 이런 프레임을
  "표지판 없음(negative sample)"으로 명시적으로 포함하면 개선 여지가 있음.

## 배포 방법

```bash
cp scripts/sign_dataset/trained_model/sign_best_retrain_v1.onnx weights/sign_best.onnx
```

배포 전 보드에서 `scripts/check_sign_webcam.py`로 실제 좌/우 표지판 앞에서 재확인 권장.
