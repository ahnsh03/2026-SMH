# Vision tuning tools (Phase 0+)

시뮬·실차 공통. 설계: [docs/lane-drive-strategy.md](../../docs/lane-drive-strategy.md).

## 어디서 실행하나

| 환경 | 가능? | 비고 |
|------|--------|------|
| 호스트 WSL | ❌ | `cv2` / `rclpy` 없음 |
| **`2026-smh-sim`** | ✅ | **이미지 재빌드 불필요** — ROS source만 |
| D3-G 보드 | ✅ | |

```bash
./scripts/dev_container.sh sim-bringup     # 터미널1
docker exec -it 2026-smh-sim bash          # 터미널2
source /opt/ros/humble/setup.bash
```

---

## 카메라 캡처 (단축키만 저장)

시뮬에 이미 `D-Racer Camera` 창이 있으므로, 이 도구는 **자동 연사 저장하지 않습니다.**

```bash
python3 scripts/vision_tune/capture_camera.py --out data/captures/sim
```

- 창 `capture_hotkey` (640×360)에 포커스
- **`c` 또는 Space** → 현재 프레임 1장 PNG 저장
- **`q`** → 종료

> 컨테이너는 보통 `root`로 돌아가서, 예전 캡처가 root 소유면 호스트에서 삭제가 안 됩니다.  
> 현재 스크립트는 저장 후 `aim06`(uid 1000)으로 chown 합니다.  
> 이미 root 파일만 남았다면:
> `docker exec 2026-smh-sim chown -R 1000:1000 /workspace/data/captures`

---

## BEV ROI 트랙바 (기본=라이브 토픽)

```bash
# 실시간 (기본). bringup 실행 중이어야 함
python3 scripts/vision_tune/tune_bev_roi.py

# 명시적 토픽
python3 scripts/vision_tune/tune_bev_roi.py --topic /camera/image/compressed

# 오프라인 스틸
python3 scripts/vision_tune/tune_bev_roi.py --folder data/captures/sim
```

| 창 | 크기 |
|----|------|
| `bev_tune_origin` / `bev_tune_roi` | 초기 **640×360** (D-Racer Camera와 동일) |
| `bev_tune_bev` | **트랙바 `bev_w`×`bev_h`에 맞춤** (이미지 크기로 자동) |

### 트랙바

| 이름 | 역할 |
|------|------|
| `crop_top_%` | 상단 제외 |
| `bottom_half_%` | 아랫변 확장 (최대 **1500** = ratio 15.0) |
| `bev_w` / `bev_h` | BEV **픽셀 해상도** (미터 아님). 수동 조절은 “보기/연산량”용 |
| `guide_half_px` | 초록 ±반폭. 직선에서 **좌·우 차선 마크**에 맞춤 (기본 **44**) |

차로 폭 **`track_width_m = 0.35` 고정** (트랙바 없음).  
`m/px_lat = 0.35 / (2 × guide_half_px)` → guide 44일 때 **≈ 3.98 mm/px**.

### 시뮬 잠정 기본값 (`config/lane_vision.yaml`)

`crop_top=0.39`, `bottom_half=6.35`, `bev=500×370`, `guide_half_px=44`, 전방 커버 **≈ 1.5 m**.

### `bev_w`/`bev_h` vs 카메라 기하 IPM

| 방식 | 정확도 | 비고 |
|------|--------|------|
| 사다리꼴 + 수동 `bev_w/h` (현재 툴) | 시각 정렬용 | 해커톤·F1TENTH식 빠른 캘리브. **종·횡 m/px가 다름** |
| Metric IPM (높이·pitch·HFoV·지면 범위) | 물리적으로 맞음 | 원태 `build_ipm_maps` 계열. BEV 크기 = `(2·y_half)/mpp` × `(x_max−x_min)/mpp` 로 **자동** |

전문 개발 경로: 기하 IPM으로 스케일 고정 → 사다리꼴은 임시/검증용.  
지금 Phase 0는 사다리꼴로 “평평함”을 맞추고, 가로는 0.35 m 가이드, 세로는 **마커 또는 metric IPM**으로 맞추는 것이 맞다.

### 가로·세로 스케일

- **가로:** 초록선을 차선에 맞추기 ✅ (이름: `track_width`, 예전의 `road_width` 폐기)
- **세로:** 가로 m/px를 그대로 쓰지 말 것. 시뮬에 **`bev_calib_mat`**(트랙 남쪽 4×2 m, 0.1/0.5 m 격자)가 배치되어 있음 — BEV에서 major 칸을 세어 `m/px_long`을 구하거나, metric IPM으로 통일.

### Gazebo 캘리브 매트

```bash
# 텍스처 재생성 (크기·간격 바꿀 때)
python3 scripts/prepare_bev_calib_mat.py
# 월드 반영: build-sim 또는 bringup 재실행
```

위치: `(2.5, -6.5)` — 기본 spawn X 근처, 트랙 바로 아래. 설정: `src/dracer_sim/config/bev_calib_mat.yaml`.

### BEV 보조선

- 노란 세로: 차량 중심 · 가로 near/mid/far · **초록 ±가이드 (0.35 m)** · 20 px 격자

- `s` → YAML 저장 · `q` → 종료
