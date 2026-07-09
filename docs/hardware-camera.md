# Logitech C920e — D-Racer 카메라 스펙

> 마지막 업데이트: 2026-07-09  
> **PC(WSL) 상위 프로젝트**를 쓰는 경우 동일 SSOT: [`../../docs/competition/camera-c920e.md`](../../docs/competition/camera-c920e.md) + PDF [`../../docs/competition/camera-c920e-datasheet.pdf`](../../docs/competition/camera-c920e-datasheet.pdf).  
> **D3-G 보드 단독 clone**에서는 이 파일이 카메라 스펙 SSOT입니다.

> 플랫폼·SBC·I2C·구동: [hardware-board.md](./hardware-board.md)

SEA:ME 해커톤 D-Racer Kit 전방 USB 카메라: **Logitech C920e Business Webcam** (M/N: V-U0028).

---

## 하드웨어 스펙

| 항목 | 값 |
|------|-----|
| 연결 | USB 2.0 / UVC |
| 최대 영상 | **1080p @ 30 fps**, 720p @ 30 fps |
| dFoV / hFoV / vFoV | **78° / 70.42° / 43.3°** (고정) |
| 초점 거리 | **3.67 mm** |
| 렌즈 / 초점 | Glass / Autofocus |
| 광학 | True 3 MP, RightLight 2 |
| 디지털 줌 | 1× |

출처: 제공 데이터시트 + [Logitech Sync Hub 스펙](https://hub.sync.logitech.com/c920e/post/specifications---c920e-business-webcam-TKnike7FetCzuAt) + [제품 페이지](https://www.logitech.com/en-us/products/webcams/c920e-business-webcam.html).

센서 크기(예: 1/2.9")는 C920e 공식 문서에 없음 → 비공식 인용은 시뮬 파라미터에 쓰지 말 것.

---

## D-Racer 런타임 (`vehicle_config.yaml`)

팀 설정: [`config/vehicle_config.yaml`](../config/vehicle_config.yaml) — `init_workspace.sh`가 `src/config/vehicle_config.yaml`로 링크합니다.

| 항목 | 팀 값 | 주최측 기본 |
|------|--------|-------------|
| `USB_CAM` | `true` (`/dev/video1`) | 동일 |
| `IMAGE_WIDTH` × `IMAGE_HEIGHT` | **320 × 180** (16:9) | 320 × 160 (2:1) |
| 토픽 | `/camera/image/compressed` (JPEG) | 동일 |
| `publish_hz` | ~30 (보드 한계 가능) | 동일 |

인지 파이프라인은 이 토픽·해상도를 기준으로 맞춥니다.

---

## Gazebo 매칭 힌트

| 파라미터 | 권장 |
|----------|------|
| `horizontal_fov` | **1.22906 rad (70.42°)** |
| 파이프라인 출력 | 320×180 JPEG → `/camera/image/compressed` |
| Gazebo 네이티브 | 640×360 (16:9) → `sim_camera_republish`에서 320×180 |
| 비고 | LIMO 기본 depth cam(~80°, 640×480)은 C920e와 불일치 |

1080p 핀홀 대략값: \(f_x \approx f_y \approx 1360\) (cx=960, cy=540). 실차는 캘리브레이션으로 확정.

시뮬 레포 검토: PC 상위 `docs/sim/limo-simulator-assessment.md` 참고.
