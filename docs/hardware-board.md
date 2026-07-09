# D-Racer / TOPST D3-G 하드웨어 스펙

> 마지막 업데이트: 2026-07-09  
> **PC(WSL) 상위 프로젝트** 동기화본: [`../../docs/competition/hardware-board.md`](../../docs/competition/hardware-board.md)  
> **D3-G 보드 단독 clone**에서는 이 파일이 보드·플랫폼 스펙 SSOT입니다.

SEA:ME 해커톤 **D-Racer Kit**는 **TOPST D3-G** SBC + Waveshare 스케일카 섀시·전원·I2C 액추에이터 보드로 구성된 **ROS2 기반 RC 레이싱 키트**입니다.

관련 문서:
- 카메라: [hardware-camera.md](./hardware-camera.md)
- 보드 셋업·주행: [setup.md](./setup.md), [board-workflow.md](./board-workflow.md)
- 주최측 조립 가이드: [D-Racer Hardware Assembly Guide](https://github.com/topst-development/D-Racer-Kit/blob/release/v1.0.0/docs/%5B1%5D%20D-Racer%20Hardware%20Assembly%20Guide.md)

---

## 1. 플랫폼 개요

| 항목 | 내용 |
|------|------|
| 키트명 | TOPST **D-Racer Kit** |
| 메인 SBC | **TOPST D3-G** (8GB RAM / 32GB eMMC — 해커톤 키트 기준) |
| 차체 | Waveshare 스케일카 섀시 (4륜, 서보 조향 + ESC 구동) |
| 제어 API | PiRacerPro 유사 — `throttle` + `steering` (-1.0 ~ +1.0) |
| OS (대회 이미지) | **Ubuntu 22.04** |
| 미들웨어 | **ROS2 Humble** |
| 아키텍처 | **aarch64** (Arm 64-bit) |

D-Racer는 LiDAR·IMU가 기본 장착되지 않은 **비전 중심 소형 주행 플랫폼**입니다. 인지 입력의 주 경로는 **USB 카메라**입니다.

---

## 2. D3-G 보드 스펙

주최측 [조립 가이드 Table 2](https://github.com/topst-development/D-Racer-Kit/blob/release/v1.0.0/docs/%5B1%5D%20D-Racer%20Hardware%20Assembly%20Guide.md) 및 [TOPST D3-G 공식](https://docs.topst.ai/product/g/d3)을 통합 정리했습니다.

### 2.1 SoC / 연산

| 항목 | 스펙 |
|------|------|
| SoC | Telechips **Dolphin3** 계열 (조립 가이드: **TCC8051**, TOPST 제품 페이지: **TCC8050**) |
| Main CPU | **Cortex-A72 ×4** @ **1.69 GHz** (~31,840 DMIPS) |
| Sub CPU | **Cortex-A53 ×4** @ **1.45 GHz** (~13,340 DMIPS) |
| MCU | **Cortex-R5** @ **600 MHz** |
| 합산 DMIPS | ~45,180 |
| GPU | **PowerVR 9XTP GT9524** — 168 GFLOPS, OpenGL ES 3.x, Vulkan 1.2, OpenCL 2.0/3.0 |

### 2.2 메모리 / 저장

| 항목 | 스펙 |
|------|------|
| RAM | **LPDDR4X** — 4 GB 또는 **8 GB** (해커톤 키트: **8 GB**) |
| Boot SNOR | 4 MB (Quad SPI) |
| eMMC | **32 GB** MLC |
| 확장 | microSD 슬롯 |

> 대회 공식 이미지 설치 후 **eMMC 파티션 확장**이 필요합니다. [D-Racer 개발 환경 가이드 §5](https://github.com/topst-development/D-Racer-Kit/blob/release/v1.0.0/docs/%5B2%5D%20Development%20Environment%20Setup%20Guide.md)

### 2.3 온보드 인터페이스 (D3-G 단독)

| 인터페이스 | 스펙 |
|------------|------|
| USB | USB **3.0** Host (Type-A) ×1, USB **2.0** Host (Type-A) ×1, USB **2.0** Device (Type-C) ×1 |
| Ethernet | **1 Gbps** |
| Display | **DP 1.4** 4-Lane — MST로 최대 4 디스플레이 |
| Camera (온보드) | **MIPI CSI-2** 2-Lane ×2 (15-pin); 커넥터 교체 시 4-Lane 옵션 |
| PCIe | **PCIe 3.0 ×1** |
| GPIO | **40-pin** (2×20, Raspberry Pi HAT+ 호환) — I2C, SPI, UART, I2S, PWM, GPIO |
| CAN | **3채널** (10-pin 헤더, 트랜시버 서브보드) |
| Debug | Cortex UART ×3, JTAG |
| 전원 | 권장 **5 V @ 5 A** |
| PCB 크기 | **90 mm × 120 mm** |

### 2.4 대회 기본 계정 / 이미지

| 항목 | 값 |
|------|-----|
| 기본 사용자 | `topst` / `topst` |
| 공식 OS 이미지 | [D-Racer Ubuntu 22.04 v1.0.1](https://topst-downloads.s3.ap-northeast-2.amazonaws.com/Ubuntu/22.04/D-Racer-ubuntu-22.04-v1.0.1.zip) |
| 원격 개발 | VSCode Remote SSH + Wi-Fi dongle 권장 |

---

## 3. D-Racer Kit 구성 (조립 기준)

[조립 가이드 Table 1](https://github.com/topst-development/D-Racer-Kit/blob/release/v1.0.0/docs/%5B1%5D%20D-Racer%20Hardware%20Assembly%20Guide.md) 요약:

| No. | 부품 | 개발 시 의미 |
|-----|------|----------------|
| 1 | **D3-G** (8GB/32GB) | 메인 컴퓨터 |
| 2 | Battery Module Board (Waveshare) | 18650×4 전원, ESC/서보 배선 |
| 4 | **I2C Interface Board** | PCA9685 PWM — 조향·스로틀 |
| 6 | USB-Hub Box | 카메라·Wi-Fi·조이스틱 USB 확장 |
| 11 | USB-Hub | D3-G USB에 연결 |
| 12 | WiFi Dongle | SSH / 원격 개발 |
| 13 | **USB Camera** (C920e) | 전방 비전 — [hardware-camera.md](./hardware-camera.md) |
| 20 | 18650 Battery ×4 | 12~16.8 V 범위 (배터리 패키지 기준) |
| 21 | Vehicle Chassis (Waveshare) | 4륜 RC 스케일카 섀시 |
| 22 | Joystick (Waveshare) | 수동 주행·E-Stop |

---

## 4. D-Racer에서 실제 쓰는 버스·주소

공식 `topst_utils` / ROS 패키지 기본값 기준입니다. 보드에서 `i2cdetect -y 3`으로 재확인하세요.

### 4.1 I2C (버스 3)

| 장치 | I2C 주소 | 역할 | 코드 위치 |
|------|---------|------|-----------|
| **PCA9685** | **0x40** | 서보(조향) + ESC(스로틀) PWM | `topst_utils/d3racer.py`, `control` 패키지 |
| **INA219** | **0x42** | 배터리 전압·전류 | `topst_utils/ina219.py`, `battery` 패키지 |
| OLED (옵션) | 0x3C | 디스플레이 (유틸 상수) | `topst_utils/ina219.py` |

```python
# topst_utils 기본값
I2C_BUS = 3
PCA9685_ADDR = 0x40   # d3racer.py 기본
INA_ADDR = 0x42
```

**배선**: D3-G 40-pin GPIO ↔ D-Racer I2C Interface Board (3.3V, SDA, SCL, GND). 조립 가이드 Figure 13~15 참고.

### 4.2 USB

| 장치 | 일반 디바이스 | 비고 |
|------|---------------|------|
| USB Camera (C920e) | `/dev/video1` (기본) | `vehicle_config.yaml`: `USB_CAM_DEVICE` |
| MIPI 카메라 | `/dev/video0` | 대회 키트는 **USB_CAM: true** |
| USB Hub | D3-G USB Host에 연결 | 카메라·Wi-Fi·조이스틱 |
| Joystick | USB HID | `joystick` 패키지 |

> 카메라 점유 충돌: `camera_node`와 `monitor_node`가 동시에 카메라를 열면 실패할 수 있음. [board-workflow.md](./board-workflow.md) ArUco 테스트 절 참고.

### 4.3 MIPI CSI

D3-G 온보드 MIPI는 키트 기본 구성에서 **USB 카메라를 사용**합니다. MIPI는 `MIPI_CAM: false`가 기본이며, 대회 인지 파이프라인과 무관합니다.

---

## 5. 차량 구동 (액추에이터)

| 항목 | 스펙 |
|------|------|
| 구동 방식 | **전륜(또는 섀시) 서보 조향** + **ESC 모터 제어** (RC카 방식) |
| 제어 추상화 | `throttle` ∈ [-1, 1], `steering` ∈ [-1, 1] |
| PWM | PCA9685 @ 50 Hz |
| 조향 채널 | CH **0** (기본) |
| 스로틀 채널 | CH **1** (기본) |
| 서보 펄스 | 중립 1500 µs, ±500 µs 스팬 (기본) |
| ESC 펄스 | 중립 1500 µs, 전진 2000 µs, 후진 1000 µs (기본) |
| 트림 | `STEER_TRIM` in `vehicle_config.yaml` (조이스틱 Y/B로 보정·저장) |

ROS2 메시지 경로:

```
/joystick  또는  /control  (control_msgs/Control)
        ↓
  control_node  →  PCA9685  →  서보 + ESC
```

자율주행 시 `inference` 패키지가 `/control`을 publish합니다.

---

## 6. 소프트웨어 스택 (개발 참고)

### 6.1 ROS2 패키지 (D-Racer-Kit)

| 패키지 | 역할 |
|--------|------|
| `camera` | USB/MIPI → `/camera/image/compressed` |
| `control` | `/control` → PCA9685 |
| `joystick` | 게임패드 → `/joystick`, E-Stop |
| `battery` | INA219 → `/battery_status` |
| `monitor` | 웹 대시보드 (카메라·제어값) |
| `opencv` | OpenCV 데모 |
| `topst_utils` | D3Racer, PCA9685, INA219 드라이버 |
| **`inference`** | **팀 자율주행** (본 레포) |

### 6.2 핵심 토픽

```
/camera/image/compressed  →  inference_node  →  /control  →  control_node
/joystick (E-Stop)         →  control_node
/battery_status            →  monitor_node
```

### 6.3 `vehicle_config.yaml` (하드웨어 연동)

| 키 | 기본값 | 설명 |
|----|--------|------|
| `USB_CAM` | `true` | USB 카메라 사용 |
| `USB_CAM_DEVICE` | `/dev/video1` | 카메라 디바이스 |
| `IMAGE_WIDTH` / `IMAGE_HEIGHT` | 320 / 160 | 카메라 출력 해상도 |
| `STEER_TRIM` | 팀별 보정값 | 조향 중립 오프셋 |
| `IMAGE_TOPIC` | `/camera/image/compressed` | 모니터·인지 입력 |

경로: `external/D-Racer-Kit/src/config/vehicle_config.yaml`

---

## 7. 개발 시 성능·제약

| 항목 | 참고 |
|------|------|
| CPU | 8코어 Arm이나 vision + ROS2 + 웹 모니터 동시 시 **CPU·메모리 부하** 큼 |
| 카메라 FPS | `publish_hz` 최대 ~30; 그 이상은 보드에서 불안정 ([Camera Package](https://github.com/topst-development/D-Racer-Kit/blob/release/v1.0.0/docs/%5B7%5D%20Camera%20Package.md)) |
| 해상도 | 기본 **320×160** — 인지·성능 균형점; 올리면 latency 증가 |
| 배터리 | INA219 기준 약 **12.0~16.8 V** → % 환산 |
| PC vs 보드 | PC Docker는 **빌드·로직 검증**; **주행·카메라·I2C**는 D3-G에서만 |
| 아키텍처 | aarch64 — PC x86 Docker 이미지와 바이너리 호환 안 됨 |

---

## 8. 공식 참고 링크

| 자료 | URL |
|------|-----|
| D-Racer-Kit (ROS2) | https://github.com/topst-development/D-Racer-Kit/tree/release/v1.0.0 |
| D3-G 제품 페이지 | https://topst.ai/product/g/d3 |
| D3-G 문서 (TOPST) | https://docs.topst.ai/product/g/d3 |
| Ubuntu 이미지 | https://topst-downloads.s3.ap-northeast-2.amazonaws.com/Ubuntu/22.04/D-Racer-ubuntu-22.04-v1.0.1.zip |
| 팀 카메라 스펙 | [hardware-camera.md](./hardware-camera.md) |
| 팀 시뮬 메모 | [simulation.md](./simulation.md) |

---

## 9. 시뮬레이터 베이스 (참고)

D-Racer 전용 Gazebo 모델은 없습니다. 인지 샌드박스용으로는 AgileX **LIMO Ackermann**이 가장 가깝고, **카메라 마운트 각도는 실차 실측** 후 URDF에 반영합니다. 상세: [simulation.md](./simulation.md).
