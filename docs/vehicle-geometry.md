# D-Racer ↔ LIMO 조향·기하 스펙 (제어 튜닝 참고)

> 작성: 2026-07-11  
> 목적: 시뮬(LIMO Ackermann)에서 잡은 조향·속도 게인을 **실차(D-Racer)에 옮길 때** 참고할 하드웨어·기하 차이  
> 관련: [hardware-board.md](./hardware-board.md) · [simulation.md](./simulation.md) · [lane-drive-strategy.md](./lane-drive-strategy.md) · [`control_bridge.yaml`](../src/dracer_sim/config/control_bridge.yaml)

---

## 핵심 (한 줄)

`/control` 토픽·부호 계약은 실차와 시뮬이 같지만, **휠베이스·트레드·액추에이터·질량**이 달라 같은 `steering` 값이 **다른 곡률·체감**을 만든다. 시뮬 게인을 실차에 그대로 쓰지 말 것.

---

## 1. 스펙 비교

| 항목 | D-Racer (실차) | LIMO (공식 매뉴얼) | LIMO Gazebo (`vendor/limo_car`) |
|------|----------------|--------------------|----------------------------------|
| 구동 / 조향 | RC **서보 조향 + ESC** (Ackermann) | **허브모터 4WD + Ackermann** | `gazebo_ros_ackermann_drive` |
| 휠베이스 \(L\) | **175 mm** (팀 실측 2026-07-12) | **200 mm** | **240 mm** (`ackermann.xacro`) |
| 트레드 \(T\) | **120 mm** (팀 실측, 좌·우 **바퀴 중심** 간) | **175 mm** | **168 mm** |
| 외형 (대략) | ~255×140×215 mm | 322×220×251 mm | URDF 베이스 ~190×310×120 |
| 최대 조향각 | **±24.44°** (`atan(L/R_min)`, 제어 SSOT) | \(R_\min=0.4\,\mathrm{m}\) 역산 ≈ **27°** | **±31.94°** (`0.5574` rad) |
| 최소 선회반경 | **0.385 m** (팀 제어 SSOT) | **0.4 m** | **0.385 m** (\(L/\tan\delta\) 맞춤) |
| 카메라 | **h / pitch**: yaml `metric_ipm` · **종배치**: 앞차축보다 **25 mm 앞** | — | `ahead_front_axle_m: 0.025` |
| 후차축→카메라 \(d_{\mathrm{rc}}\) | **\(L + 0.025 = 0.200\,\mathrm{m}\)** | — | **\(0.24 + 0.025 = 0.265\,\mathrm{m}\)** |
| 최대 속도 | ESC·배터리 의존 (고속 RC) | **~1 m/s** | 브릿지 `max_linear_speed: 1.2` |
| 질량 | 경량 RC (~1–2 kg대) | ~4.8 kg | URDF `base_mass` 4.34 kg |
| `/control` 부호 | −1=좌 / +1=우 | — | 동일 규약 → Gazebo 좌(+)로 **부호 반전** |

### 출처

| 값 | 출처 |
|----|------|
| D-Racer \(L=175\) mm, \(T=120\) mm | 팀 실측 2026-07-12 (안승현) |
| 팀 \(R_{\min}=0.385\) m (제어 SSOT) | 2026-07-15 — 시뮬도 같은 \(R_{\min}\)에 \(\delta=\arctan(L/R)\) 맞춤 |
| D-Racer 카메라 h=0.13 m / pitch 10° | 시뮬과 동일하게 기구 맞춤 (IPM yaml; bag 재튜닝 시 yaml이 SSOT) |
| 카메라가 앞차축보다 25 mm 앞 | `sim_interface.yaml` `mount.ahead_front_axle_m: 0.025` (실차 동일 마운트) |
| 후차축→카메라 \(d_{\mathrm{rc}}=L+0.025\) | 실차 **0.200 m** · 시뮬 Gazebo **0.265 m** → `path.perception_to_rear_axle_x_m` |
| LIMO \(L=200\), \(T=175\), \(R_\min=0.4\) m | AgileX LIMO 사용자 매뉴얼 |
| Gazebo \(L=0.24\), \(\delta_\max=0.5574\), joint `velocity=8` | `ackermann.xacro`, `control_bridge.yaml` |
| D-Racer 액추에이터·PWM | [hardware-board.md](./hardware-board.md) §5 |

> 2026-07-12 측정(\(\delta=17.5°\), \(R\approx0.535\) m)은 기록용. **현재 튜닝 SSOT는 \(R_{\min}=0.385\) m**.

---

## 2. 왜 게인을 그대로 옮기면 안 되나

### 2.1 기하 (bicycle / Ackermann 근사)

\[
\kappa \approx \frac{\tan\delta}{L},\quad R \approx \frac{L}{\tan\delta}
\]

- D-Racer \(L=0.175\) m vs LIMO 시뮬 \(L=0.24\) m → **휠베이스 약 1.37배**.
- **같은 조향각**이면 LIMO 쪽 선회가 더 완만하다.
- **같은 곡률**을 내려면 LIMO가 더 큰 \(\delta\)가 필요하다.
- 트레드도 D-Racer \(T=0.12\) m vs LIMO 시뮬 \(0.168\) m로 좁다. 순수 bicycle PP에는 \(T\)가 안 들어가지만, 슬립·내외륜 차가 커지면 체감된다.

### 2.2 액추에이터 의미

| | D-Racer 실차 | LIMO 시뮬 (`sim_control_bridge`) |
|--|--------------|-----------------------------------|
| 입력 | `steering`, `throttle` ∈ [-1, 1] | 동일 |
| 출력 | PCA9685 PWM (조향 CH0, 스로틀 CH1) | `/cmd_vel` — 조향각(rad) / 선속도 |
| 스케일 | 중립 1500 µs, 조향 ±500 µs 스팬 | `max_steer_angle_rad: 0.5574`, `max_linear_speed: 1.2` |

인터페이스는 같고 **물리량 매핑은 다르다**.

### 2.3 동역학

LIMO는 무겁고 저속(~1 m/s), D-Racer는 가볍고 ESC 기반이라 가속·오버슈트·서보 지연이 다르다.  
시뮬에서 안정이어도 실차에서 진동·과조향이 날 수 있다.

### 2.4 Gazebo 조향 한계 · \(R_{\min}\) 맞춤 (2026-07-15)

팀 제어 SSOT 최소 선회반경 \(R_{\min}=0.385\,\mathrm{m}\).  
Gazebo LIMO 휠베이스 \(L=0.24\,\mathrm{m}\)에 맞추면

\[
\delta_{\max} = \arctan(L / R_{\min}) \approx 0.5574\,\mathrm{rad}\ (31.94^\circ)
\]

→ joint `limit` / plugin `max_steer` / `control_bridge` / `main_planner` 모두 **0.5574**.  
실차(\(\L=0.175\))는 같은 \(R_{\min}\)으로 \(\delta_{\max}\approx 0.4266\) rad (24.44°).

| | vendor 기본 | 팀 패치 (현재) |
|--|-------------|----------------|
| steer joint `upper`/`lower` | ±0.5236 (30°) | **±0.5574 (31.94°)** |
| plugin `max_steer` | 0.5236 | **0.5574** |
| steer joint `velocity` | 0.5 rad/s | **8.0 rad/s** |
| `damping` / `friction` | 1.0 / 2.0 | 0.05 / 0.05 |
| 풀 락 근사 시간 | ~1.0 s | **~0.07 s** |

패치: `vendor/limo_car/gazebo/ackermann.xacro`, `limo_steering_hinge.xacro`,  
`src/dracer_sim/config/control_bridge.yaml`, `config/main_planner.yaml`.  
**URDF 변경은 Gazebo 리스폰 / `sim-bringup` 재실행 후에만 적용**된다.

소프트웨어 조향 속도 한도는 `main_planner.yaml`의 `steering_rate_limit_per_sec`  
(현재 **16.0** /s on [-1,1] ≈ 물리적 ~8.9 rad/s ≤ joint vel).

실차(D-Racer)는 PWM 서보라 vendor 0.5 rad/s를 복제하지 말 것.

---

## 3. 시뮬 내부 주의

| 모델 | 용도 | 조향 기하 튜닝에 |
|------|------|------------------|
| `robot:=limo` (기본) | LIMO Ackermann — 팀 기본 시뮬 | **이 문서의 Gazebo 열** 기준 |
| `robot:=dracer` | 경량 박스 + **diff_drive** | **쓰지 말 것** (휠베이스 없음, 트레드≈0.15 m 근사) |

공식 LIMO는 \(L=0.20\) m인데, 팀 Gazebo URDF는 \(L=0.24\) m이다.  
일부 파일(`limo_ackerman_base.xacro`)은 `0.2`를 쓰므로, **시뮬 제어·변환은 `0.24`**, 실물 LIMO 스펙 인용은 **`0.20`**으로 구분한다.

---

## 4. 실차 튜닝 체크리스트

1. 시뮬에서 잡은 `steering` gain을 실차에 그대로 복사하지 말고, \(L\) 비로 1차 보정한 뒤 D3-G에서 재튜닝한다.
2. D-Racer **트레드·최대 조향각은 레포에 없음** → 실측 권장:
   - 전·후축 중심 간 거리 (휠베이스)
   - 좌·우 타이어 접지면 중심 간 거리 (트레드)
   - 전륜 최대 타이어각, 또는 제자리 최소 선회반경 \(R_\min\)
3. \(R_\min\)만 알면 \(\delta_\max \approx \arctan(L / R_\min)\)으로 최대 조향각을 추정할 수 있다.
4. `STEER_TRIM`(`vehicle_config.yaml`)은 기하가 아니라 **기구/서보 중립 오프셋** 보정용이다.
5. 최종 검증은 보드에서 `./scripts/board_sync.sh` 후 실차 주행으로 한다.

### 4.1 Pure Pursuit용 D-Racer 실측 현황

| 항목 | 상태 | 값 |
|------|------|-----|
| 휠베이스 \(L\) | ✅ | **0.175 m** |
| 트레드 \(T\) | ✅ (바퀴 중심 간) | **0.120 m** |
| 카메라 높이·피치 | ✅ 튜너 재측정 가능 | yaml `metric_ipm.camera` (문서 기본 0.13 m / 10°) |
| 카메라 ↔ 앞차축 (종) | ✅ 마운트 | **\(d_{\mathrm{cf}} = 0.025\,\mathrm{m}\)** (카메라가 앞차축보다 앞) |
| **후차축 → 카메라** \(d_{\mathrm{rc}}\) | ✅ \(L + d_{\mathrm{cf}}\) | **실차 0.200 m** · 시뮬 0.265 m |
| 인지→후차축 오프셋 | ✅ = \(d_{\mathrm{rc}}\) | `path.perception_to_rear_axle_x_m` |
| \(\delta_{\max}\) | ✅ \(R_{\min}\) 기준 SSOT | **24.44°** → `max_steer_angle_rad ≈ 0.4266` |
| \(R_{\min}\) | ✅ 제어 SSOT | **0.385 m** |
| `steering=±1` ↔ 타이어각 | ✅ 풀락 = \(\delta_{\max}\) | ±1 ↔ ±24.44° |

기록: 2026-07-12 직접 잰 \(\delta=17.5°\) / \(R\approx0.535\) m 은 archive.  
**현재 PP·mask 정규화 SSOT는 \(R_{\min}=0.385\) → \(\delta=\arctan(L/R)\)**.

`config/lane_control.yaml` / `main_planner.yaml` 기본은 **시뮬(LIMO, δ=0.5574)**. 실차 보드:

```text
wheelbase_m: 0.175
max_steer_angle_rad: 0.4266   # atan(0.175/0.385)
```

**트레드:** 좌·우 바퀴 **중심** 사이 횡거리 (전폭 아님). 12 cm OK.

### 4.1.1 종방향 기하 (후차축 · 앞차축 · 카메라)

전방 \(+x\). Metric IPM 지면 \(x\)는 **카메라** 원점이다.

```text
후차축 (PP 원점) ---- L (wheelbase) ---- 앞차축 ---- d_cf ---- 카메라 (IPM 원점)
                         ↑                              ↑
                    실차 0.175 m                   0.025 m
                    시뮬 0.240 m              (ahead_front_axle_m)
```

\[
d_{\mathrm{rc}} = L + d_{\mathrm{cf}}
\quad\text{(후차축 → 카메라 종거리)}
\]

| 플랫폼 | \(L\) | \(d_{\mathrm{cf}}\) | **\(d_{\mathrm{rc}}\)** | 플래너 키 |
|--------|-------|---------------------|-------------------------|-----------|
| **D-Racer 실차** | 0.175 | 0.025 | **0.200 m** | [`main_planner.real_car.yaml`](../config/main_planner.real_car.yaml) `path.perception_to_rear_axle_x_m` |
| **LIMO Gazebo** | 0.240 | 0.025 | **0.265 m** | [`main_planner.yaml`](../config/main_planner.yaml) 동키 · [`sim_interface.yaml`](../src/dracer_sim/config/sim_interface.yaml) `mount.ahead_front_axle_m` |

사용:

- IPM path 점 \((x_{\mathrm{cam}}, y)\) → 후차축 프레임: \(x_{\mathrm{rear}} = x_{\mathrm{cam}} + d_{\mathrm{rc}}\) (`MainPlanner._path_in_rear_axle_frame`).
- 앞차축은 카메라보다 \(d_{\mathrm{cf}}\) 뒤(후차축 기준 \(x=L\)). 이미지 최하단 ≈ `x_min_m`이지 범퍼/앞차축이 아님.

### 4.2 \(\delta_{\max}\) · \(R_{\min}\) · 실제 타이어각 — 측정 방법

셋은 한 관계로 묶인다 (후륜 중심 기준 bicycle):

\[
R_{\min} \approx \frac{L}{\tan\delta_{\max}},\qquad
\delta_{\max} \approx \arctan\!\left(\frac{L}{R_{\min}}\right)
\]

\(L=0.175\) m일 때 예: \(\delta_{\max}=25°\) → \(R_{\min}\approx0.375\) m, \(30°\) → \(\approx0.303\) m.

**방법 A — 타이어각 직접 (추천, 정확)**

1. 차 정지, 조향을 `/control`로 `steering=+1.0`(풀락 우) 고정. (보드에서 조이스틱 풀락도 가능)
2. 전륜 **한 쪽**(보통 외측 또는 평균용으로 좌·우 둘 다) 위에서 내려다보고, 차체 전진축(직진 때 휠면) 대비 타이어 면이 돌아간 각을 잰다.
   - 쉬운 방법: 종이/자를 차체 세로축에 맞추고, 다른 자를 휠 면에 붙인 뒤 각도기·스마트폰 각도 앱.
3. `steering=-1.0`도 반복 → 좌·우 \(\delta\)의 평균을 \(\delta_{\max}\)로 씀.
4. 가능하면 `steering=0, ±0.5, ±1.0`에서 각을 찍어 **정규화 조향 ↔ °** 표를 만든다 (중간이 선형인지 확인).

**방법 B — 최소 선회반경 \(R_{\min}\) (공간만 있으면 쉬움)**

1. 평평한 바닥에 분필/테이프.
2. 조향 풀락(`±1`) 유지, **아주 느리게** 한 바퀴(또는 반원) 돌린다. (고속·슬립 금지)
3. 궤적 원의 **중심 → 후축 중심**(또는 차체 중심)까지 반지름을 잰다. 그게 \(R_{\min}\).
   - 앞바퀴 자국만 보이면: 후축이 그리는 원이 더 작다. 가능하면 후륜 자국 기준.
4. \(\delta_{\max} = \arctan(L / R_{\min})\) (라디안; 도는 `degrees`).

**방법 C — 둘 다 해서 교차검증**

A로 \(\delta\)를 재고 B로 \(R\)를 재면 \(R \approx L/\tan\delta\)와 비슷한지 확인. 슬립·Ackermann(내외륜 각 다름) 때문에 10~20% 어긋날 수 있음 → PP에는 **A의 \(\delta_{\max}\)**를 `max_steer_angle_rad`에 넣는 편이 낫다.

**실차에 넣을 값 (2026-07-12 실측 반영)**

```text
wheelbase_m: 0.175
max_steer_angle_rad: 0.4266   # δ_max = atan(0.175/0.385) ≈ 24.44°
# R_min SSOT 0.385 m
```

시뮬 YAML 기본은 **0.24 / 0.5574** (같은 \(R_{\min}\)). 보드에서는 위 덮어쓰기.

---

## 5. 코드·설정 위치

| 항목 | 경로 |
|------|------|
| 브릿지 최대 조향·속도·휠베이스 | `src/dracer_sim/config/control_bridge.yaml` |
| Control → Twist 매핑 | `src/dracer_sim/dracer_sim/control_mapping.py` |
| Gazebo 휠베이스·트레드·조향 한도 | `vendor/limo_car/gazebo/ackermann.xacro` |
| **PP 기하·게인 (시뮬 기본)** | `config/lane_control.yaml` (`wheelbase_m`, `max_steer_angle_rad`, `lookahead_x_m`) |
| 실차 트림 | `config/vehicle_config.yaml` → `STEER_TRIM` |
