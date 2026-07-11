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
| 휠베이스 \(L\) | **~174 mm** (PiRacer Pro 계열 FAQ) | **200 mm** | **240 mm** (`ackermann.xacro`) |
| 트레드 \(T\) | 미공개 (전폭 ~140 mm → 트레드 더 작음) | **175 mm** | **168 mm** |
| 외형 (대략) | ~255×140×215 mm | 322×220×251 mm | URDF 베이스 ~190×310×120 |
| 최대 조향각 | 서보 PWM 범위 의존 (문서화 없음) | \(R_\min=0.4\,\mathrm{m}\) 역산 ≈ **27°** | **±30°** (`0.5236` rad) |
| 최소 선회반경 | 미공개 (≈ \(L/\tan\delta\)) | **0.4 m** | ≈ **0.42 m** (\(0.24/\tan 30^\circ\)) |
| 최대 속도 | ESC·배터리 의존 (고속 RC) | **~1 m/s** | 브릿지 `max_linear_speed: 1.2` |
| 질량 | 경량 RC (~1–2 kg대) | ~4.8 kg | URDF `base_mass` 4.34 kg |
| `/control` 부호 | −1=좌 / +1=우 | — | 동일 규약 → Gazebo 좌(+)로 **부호 반전** |

### 출처

| 값 | 출처 |
|----|------|
| D-Racer 휠베이스 ~174.1 mm, 전폭 ~140 mm | [Waveshare PiRacer Pro FAQ](https://www.waveshare.com/wiki/PiRacer_Pro_AI_Kit) (키트 차체가 동일 계열) |
| LIMO \(L=200\), \(T=175\), \(R_\min=0.4\) m | AgileX LIMO 사용자 매뉴얼 |
| Gazebo \(L=0.24\), \(T=0.168\), \(\delta_\max=30^\circ\) | `vendor/limo_car/gazebo/ackermann.xacro`, `src/dracer_sim/config/control_bridge.yaml` |
| D-Racer 액추에이터·PWM | [hardware-board.md](./hardware-board.md) §5 |

---

## 2. 왜 게인을 그대로 옮기면 안 되나

### 2.1 기하 (bicycle / Ackermann 근사)

\[
\kappa \approx \frac{\tan\delta}{L},\quad R \approx \frac{L}{\tan\delta}
\]

- D-Racer \(L\approx0.174\) m vs LIMO 시뮬 \(L=0.24\) m → **휠베이스 약 1.4배**.
- **같은 조향각**이면 LIMO 쪽 선회가 더 완만하다.
- **같은 곡률**을 내려면 LIMO가 더 큰 \(\delta\)가 필요하다.
- 트레드는 LIMO가 넓어 내·외륜 조향각 차(Ackermann angle)가 다르다. 순수 bicycle만 쓰면 영향은 작지만, 슬립·내외륜 속도 차가 커지면 체감된다.

### 2.2 액추에이터 의미

| | D-Racer 실차 | LIMO 시뮬 (`sim_control_bridge`) |
|--|--------------|-----------------------------------|
| 입력 | `steering`, `throttle` ∈ [-1, 1] | 동일 |
| 출력 | PCA9685 PWM (조향 CH0, 스로틀 CH1) | `/cmd_vel` — 조향각(rad) / 선속도 |
| 스케일 | 중립 1500 µs, 조향 ±500 µs 스팬 | `max_steer_angle_rad: 0.5236`, `max_linear_speed: 1.2` |

인터페이스는 같고 **물리량 매핑은 다르다**.

### 2.3 동역학

LIMO는 무겁고 저속(~1 m/s), D-Racer는 가볍고 ESC 기반이라 가속·오버슈트·서보 지연이 다르다.  
시뮬에서 안정이어도 실차에서 진동·과조향이 날 수 있다.

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

---

## 5. 코드·설정 위치

| 항목 | 경로 |
|------|------|
| 브릿지 최대 조향·속도·휠베이스 | `src/dracer_sim/config/control_bridge.yaml` |
| Control → Twist 매핑 | `src/dracer_sim/dracer_sim/control_mapping.py` |
| Gazebo 휠베이스·트레드·조향 한도 | `vendor/limo_car/gazebo/ackermann.xacro` |
| 실차 트림 | `config/vehicle_config.yaml` → `STEER_TRIM` |
