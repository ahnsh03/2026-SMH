# 차선 인지 요건·개발 방향 (갈림·소실·중앙선)

> 개정: 2026-07-14 (안승현) — **In=노란 / Out=흰** 코스↔색 계약 명시  
> 이전: 2026-07-13 밤 — 점선 과투자 회고, **4선→쌍→중앙** 본게임으로 재정렬  
> 범위: `modules/lane_detection.py` (조향·MainPlanner FSM **비범위**)  
> 관련: [lane-perception-topic.md](./lane-perception-topic.md) · [main-planner.md](./main-planner.md) · [competition.md](./competition.md) · [roles.md](./roles.md)

---

## 0. 한 줄 결론 (사용자 가설 검토)

**맞다 — 그리고 더 단순화한다.**  
갈림 = outer 2가닥을 잡고, **고정 차로폭(0.35 m) 11자 평행 레일**로 inner/center를 그린 뒤, stem에서는 **공유 mid**를 유지하다가 갈라지면 각 갈래 `outer±half_w`로 분리한다.

- **중앙선은 outer±half_width가 정답** (양쪽 동등 평균·outer 가중 평균 모두 실패 패턴이 있음).  
- **점선 정밀 피팅은 본게임이 아니다** — far에서 parallel에 가까울 때만 soft hint.  
- **카메라=로봇 전방 중앙:** 근거리 ego 축·stem 공유로 시작점을 고정.

용어: left/right **갈래(fork path)**. 코드 계약은 `RoadBranch` — **msg 개명하지 않음.**

---

## 0.1 코스 ↔ 차선 색 (SSOT)

트랙·미션에서 **어느 코스를 타느냐 = 어느 색 경계를 추종하느냐**다. 혼동 금지.

| 코스 | 미션 구간 | **차선 색** | 인지 중심 출력 | spawn/검증 예 |
|------|-----------|-------------|---------------|---------------|
| **In** | 회전교차로 진입·원호·탈출 | **노란색** | `yellow_centerline` / 노란 4선→`branches` | `in_roundabout_*`, `in_roundabout_exit` |
| **Out** | 일반·S자·좌우 갈림 | **흰색** | `white_centerline` / 흰 4선(또는 2외곽+폭)→`branches` | `out_fork`, 일반 직선 |

- 공통 구간(출발·장애물·도착)도 기본 차선은 **흰**. In은 회전교차로 구간에서만 노란이 주 경계.  
- 인지 모듈은 **흰·노란을 둘 다 검출**한다. **어느 색을 경로로 쓸지**는 `route_mode` / ego 코스 색(`ego_road_color`)·플래너가 고른다.  
- 검증 시: Out 갈림 튜닝에 노란 파라미터만 맞추거나, In 탈출에 흰 V만 보면 **코스 계약 위반**.

코드 정합 (이미 반영된 것):

- 노란 경계 추적: 시계방향 회전 → `required_side="right"` (인코스 우선)  
- 마킹 갈림: `yellow_marks` (In 탈출) / `white_marks`·`road_split` (Out 갈림)  
- MainPlanner: `route_mode:=in` → 노란 우선, `:=out` → 흰 우선 ([main-planner.md](./main-planner.md))

---

## 1. 역할·계약 (불변)

| 층 | 책임 |
|----|------|
| **인지** | 경계 검출 · side/갈래 ID · 소실 상태 → **centerline / `branches[].points`** |
| **MainPlanner** | 표지·`route_mode`·FSM으로 **어느 branch를 탈지** 선택 · PP |

유지: `fork_active`, `RoadBranch[]`(`lateral_rank` 0=가장 왼쪽), `left_visible`/`right_visible`, 색상 centerline, Metric IPM.  
YOLO26n = **표지판만**. 차선 토폴로지에 추가하지 않음.

PP가 필요한 것 = **경로점열** (면이 아님). 이상값은 차로 중앙; 한쪽 소실시 `관측선 ± track_width/2`.

---

## 2. 논의에서 모은 요건 체크리스트

### 2.1 기능 요건

| ID | 요건 | 비고 |
|----|------|------|
| R1 | 단일 차로: 색상별 L/R 2경계 → 1 중앙 | 기존 WonTae 경로 (유지·강화) |
| R2 | 갈림 감지: ego 차로가 전방에서 **2 경로로 분기** | `fork_active`, branches≥2 |
| R3 | 갈림 시 **최대 4 경계**(외L·내L·내R·외R) 검출·유지 | 흰 V / 노란 실+점선 모두 |
| R4 | **2+2 매칭** → 갈래별 중앙 polyline | `branches[0]`, `branches[1]` |
| R5 | 근거리 ego 차로 고정 (카메라 전방 중앙) | 시작 association 앵커 |
| R6 | 한쪽 경계 FOV 소실 시 side ID 유지 + 폭 prior 예측 | 이탈 방지 |
| R7 | 갈림 중 한쪽 바깥선 소실 시에도 **갈래 rank 유지** | 선택 경로 연속 |
| R8 | 출력은 MainPlanner가 이미 읽는 필드만 신뢰도↑ | 스키마 변경 최소화 |

### 2.2 장면 요건

| 장면 | 코스 | 차선 색 | 기대 |
|------|------|---------|------|
| 직선·완만 곡선 | Out / 공통 | **흰** | 2선 + 중앙 |
| Out 갈림 (V, 유도선 무) | **Out** | **흰** | 4선 → 2갈래 중앙 (`white_marks`) |
| 회전교차로 원호 (CW) | **In** | **노란** | 2선 + 중앙; 반경 prior는 **보조** |
| 회전 탈출 분기 (V/점선) | **In** | **노란** | 4선 → 2갈래 중앙 (`yellow_marks`) |
| 곡선+한쪽 소실 | 해당 코스 색 | 동색 | predicted side + 폭 0.35 m |

### 2.3 기하·센서 prior

| Prior | 값·용도 |
|-------|---------|
| 차로폭 | **0.35 m** (YAML `track_width_m`; 주최 원형 내·외측 차 ≈0.356 m와 일치) |
| 카메라 | 전방·대략 차체 중앙 → BEV 하단 ego 축을 L/R 시드 |
| Metric IPM | x≈0.22–1.5 m, \|y\|≤0.77 m, 4 mm/px — **SSOT** |
| 원형 반경 | \(R_\mathrm{in}=1.278\,\mathrm{m}\), \(R_\mathrm{out}=1.634\,\mathrm{m}\) — **회전 유지·소실 보정 보조**, 탈출 V 연결의 주 규칙 아님 |
| 진행 방향 | 회전교차로 **시계방향** → 곡률 부호 prior |

### 2.4 설계 원칙 (합의)

1. **Side / path identity first** — 선 개수=1이어도 반대 side로 재라벨 금지; 소실=`predicted`.  
2. **갈림=토폴로지 분기** — 인지는 후보 2경로를 유지, **선택은 플래너**.  
3. **점선 정밀 연결 ≠ 본게임** — 흔들림에 약함; 경계 4가닥·매칭이 우선.  
4. **연결 시 경향** — 옆 차로로 가로 점프하는 링크 기각 (이미 일부 반영); 곡선은 전진 성분 허용.  
5. **`road_clean` 면적** — 갈래 중앙 검증·셀 추적에 유리하나, **한 덩어리 Y는 커터(마킹) 없이 L/R로 안 갈라짐**. 하이브리드: 마킹으로 자르고 면적/중점으로 중앙.  
6. **시각화** — 튜닝 대상이 가려지면 안 됨 (dash 모드 HSV 깔개 제거 교훈).  
7. **검증** — `tune_lane_detect` + `c` 캡처(`data/captures/lane_tune_logs/`); Gazebo 중복 launch 금지; `teleport`로 구간 이동.
8. **선택 갈래 추종** — 플래너가 LEFT/RIGHT를 잠그면 PP·차로 소실 보정은 **그 `lateral_rank`의 `RoadBranch`만** 보면 된다. 인지는 `branches[0|1]` + `fork_active`를 유지해 선택을 가능하게 하고, 잠금 이후에는 ego centerline / visible / width-prior를 선택 갈래에만 적용하면 판단·제어가 단순해진다. (선택) 플래너→인지 `active_branch_rank` 피드백은 후속; 스키마 필수는 아님.
9. **Parallel-rail (11자) SSOT (2026-07-14 재정렬)** — BEV 스케일·차로폭이 상수이므로:
   - **관측**: 주로 **outer** (흰/노란 바깥·road_clean 가장자리).  
   - **생성**: `inner = outer ± 0.35 m`, `center = outer ± 0.175 m`.  
   - **Stem**: `sep ≈ width` 구간은 양 갈래 **동일 mid**; fork apex 이후 t로 분리.  
   - **점선/고어**: parallel 근처일 때만 soft hint. 행마다 paint snap 금지.

### 2.5 비목표

- MainPlanner FSM / PP 게인 / `route_mode` 로직 변경  
- 차선용 YOLO 추가  
- Metric IPM 범위 임의 확대 (FOV 한계는 추정으로 보완)  
- 인지가 LEFT/RIGHT **미션 결정**

---

## 3. 회고: 점선 Phase A에서 배운 것

| 시도 | 결과 |
|------|------|
| gap/lat/head 수동·자동 스윕 | 후보 **gap=0.30, lat=0.04, head=27** |
| HSV 깔개 미리보기 | 연결 변화 **안 보임** → 마스크 중심 뷰로 수정 |
| 점선만 링크 / 과도한 횡점프 기각 | 연결 0% 또는 곡선 링크 과살 → **가로 점프만** 기각으로 완화 |
| `auto_tune_dash.py` | 재현 가능한 스윕·랭킹 캡처 |

**의미:** 인프라·교훈은 남기되, **다음 스프린트의 KPI는 4선 매칭·갈래 중앙**으로 옮긴다.  
점선 파라미터는 커터 품질용으로 yaml에 유지.

### 3.1 갈림 튜닝 시행착오 (out_fork / in_roundabout_exit)

| 시도 | 증상 | 원인 |
|------|------|------|
| half-width fake inner | 중심이 **¼ 지점** | 합성 inner를 `outer±½w`로 둠 |
| 전역 polyfit / 과직선화 | 곡선·흰 페인트 이탈 | 관측 기하를 직선으로 덮어씀 |
| 행마다 paint snap | 점선 **구불구불**, 커팅 튐 | dash blob마다 u 점프 |
| jump clamp ≤0.10–0.22 m | far **계단 kink** | 곡선 허용치 < 실제 Δu |
| 반대 outer를 `±w`로 날조 | exit far 붕괴 | diverged fork에 stem prior 적용 |
| **outer 가중 0.65 center** | 중심이 **바깥 ⅓**, stem부터 분리 | `0.65o+0.35i` + crossed stem inners → 양 갈래 mid 불일치 |
| 점선=본게임 | exit 안쪽 차선 안 맞음 | 갭·곡선에 약함; PP도 점선 불필요 |

**채택 모델 (parallel-rail):**

1. Outer만 관측(추적)한다.  
2. `inner = outer ± 0.35 m`, `center = outer ± 0.175 m`.  
3. Stem(`sep ≲ 1.25w`): 양 갈래 **동일 mid**, inner=opposite outer.  
4. Fork apex 이후: t∈[0,1]로 11자 레일로 분리 (꼭짓점 lerp).  
5. 점선은 parallel ±0.3w 안에 있을 때만 soft hint.

참고 캡처: `out_fork/.../r06_s067.4_.../preview.png` (분리 시작점·stem 품질 기준).

---

## 4. 목표 파이프라인 (제안)

```
카메라 프레임
  → HSV + Metric IPM
  → road_clean (면적) + 색 경계
  → outer 트랙 2가닥 (L/R)  [점선은 약한 hint]
  → parallel-rail stitch
        stem: 공유 mid · inner=opposite outer
        fork: path = outer ± 0.35 m, center = outer ± 0.175 m
  → RoadBranch[0|1] (선택 후 해당 rank만 PP·소실)
  → LaneDetections
```

단일 차로일 때: Path ego 하나만 → 기존 yellow/white centerline.  
갈림일 때: branches 2개 + fork_active; 단일 centerline은 ego/선택 전 fallback.

원태 `build_road_branches_cells`는 **면적 갈래**로 병행·교차검증; 마킹 4선 매칭이 안정되면 marking-derived center를 우선해도 됨 (노란 갈림에서 이미 실험한 `yellow_marks` 경로와 동일 취지).

---

## 5. 구현 로드맵 (재정렬)

| Phase | 내용 | 완료 기준 | spawn/검증 |
|-------|------|-----------|------------|
| **P0** | 문서·계약·튜너/teleport/캡처 유지 | 이 문서 SSOT | — |
| **P1** | Ego 근거리 L/R 앵커 + BoundaryTrack ID | 곡선에서 한쪽 소실시 ID 안 뒤집힘 | 임의 yaw / 곡선 |
| **P2** | 단일 차로 소실 시 폭 prior 중앙 | center가 차로 안 유지 | 곡선 |
| **P3** | 갈림: 4 경계 검출·2+2 매칭·centers → `branches` | 정지 갈림에서 L/R 쌍·중앙 육안 OK | `out_fork`, `in_roundabout_exit` |
| **P4** | 갈림 중 outer 소실 + rank 유지 | 헤딩 틀어도 선택 갈래 중앙 연속 | 저속 주행 |
| **P5** | `road_clean` 셀과 교차검증·fallback | 마킹 약할 때 셀 중앙으로 버팀 | 동 구간 |
| **P6** | (선택) 회전 중 \(R\) prior 약한 검증 | 원호에서 이상 곡률 기각 | 회전 구간 |
| **P7** | 원태 핸드오프 | 튜너 모드 + 이 문서 | — |

**우선순위:** P1→P2→**P3**가 미션 직결. 점선 auto-tune은 P3 커터가 부족할 때만 재개.

### 진행 메모 (2026-07-13 밤 / 07-14 보강)

- **코스↔색 SSOT 문서화 (07-14):** In=노란, Out=흰 — 튜닝·코드는 이미 이 전제로 동작 (`in_roundabout_exit`→`yellow_marks`, `out_fork`→`white_marks`). §0.1·competition·main-planner·lane-drive에 명시.
- `scripts/vision_tune/auto_tune_fork.py` — dash와 동일 패턴의 스윕·랭킹·캡처 (`data/captures/lane_tune_logs/auto_fork/`)
- P1: 근거리 ego 축 보너스 + fork near-zone 시드
- P2: `centerline_from_boundaries(..., synthesize_missing=True)` 폭 prior
- P3: 노란 4선 매칭 / 흰 2외곽 발산→쌍 / `road_split` 이중 코리도 폴백 → `RoadBranch[]`
- 검증 스냅: `in_roundabout_exit` yellow_marks pairs=2; `out_fork` white_marks pairs=2
- yaml `detect_tune` fork_* — exit_v2 기준 assoc=0.10 / min_rows=14 / far=0.45 / gap=12 / near=0.28 / width=0.35

### 시각화 (P3용)

- `fork` / `fork_left` / `fork_right`: **외·내·중앙** 색 분리 (이미 골격 있음)  
- 단일: L/R + predicted 스타일  
- `c` 캡처로 피드백

### 수락 기준

1. 단일: 소실 시에도 L/R ID·중앙 안정  
2. `out_fork` / `in_roundabout_exit` 정지: branches 2, rank 0/1, 중앙이 각 차로 안  
3. 갈림 진입 후 바깥선 소실에도 rank·중앙 연속  
4. MainPlanner 직선 회귀 깨지지 않음  
5. 계약 필드 유지

---

## 5.1 Spawn 단위 테스트 (Out fork / In exit)

계약:

| 항목 | 값 |
|------|-----|
| LEFT | rank **0** |
| RIGHT | rank **1** |
| `route_mode:=in` | 노란 우선 + 노란 fork layers |
| `route_mode:=out` | 흰 중앙 + 흰/`road_split` layers |
| 선택 후 | PP는 **선택 레이어만** (`_selected_layer_path`) |

오프라인 / 라이브 하네스:

```bash
# 컨테이너 안
python3 scripts/drive_test/fork_spawn_unit.py --mode offline --scenario all
# 시뮬 bringup 후
python3 scripts/drive_test/fork_spawn_unit.py --mode live --scenario out_left --duration 8
./scripts/dev_container.sh teleport out_fork   # 수동 텔레포트도 가능
```

로그: `data/captures/fork_drive_logs/<stamp>/`

참고 캡처(인지 LOCK): `auto_fork/out_fork/verify/current.png` (2026-07-14 확정).

---

## 6. 즉시 다음 액션

1. P3 설계 구현 착수: **근거리 2선 앵커 → 전방 4선 분기 → 쌍 중앙 → RoadBranch**.  
2. 검증: `teleport out_fork` / `in_roundabout_exit` + `tune_lane_detect --mode fork` + `c`.  
3. 점선 yaml(`gap=0.30` 등)은 커터용으로 유지, KPI에서 제외.

---

## 7. 참고 산출물

- 캡처: `data/captures/lane_tune_logs/` · `auto_dash/`  
- 튜너: `scripts/vision_tune/tune_lane_detect.py` (`c`=번들)  
- 자동스윕: `scripts/vision_tune/auto_tune_dash.py`  
- 텔레포트: `./scripts/dev_container.sh teleport <pose>`
