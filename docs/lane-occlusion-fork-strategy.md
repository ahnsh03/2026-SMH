# 차선 인지 요건·개발 방향 (갈림·소실·중앙선)

> 개정: 2026-07-15 (안승현) — **In 탈출 직전 mask 게이트** · Out 표지 게이트 · tip_mode · 원형 fork-PP 억제  
> 이전: 2026-07-15 — Out 표지 게이트 · In/Out tip_mode 분리 · 원형 fork-PP 억제  
> 이전: 2026-07-14 — 용어·코스색·viz·forced_turn 전면 정합 · Out=흰만 · In=노란우선  
> 범위: `modules/lane_detection.py` + `pipeline.py` 갈림 게이트 (조향 게인 튜닝은 [main-planner.md](./main-planner.md))  
> 관련: [lane-perception-topic.md](./lane-perception-topic.md) · [main-planner.md](./main-planner.md) · [hsv-profiles.md](./hsv-profiles.md) · [competition.md](./competition.md) · [roles.md](./roles.md)

---

## 0. 용어·규약 SSOT (브랜치 작업 시 항상 여기부터)

이 문서가 **갈림·코스 색·시각화 테스트 용어**의 단일 진실 원천이다.  
실행 명령: [dev-environment.md](./dev-environment.md) · 플래너: [main-planner.md](./main-planner.md).

문서·대화에서는 아래 **한국어 용어**를 쓴다. 괄호는 **코드 식별자**(와이어·함수명)이며, **개명하지 않는다**.

### 0.0 갈림길 장면은 딱 두 가지

| 문서 용어 | 코스·색 | 무엇이 갈라지나 | spawn / FSM |
|-----------|---------|-----------------|-------------|
| **Out 갈림** | Out · **흰** | 유도선(내부 V) 없는 **진짜 갈림** — 좌/우 차로 | `out_fork` · `FORK_TURN` |
| **In 탈출 분기** | In · **노란** | 회전교차로에서 **유지(순환) vs 탈출** | `in_roundabout_exit` · `ROUNDABOUT_EXIT*` |

그 외(“분기”, “fork”, “exit”를 서로 바꿔 쓰기)는 위 둘을 가리킬 때만 쓰고, **어느 장면인지 한 단어로 붙인다.**

### 0.1 계층 용어

| 말할 때 | 뜻 | 코드 (유지) |
|---------|-----|-------------|
| **갈림 활성** | 전방에 **추종 후보 갈래가 2개**임 | `fork_active` |
| **갈래** | 차량이 탈 **한 경로의 센터라인** (좌=0, 우=1) | `RoadBranch` / `LaneDetections.branches[]` · `lateral_rank` |
| **차로 쌍** | 한 갈래를 가두는 **바깥+안 경계 2선** (+중앙) | `ForkLanePair` / `fork_lane_pairs` |
| **주 경계 코스** | DP가 먼저 이은 L/R 경계 1쌍 | `*_left` / `*_right` (primary) |
| **보조 경계 코스** | DP를 **한 번 더** 풀어 얻은 반대 갈래 L/R | `*_alt_*` · `find_alternate` (**alt = alternate**, “대체 갈림 알고리즘” 아님) |
| **이중 코스 → 차로 쌍** | 주+보조 = 경계 4선 → 갈래 2개 | `fork_lane_pairs_from_dual_courses` |
| **갈림 분리 소스** | 이번 프레임의 갈래가 어디서 왔는지 | `fork_split_source` (아래) |
| **선택 갈래** | 표지/FSM이 잠근 한쪽만 남김 | `active_branch_rank` · `active_lane` · locked |
| **갈림 인지 게이트** | Out에서 표지 본 뒤에만 갈림 발행 | `enable_fork` / debug `fork_on` · `out_fork_require_sign` |
| **tip 모드** | In/Out 갈림 rail tip 후처리 분리 | `tip_mode=in_curve\|out_forward` |

`fork_split_source` 값 읽는 법:

| 값 | 장면 쪽 의미 |
|----|----------------|
| `road_split_marks` | **Out 갈림** 주력 — 도로 마스크 이중 코리도 |
| `white_marks` / `white_alt_marks` | Out 폴백 (마킹 트랙 / 흰 이중 코스) |
| `yellow_alt_marks` | **In 탈출 분기** 주력 — 노란 주+보조 DP(=4선) |
| `yellow_marks` | In 폴백 — 노란 마킹 트랙 |
| `cells` | 셀 추적 폴백 |

금지에 가까운 혼동:

- **branch(Git)** ≠ 갈래 `RoadBranch`
- **alt** ≠ “다른 인식 파이프라인 전체” → **같은 DP의 두 번째 경계 코스**
- **fork** 단독 → Out 갈림인지 In 탈출인지 반드시 구분

### 0.0.1 갈림(분기) vs 합류(merge join) — 인식 정책

**같은 “선이 두 갈래로 보인다”**여도, 기하·미션 관점에서 처리가 다르다.

| 상황 | 예 | 인식·플래너 | spawn 예 |
|------|-----|-------------|----------|
| **내 차로에서 분기** | Out 갈림, In 탈출(유지 vs 탈출) | `fork_active=True`, 갈래 **2개** 유지 → 표지/FSM으로 rank 잠금 | `out_fork`, `in_roundabout_exit` |
| **다른 차로가 내 차로로 합류** | Out 갈림 합류·In→Out 합류 | **갈림으로 취급하지 않음** — ego 근거리에 닿지 않는 far-only spur 무시 | `out_fork_merge_*`, `in_out_merge`, `out_in_merge` |

판별 직관 (BEV / base_link):

- **분기**: ego bumper 근처(작은 x)에서도 **두 갈래 모두** near point가 있거나, stem에서 공유 mid 후 갈라짐.
- **합류**: ego 차로는 near에 **연속**인데, 옆에서 들어오는 페인트는 **멀리(far)에서만** 처음 보임 → 합류 spur.

**코드 반영 (2026-07-14)**

| 계층 | 구현 | 한계 |
|------|------|------|
| **선택 갈래 정책** | `active_lane.suppress_merge_spur_branches` — near x ≤ **0.55 m** 에 점 ≥ **3** 인 갈래만 유지 (`MERGE_SPUR_*`) | explore·잠금 전 단계에서 `branches`/`fork_active` 정리 |
| **적용 시점** | `apply_active_lane_policy(..., active_branch_rank=None)` — 플래너가 아직 rank를 안 잡은 **탐색** 구간 | `pipeline.step` → `detect(..., active_branch_rank=...)` 경로 |
| **미구현·보강 여지** | `lane_detection` fork pair 생성 단계에서 merge 전용 게이트 없음 | merge spur가 near 노이즈로 3점 이상이면 **오탐 fork** 가능 → spawn별 튜닝·오프라인 harness 추가 권장 |

**OUT 표지 게이트 (2026-07-15):** MainPlanner가 `detect(..., enable_fork=)` 를 켠다.  
평소 `false`(흰만) → 방향 표지 관측·hold 동안만 `true` → `FORK_TURN` 종료 시 hold 클리어.  
`forced_turn`은 기본으로 이 게이트를 우회하지 않음 (`out_fork_forced_turn_arms`).

합류 구간 검증·고정 파라미터·창 설정: **[fork-test-pipeline.md](./fork-test-pipeline.md)** (구간별 SSOT).

---

## 0.2 한 줄 결론 (기하)

Out 갈림·In 탈출 분기 모두: outer를 잡고 **고정 차로폭(0.35 m) 11자 평행 레일**로 inner/center를 그린 뒤, stem에서는 **공유 mid**, 갈라지면 각 갈래 `outer±half_w`.

- **중앙선은 outer±half_width가 정답** (양쪽 동등 평균·outer 가중 평균 모두 실패 패턴이 있음).  
- **점선 정밀 피팅은 본게임이 아니다** — far에서 parallel에 가까울 때만 soft hint.  
- **카메라=로봇 전방 중앙:** 근거리 ego 축·stem 공유로 시작점을 고정.

---

## 0.3 코스 ↔ 차선 색 (SSOT)

트랙·미션에서 **어느 코스를 타느냐 = 어느 색 경계를 추종하느냐**다. 혼동 금지.

| 코스 | 미션 구간 | **차선 색** | 인지 중심 출력 | spawn/검증 예 |
|------|-----------|-------------|---------------|---------------|
| **In** | 진입(흰)→원·탈출(노란)→합류(흰) | **노란이 있으면 노란 우선**, 없으면 흰 | `yellow`/`white_centerline` · 탈출 시 노란 갈래 | `in_roundabout_*` |
| **Out** | 일반·S자·**Out 갈림** | **흰색만** | `white_centerline` / `road_split` 갈래 | `out_fork` |

- 공통 구간(출발·장애물·도착)도 기본은 **흰**.  
- `route_mode:=in`이어도 **노란이 안 보이면 흰을 탄다**(진입·합류). 노란이 잡히면 그걸 우선.  
- Out에서는 노란이 보여도 **절대 추종하지 않음**.

코드 정합 (이미 반영된 것):

- 노란 경계 추적: 시계방향 회전 → `required_side="right"` (인코스 우선)  
- **In 탈출 분기**: `yellow_alt_marks` 우선 (주+보조 DP) → `yellow_marks` → `road_split` / 흰 폴백  
- **Out 갈림**: **흰만** — `road_split_marks` / `white_*` (노란 갈래·노란 `_color_path` **금지**)  
- MainPlanner: `route_mode:=out` → 흰만 · `:=in` → **노란이 있으면 노란 우선**, 없으면 흰 (진입·원·합류가 자동 전환)
- 잠금 후 active_lane도 같은 코스 계약 (`prefer_yellow` on `LaneDebugFrame`)

---

## 1. 역할·계약 (불변)

| 층 | 책임 |
|----|------|
| **인지** | 경계 검출 · side/갈래 ID · 소실 상태 → **centerline / `branches[].points`(갈래)** |
| **MainPlanner** | 표지·`route_mode`·FSM으로 **어느 갈래를 탈지** 선택 · PP |

유지: `fork_active`, `RoadBranch[]`(`lateral_rank` 0=가장 왼쪽), `left_visible`/`right_visible`, 색상 centerline, Metric IPM.  
YOLO26n = **표지판만**. 차선 토폴로지에 추가하지 않음.

PP가 필요한 것 = **경로점열** (면이 아님). 이상값은 차로 중앙; 한쪽 소실시 `관측선 ± track_width/2`.

---

## 2. 논의에서 모은 요건 체크리스트

### 2.1 기능 요건

| ID | 요건 | 비고 |
|----|------|------|
| R1 | 단일 차로: 색상별 L/R 2경계 → 1 중앙 | 기존 WonTae 경로 (유지·강화) |
| R2 | Out 갈림·In 탈출에서 ego 전방 **갈래 2개** | `fork_active`, `branches`≥2 |
| R3 | 그때 **최대 4 경계**(외L·내L·내R·외R) 검출·유지 | Out 흰 · In 노란 |
| R4 | **2+2 매칭** → 갈래별 중앙 polyline | `branches[0]`/`[1]` (=갈래) |
| R5 | 근거리 ego 차로 고정 (카메라 전방 중앙) | 시작 association 앵커 |
| R6 | 한쪽 경계 FOV 소실 시 side ID 유지 + 폭 prior 예측 | 이탈 방지 |
| R7 | 갈림 중 한쪽 바깥선 소실 시에도 **갈래 rank 유지** | 선택 경로 연속 |
| R8 | 출력은 MainPlanner가 이미 읽는 필드만 신뢰도↑ | 스키마 변경 최소화 |

### 2.2 장면 요건

| 장면 | 코스 | 차선 색 | 기대 |
|------|------|---------|------|
| 직선·완만 곡선 | Out / 공통 | **흰** | 2선 + 중앙 |
| **Out 갈림** (유도선 없는 진짜 갈림) | **Out** | **흰** | 차로 쌍×2 → 갈래 2 (`road_split` 주력) |
| 회전교차로 원호·유지 (CW) | **In** | **노란** | 2선 + 중앙 (갈림 아님) |
| **In 탈출 분기** (유지 vs 탈출) | **In** | **노란** | 주+보조 경계 코스 → 갈래 2 (`yellow_alt` 주력) |
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
2. **Out 갈림·In 탈출 = 토폴로지상 갈래 2개** — 인지는 후보 유지, **선택은 플래너**.  
3. **점선 정밀 연결 ≠ 본게임** — 흔들림에 약함; 경계 4가닥·매칭이 우선.  
4. **연결 시 경향** — 옆 차로로 가로 점프하는 링크 기각 (이미 일부 반영); 곡선은 전진 성분 허용.  
5. **`road_clean` 면적** — 갈래 중앙 검증·셀 추적에 유리하나, **한 덩어리 Y는 커터(마킹) 없이 L/R로 안 갈라짐**. 하이브리드: 마킹으로 자르고 면적/중점으로 중앙.  
6. **시각화** — 튜닝 대상이 가려지면 안 됨 (dash 모드 HSV 깔개 제거 교훈).  
7. **검증** — `tune_lane_detect` + `c` 캡처(`data/captures/lane_tune_logs/`); Gazebo 중복 launch 금지; `teleport`로 구간 이동.
8. **선택 갈래 추종 (구현됨)** — 표지/FSM이 LEFT/RIGHT를 잠그면:
   - 인지: `active_branch_rank` 피드백 → `modules/active_lane.py`가 **선택 `ForkLanePair`/`RoadBranch`만** 남기고 `fork_active=False`로 ego 단일차로 투영.
   - 플래너: 2갈래면 `_selected_layer_path`, 잠금 후 단일이면 `out_fork_ego_follow_rank*`.
   - 시각화: `/debug/planner`의 `rank=`로 프리뷰 focus 자동 동기 (`lane_preview_node`).
   - **합류**: 근거리(ego)에 닿지 않는 far-only spur는 `suppress_merge_spur_branches`로 무시.
   - 편측 페인트 소실: 선택 pair의 parallel-rail 폭 prior(기존 stitch)로 중앙 유지.
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

원태 `build_road_branches_cells`는 **면적 갈래**로 병행·교차검증; 마킹/이중 코스가 안정되면 marking-derived center를 우선 (**In 탈출** = `yellow_alt_marks` 주력, **Out 갈림** = `road_split`/`white_*`).

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

- **코스↔색 SSOT (유지):** Out=흰만 · In=**노란이 있으면 노란 우선 / 없으면 흰** (흰→노란 원→흰 합류). fork: Out=`road_split_marks`, In=`yellow_alt_marks`.
- 검증 스냅: Out `src=road_split_marks` pairs=2; In `src=yellow_alt_marks` pairs=2
- `scripts/vision_tune/auto_tune_fork.py` — scene별 `prefer_yellow` (Out=False / In=True)
- P1: 근거리 ego 축 보너스 + fork near-zone 시드
- P2: `centerline_from_boundaries(..., synthesize_missing=True)` 폭 prior
- P3: In `yellow_alt` / Out `road_split` → `RoadBranch[]`
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

## 5.1 Spawn 단위 테스트 (Out 갈림 / In 탈출 분기)

**구간별 파이프라인 SSOT:** [fork-test-pipeline.md](./fork-test-pipeline.md) (합류 무시·viz·고정값 포함).

계약 (SSOT §0·§0.3):

| 항목 | 값 |
|------|-----|
| LEFT / RIGHT | rank **0** / **1** |
| `route_mode:=out` | **흰만** (`prefer_yellow` 강제 False) · fork=`road_split`/`white_*` |
| `route_mode:=in` | **노란이 있으면 노란 우선**, 없으면 흰 · fork=`yellow_alt`/`yellow` 후 폴백 |
| 선택 후 | PP는 **선택 갈래만** (`_selected_layer_path` / locked ego) |
| `forced_turn` | 카메라 표지 **방향·rank만** 고정 (`sign_ignored(forced=…)`) · **기본 YAML에서 `fork_on`을 켜지 않음** |
| Out 갈림 인지 | 실표지 hold, 또는 `config/main_planner.yaml`에서 `out_fork_forced_turn_arms: true` / `out_fork_require_sign: false` (재시작) |
| Out 실표지 | `out_fork_require_sign` — 표지 보인 뒤 `out_fork_sign_hold_sec`(기본 3s)만 `fork_on=1` |
| In 원형 | `circle_ignore_fork_for_control` — 순환 중 mask→fork PP **제어 전환 금지** (탈출 카운트용 fork는 유지) |

```bash
# 오프라인
python3 scripts/drive_test/fork_spawn_unit.py --mode offline --scenario all

# Out 갈림 (라이브) — forced_turn만으로는 fork_on=0 (기본 YAML)
./scripts/dev_container.sh sim-bringup spawn_pose:=out_fork view:=none
./scripts/dev_container.sh sim-auto route_mode:=out forced_turn:=left viz:=lane
# 표지 없이 fork 패널이 필요하면 config/main_planner.yaml 의 route 블록에서:
#   out_fork_forced_turn_arms: true   # forced_turn이 갈림 인지까지 켬
#   또는 out_fork_require_sign: false # Out에서도 상시 fork 탐색
# 저장 후 sim-auto 재시작 (launch 인자 아님)
# Out 실표지 게이트 확인 (forced 없음 → 표지 전에는 fork_on=0)
./scripts/dev_container.sh sim-auto route_mode:=out viz:=lane

# In 탈출 / 원 유지
./scripts/dev_container.sh sim-bringup spawn_pose:=in_roundabout_exit view:=none
./scripts/dev_container.sh sim-auto route_mode:=in forced_turn:=left viz:=lane   # 탈출
./scripts/dev_container.sh sim-auto route_mode:=in forced_turn:=right viz:=lane  # 원 유지
```

로그: `data/captures/fork_drive_logs/<stamp>/` · 캡처: `auto_fork/out_fork/verify/current.png` · 코스 tip 비교: `fork_rail_sweeps/_course_split_compare/`

---

## 5.1.1 Out 표지 게이트 · In/Out tip 분리 (2026-07-15)

| 주제 | 동작 |
|------|------|
| **Out 평상 흰 차로** | `enable_fork=false` → marking/cell 갈림 **미발행**. mask COM이 가짜 갈림에 흔들리지 않음 |
| **Out 표지 후** | 표지 프레임마다 타이머 갱신 → hold 초 동안만 갈림 탐색·`FORK_TURN` 가능 |
| **Out hold 만료** | 다시 `fork_on=0`. 이미 `FORK_TURN` 중이면 완료까지 유지 |
| **In tip** | `tip_mode=in_curve` — 곡선 출구 (outer→FOV 측, paint tip) |
| **Out tip** | `tip_mode=out_forward` — stem X→parallel 연속 보간, gore mid shelf 금지 |
| **In 원형 제어** | `ROUNDABOUT_CIRCLE`에서 `_forkish_for_mask=False` — `roundabout_circle_mask` 유지, 이벤트만 카운트 (debounce ↑) |

YAML: `config/main_planner.yaml` → `route.out_fork_*`, `roundabout.circle_ignore_fork_for_control`, `branch_on_frames`.

---

## 5.1.2 In 탈출 직전 시점 — yellow mask 게이트 (2026-07-15 lock)

**목표:** 회전교차로 **유지(좌) vs 탈출(우)** 직전을 다른 시점과 혼동하지 않고 포착.  
**범위:** 시점 감지 규칙만 SSOT. L/R 선택·반폭 추종·`yellow_alt` PP 연결은 **후속**.

### 샘플 (IN bag `from_bag/in` · mosaics `raw_hsv_masks/in/`)

| id | stem | 비고 |
|----|------|------|
| **0008** | `frame_20260715_045830_994784_0714` | far Y gore 깊음, ego도 Y |
| **0009** | `frame_20260715_045837_397668_0734` | 분기 꼭지 가깝고 양갈래 뚜렷 |
| **0019** | `frame_20260715_045902_029012_1174` | ego는 단일 CC여도 **노란 far dual** 선명 (재접근/다른 랩) |

주의: **ego blob CC 개수만으로는 불가** (0019는 morph 후 한 덩어리). 주도 신호 = **yellow**.

### 밴드 정의 (Metric IPM BEV, 예: 271×386)

| 밴드 | 행 비율 (이미지 위→아래, v=0=far) |
|------|----------------------------------|
| far | 5% – 45% |
| mid | 40% – 70% |
| near | 70% – 95% |
| top20 | 0% – 20% |

`free = road_raw & ~dilate(yellow)` (`road_raw` = black\|red\|cyan, HSV SSOT는 [hsv-profiles.md](./hsv-profiles.md)).

### 주 게이트 `in_circle_fork_moment`

전제: `route_mode=in` / `prefer_yellow`.

**hard_base (노란 통계 AND):**

1. far에서 yellow **row-run ≥ 2** 인 행 비율 **≥ 70%** (`far_dualY`)
2. 그 dual 행들의 좌·우 yellow mid 간격 median **≥ 55 px** (`far_sep`)
3. far yellow CC 상위 면적비 `a2/a1` **≥ 0.20** (`ya2_ratio`)
4. mid `dualY` **≥ 35%**

**hard (런타임 SSOT = hard_base + AND)** — 단일 차로 L/R 노란(캡처 0004) 오탐 억제:

5. far에서 `free=road&~dilate(yellow)` dual-run 비율 **≥ 70%** (`far_dualF`)
6. `span_ratio = far_free_span / near_free_span` **≥ 1.3**

### 보강 (OR, 있으면 신뢰↑)

- `span_ratio ≥ 1.5` **또는** top20% `dualFree ≥ 60%` (`boosted`)

### 명시적 제외

| 실패 패턴 | 예 (캡처 id) | 이유 |
|-----------|--------------|------|
| far dualY≈0 + free dual만 큼 | 0010, 0017–18, 0020 | 도로 gore/개방 — **노란 없음** |
| near/mid만 dual, far 약함 | 0002 | 곡선 점선 노이즈 |
| farY 중간 + gore만 큼 | 0015 | hard에서 farY 70% 미만으로 탈락 |

### bag 수치 요약 (2026-07-15 오프라인)

| id | farY% | midY% | far_sep | ya2/ya1 | spanR | topF% | hard |
|----|-------|-------|---------|---------|-------|-------|------|
| 0008 | 100 | 100 | 123 | 0.46 | 2.2 | 96 | hard |
| 0009 | 88 | 53 | 185 | 0.87 | 2.5 | 100 | hard |
| 0019 | 100 | 100 | 96 | 0.40 | 2.6 | 65 | hard |
| 0021 | 100 | (farF 100) | 107 | 0.99 | 2.0 | 100 | hard (재접근, 미라벨·허용) |
| 0007 | 100 | 70 | 73 | 0.73 | 0.83 | 43 | hard_base only (early arm) |
| 0004 | 100 | — | 92 | 0.84 | **1.03** | 100 | hard_base만 — **span으로 기각** |
| 0010 | 0 | 7 | 0 | 0 | 1.2 | 41 | miss |
| 0015 | 55 | 98 | 247 | 0.23 | 2.3 | 100 | miss |
| 0002 | 33 | 100 | 43 | 0.13 | 0.8 | 0 | miss |

### 시간 안정화

- `hard`가 **K프레임 연속** (플래너 `branch_on_frames`≈6과 맞춤) → rising `in_circle_fork_moment`
- falling: far dualY 붕괴 (0010류)로 자연 해제
- **0007**은 `hard_base`만 (span 부족) — 조기 arm이 필요하면 `hard_base` debounce를 별도 arm에 쓰고, 조향 게이트는 **`hard`만** 사용

### 기존 갈림·제어와의 관계

| 계층 | 역할 |
|------|------|
| 본 게이트 | **시점 포착** (yellow 통계). blob ego와 독립 플래그로 둘 것 |
| `yellow_alt_marks` / `fork_lane_pairs_from_dual_courses` | 갈래 2개 geometry (후속·기존 legacy) |
| MainPlanner | L=탈출 / R=유지 잠금; circle 중 `circle_ignore_fork_for_control`이면 fork는 **이벤트만** |

후속(미구현): 선택 rank outer±half_w · 노란 커팅 단일 갈래 · 기존 `RoadBranch` PP.

### 오프라인 검증 (스크립트)

```bash
# 2026-smh-sim
PYTHONPATH=scripts/vision_tune:src/inference python3 scripts/vision_tune/score_in_fork_moment.py \
  --folder data/captures/from_bag/in \
  --csv data/captures/raw_hsv_masks/in_fork_moment_scores.csv
```

런타임 API: [`perception/fork/moment.py`](../src/inference/inference/modules/perception/fork/moment.py) `score_in_circle_fork_moment`  
오프라인: [`scripts/vision_tune/score_in_fork_moment.py`](../scripts/vision_tune/score_in_fork_moment.py)  
데이터·코드 맵: [fork-moment-detection.md](./fork-moment-detection.md)  
입력/지표: yellow · free=(black|red|cyan)&~dilate(yellow) → far/mid dualY, far_sep, ya2_ratio, spanR, top_dualFree.

OUT 흰 갈림에 이 규칙 **금지** (`prefer_yellow` 가드).

---

## 5.1.3 Out 갈림 직전 시점 — white + road 게이트 (2026-07-15)

**목표:** Out **진짜 갈림** 직전(흰 유도선/고어가 보이며 차로가 벌어지는 순간)을 직선·곡선·개방 영역과 혼동하지 않고 포착.  
IN 탈출(`§5.1.2`)과 구별할 필요는 없음. **분기가 아닌 구간의 오탐만 금지.**

### 샘플 (`from_bag/out` · `raw_hsv_masks/out/`)

| id | stem | 비고 |
|----|------|------|
| **0011** | `frame_20260715_045046_248624_1758` | stem→Y, 상단 gore 뚜렷 |
| **0012** | `frame_20260715_045053_939644_1784` | 분기 더 가깝고 ego Y 넓음 |
| 0013 | `…045104…1976` | **분기 직후** 한 갈래 선택 — 직전 게이트에서는 miss가 정상 |

주도 신호 = **white** (OUT 차선). road dual/span만 쓰면 개방·합류에서 대량 오탐.

### 런타임 hard `out_fork_moment` (필수 AND)

전제: `route_mode=out` / `prefer_yellow=False`.

| # | 조건 | 임계 |
|---|------|------|
| 1 | far white dual-run 비율 | **≥ 90%** |
| 2 | mid white dual-run | **≥ 70%** |
| 3 | far white 좌·우 mid 간격 `sepW` | **≥ 150 px** (단일 차로 L/R ≈80–100px보다 큼) |
| 4 | far white CC `a2/a1` | **≥ 0.50** |
| 5 | far road dual-run | **≥ 80%** |
| 6 | `span_road = far_road_span / near_road_span` | **≥ 2.2** |
| 7 | 전체 `road_pct` | **≥ 28%** (직전 Y가 마스크에서 넓음; 0013=21%로 걸러짐) |

밴드 정의는 §5.1.2와 동일 (far 5–45%, mid 40–70%, near 70–95%).

### 왜 이렇게 나뉘나

| 실패 패턴 | 예 | 차단 조건 |
|-----------|-----|-----------|
| 도로만 벌어짐·흰 갈림 약함 | 0010, 0015, 0021… | farWd / sepW |
| 평행 흰 레일만 (차로폭 sep) | 0001, 0004, 0025… | **sepW ≥ 150** |
| 개방/원 주변 넓은 free | 다수 | white dual 필수 |
| 분기 통과 후 한 줄기 | 0013 | road_pct / span 조합 |

### bag 검증 (2026-07-15, n=37)

```bash
PYTHONPATH=scripts/vision_tune:src/inference python3 scripts/vision_tune/score_out_fork_moment.py \
  --folder data/captures/from_bag/out \
  --csv data/captures/raw_hsv_masks/out_fork_moment_scores.csv
```

| 결과 | |
|------|--|
| 양성 0011·0012 hard | **PASS** |
| nontarget hard FP | **0 (PASS)** |
| hard hit | **0011, 0012 only** |

런타임 API: [`perception/fork/moment.py`](../src/inference/inference/modules/perception/fork/moment.py) `score_out_fork_moment`  
오프라인: [`scripts/vision_tune/score_out_fork_moment.py`](../scripts/vision_tune/score_out_fork_moment.py)  
데이터·코드 맵: [fork-moment-detection.md](./fork-moment-detection.md)

후속: `enable_fork`/표지 게이트와 AND로 연결 · L/R은 기존 `road_split` / `white_*` · FSM.

### 5.1.3+ Stretch 통합 (2026-07-16)

흰 tip(`out_fork_moment`)만으로는 미션 **구간**(~1690–1783)을 덮지 못한다.  
ego_blob Y-stretch와 fuse한 **`score_out_fork_capture`** 를 bag으로 검증한다.

| | |
|--|--|
| 문서 | [out-ego-fork-shape.md](./out-ego-fork-shape.md) |
| CLI | `scripts/vision_tune/score_out_fork_capture.py --from-bag out` |
| Fuse | `capture = ego.hard ∨ (moment.hard ∧ ego soft/hard)` |

---

## 5.1.4 판단 구조 — OUT 표지∧capture · IN moment 패스 정책 (2026-07-16)

**검토 결론: 요청 구조 채택.**

| 장면 | 트리거 | L/R |
|------|--------|-----|
| **Out 갈림** | **표지 인식 ∧ `out_fork_capture`** (`decide_out_fork_arm`) | 표지가 rank 잠금 (0=좌, 1=우) |
| **In 탈출** | **`in_circle_fork_moment`만** (표지 불필요) | **1회 rising → 우(rank1)=원 유지** · **2회 rising → 좌(rank0)=탈출** |

```
OUT:  sign_window ──┐
      capture ──────┼──► enable_fork → yellow_alt/road_split follow → FORK_TURN
                    │
IN:   moment.hard ──(K debounce rising)──► pass1 keep_rank=1
                                         ► pass2 exit_rank=0 → EXIT_READY/EXIT
```

왜 AND(OUT): 표지만이면 평상 흰 차로에서 가짜 fork, capture만이면 표지 없는 먼 stretch에서 arm.  
왜 IN에 moment만: 회전교차로 유지/탈출은 대회에 방향 표지가 없음. yellow far dual이 SSOT (§5.1.2). **OUT ego capture를 IN에 쓰지 말 것.**

코드:

| | |
|--|--|
| 정책 | [`perception/fork/judgment.py`](../src/inference/inference/modules/perception/fork/judgment.py) |
| 플래너 | `MainPlanner._fork_perception_allowed` · `_apply_in_moment_pass` · `_wants_roundabout_exit` |
| YAML | `route.out_fork_require_capture` · `roundabout.in_exit_use_moment` / `in_keep_passes` / `in_keep_branch_rank` |
| 인지 플래그 | `LaneDetections.out_fork_capture` · `.in_circle_fork_moment` (blob detect) |

원형 중 `circle_ignore_fork_for_control`: moment가 rank를 **고른 뒤에는** fork PP 허용 (유지 갈래 / 탈출 갈래 추종 ON).

상세 bag 검증: [out-ego-fork-shape.md](./out-ego-fork-shape.md).

---

## 5.2 통합 결정 (2026-07-14)

| 소스 | 역할 | 채택 |
|------|------|------|
| **origin/main PR #32 (원태)** | 노란 점선 연결·합류 L/R 감점 | **베이스** |
| **parallel-rail 차로 쌍** | `ForkLanePair` → 갈래(`RoadBranch`) | **플래너 SSOT** |
| **WonJung 이중 DP** | 주+**보조** 경계 코스 (`find_alternate`) | **채택** — In 탈출 주력, Out는 `road_split` 실패 시만 |

우선순위 (§0 용어 기준):

1. **In 탈출 분기**: `yellow_alt_marks` → `yellow_marks` → `road_split` / 흰 폴백
2. **Out 갈림**: **`road_split_marks` / `white_*`만** (노란 갈래·노란 `_color_path` 금지)

검증: `pytest` fork/planner/active_lane · `fork_spawn_unit.py --mode offline --scenario all`

---

## 6. 즉시 다음 액션

1. P3 설계 구현 착수: **근거리 2선 앵커 → 전방 4선 분기 → 쌍 중앙 → RoadBranch**.  
2. 검증: `teleport out_fork` / `in_roundabout_exit` + `tune_lane_detect --mode fork` + `c`.  
3. 점선 yaml(`gap=0.30` 등)은 커터용으로 유지, KPI에서 제외.
4. **In 탈출 직전:** §5.1.2 `in_circle_fork_moment` 게이트를 blob/legacy에 플래그로 이식 → 이후 L/R 추종(반폭·yellow_alt PP). 조향은 게이트 SSOT **이후** 작업.

---

## 7. 참고 산출물

- 캡처: `data/captures/lane_tune_logs/` · `auto_dash/`  
- 튜너: `scripts/vision_tune/tune_lane_detect.py` (`c`=번들)  
- 자동스윕: `scripts/vision_tune/auto_tune_dash.py`  
- 텔레포트: `./scripts/dev_container.sh teleport <pose>`
