# 팀원 공지용 (카카오톡 복사)

---

[SEA:ME 해커톤] 팀 GitHub + 협업 구조 업데이트

🔗 https://github.com/ahnsh03/2026-SMH

【변경 사항】
- MainPlanner 통합 (/control = inference_node 하나) · mask_p + 표지 게이트 Out 갈림
- 보드: ./scripts/board_sync.sh · PC 시뮬: sim-bringup + sim-auto
- 협업 가이드: docs/collaboration.md ★필독

【담당 파일】
- 차선: 안승현(임시)/장원태 → modules/lane_detection.py
- 신호등·표지판: 장원정 → modules/traffic_sign.py
- ArUco 검출: 안승현 → modules/aruco/detector.py
- ArUco 정지: 박성준 → modules/aruco/stop_logic.py
- MainPlanner·In/Out: 양서준 → pipeline.py + config/main_planner.yaml

【D3-G 보드 — 최초】
git clone https://github.com/ahnsh03/2026-SMH.git
cd 2026-SMH
chmod +x scripts/*.sh
./scripts/board_sync.sh --no-pull

【D3-G 보드 — 코드 받을 때】
./scripts/board_sync.sh
source install/setup.bash
ros2 launch inference auto_driving.launch.py
# 코스: route_mode:=in|out · 디버그: /debug/planner

【PC 시뮬】
./scripts/dev_container.sh sim-bringup
./scripts/dev_container.sh sim-auto route_mode:=out viz:=lane
→ docs/simulation-setup.md

【개발 — Git 규약 ★필독】
1. main 직접 push 금지 (팀장 예외·긴급 문서 동기화만)
2. feature/이름-기능 브랜치 생성
3. 담당 파일 위주 수정 → commit → push
4. Pull Request → 팀장 merge
5. 보드: ./scripts/board_sync.sh

→ docs/collaboration.md · docs/roles.md

【Notion】
https://app.notion.com/p/55e1b0cdce9b8292a19d81c5b1605983

질문은 카톡 또는 월요일 15시 정기 회의에!

---
