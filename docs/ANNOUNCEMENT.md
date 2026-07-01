# 팀원 공지용 (카카오톡 복사)

---

[SEA:ME 해커톤] 팀 GitHub 개설 안내

안녕하세요, 팀 GitHub 저장소를 구축했습니다.

🔗 https://github.com/ahnsh03/SEA-Me-Hackathon

【저장소 내용】
- 대회 정보·규정·주최측 문답 (docs/competition.md)
- 역할 분담 및 담당 파일 (docs/roles.md)
- D3-G 개발 환경 셋업 (docs/setup.md)
- inference ROS2 패키지 골격 (src/inference/)

【역할 분담】
- 차선 인지: 장원태 → modules/lane_detection.py
- 신호등·표지판: 장원정 → modules/traffic_sign.py
- ArUco 마커: 안승현, 박성준 → modules/aruco_detection.py
- 회전 교차로: 양서준 → modules/roundabout.py

【D3-G 보드 셋업】
git clone https://github.com/ahnsh03/SEA-Me-Hackathon.git
cd SEA-Me-Hackathon
./scripts/init_workspace.sh
colcon build --symlink-install

【브랜치】
feature/이름-기능 으로 작업 후 PR

【Notion】
https://app.notion.com/p/55e1b0cdce9b8292a19d81c5b1605983

질문은 카톡 또는 월요일 15시 정기 회의에!

---
