# 차선 마킹 하나 (인지 결과 후보)

uint8 COLOR_UNKNOWN=0
uint8 COLOR_WHITE=1
uint8 COLOR_YELLOW=2

uint8 SIDE_UNKNOWN=0
uint8 SIDE_LEFT=1
uint8 SIDE_RIGHT=2
uint8 SIDE_CENTER=3

# 한 프레임 안에서 후보 구분용 번호
uint16 id
# 흰색/노란색
uint8 color
# 차량 기준 왼쪽/오른쪽/중앙 추정
uint8 side_hint
# 검출 신뢰도 0~1
float32 confidence
# BEV상 검출된 차선 길이 (m)
float32 length
# 차선 진행 방향 (rad)
float32 heading
# 차선 곡률
float32 curvature
# base_link (x 전방, y 왼쪽, z=0) polyline
geometry_msgs/Point32[] points
