# 인지 → 판단제어 단일 토픽. 프레임 단위 차선 인지 결과 전체.

std_msgs/Header header

# 색상별 차선 마킹 후보들
LaneMarking[] lanes

# 색상/좌우 가시성 및 신뢰도
bool white_visible
bool yellow_visible
bool left_visible
bool right_visible
float32 white_confidence
float32 yellow_confidence
float32 left_confidence
float32 right_confidence

# 흰/노란 차선 센터라인 (좌우 경계 중점, base_link Point32 polyline)
geometry_msgs/Point32[] white_centerline
geometry_msgs/Point32[] yellow_centerline

# 노란 가로 실선(정지선/원형교차로 진입선 등) 등장 여부
bool yellow_crossing_line

# 갈림길: 발생 여부와 각 분기 경로(번호+센터라인)
# fork_active=false 이면 branches 는 단일 주행 경로 1개.
bool fork_active
RoadBranch[] branches

# 주행가능영역: BEV mono8 그리드(0/255)
sensor_msgs/Image drivable_area
# 그리드 -> base_link 변환 스케일
#   x = x_forward_max - row * meters_per_pixel
#   y = ((width - 1) / 2 - col) * meters_per_pixel
float32 meters_per_pixel
float32 x_forward_max
