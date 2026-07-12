# 갈림길 분기 하나 (공통 진입부 + 분기 센터라인)

# 분기 번호 (0 = 가장 왼쪽, 오른쪽으로 갈수록 증가)
uint16 branch_id
# 분기 신뢰도 0~1
float32 confidence
# 분기 폭 (m)
float32 width
# base_link (x 전방, y 왼쪽, z=0) 센터라인 polyline
geometry_msgs/Point32[] centerline
