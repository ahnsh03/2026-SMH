# 시뮬레이터 도입 메모 (LIMO)

> 마지막 업데이트: 2026-07-08  
> PC 상위 상세 평가: [`../../docs/sim/limo-simulator-assessment.md`](../../docs/sim/limo-simulator-assessment.md)  
> 카메라 스펙: [hardware-camera.md](./hardware-camera.md)

시운전 공간·공식 D-Racer 시뮬 부재 → AgileX **LIMO** Gazebo를 빌려 **인지 개발 샌드박스**로 쓰는 방안을 검토 중.

## 한 줄 결론

| 레포 | 판정 |
|------|------|
| `limo_sim_code_v2` | ROS1 — 작년 알고리즘 참고용 |
| `ugv_gazebo_sim` | ROS1 Melodic 계열 Gazebo — 팀 Humble과 불일치 |
| **`limo_ros2` (`limo_car`)** | **Humble — 주 후보.** 카메라를 C920e(hFoV 70.42°, 가능하면 320×160 JPEG `/camera/image/compressed`)로 맞춰 사용 |

클론은 PC 상위 `external/`에만 둠 (보드 `2026-SMH` 단독 clone에는 없음).

## 반드시 맞출 것

1. C920e FOV (기본 LIMO depth ~80°와 다름)  
2. D-Racer 토픽·해상도  
3. 차체 역학·LiDAR는 D-Racer와 다름 → 인지 우선, 제어는 실차 검증
