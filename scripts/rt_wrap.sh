#!/bin/bash
# [비활성] 원래 이 래퍼는 control/battery를 SCHED_RR(실시간)로 올려 i2c 타이머를
# CPU 스타베이션에서 보호하려 했다. 그러나 이 보드(tcc, 하드웨어 워치독)에서는
# userspace를 SCHED_RR로 올리면 커널/드라이버 스레드나 워치독 갱신이 굶어
# 보드가 리셋(SSH 끊김)되는 문제가 확인되었다(2026-07-15). 따라서 RT 승격을
# 비활성화하고 일반 우선순위로 그대로 실행한다.
#
# CPU 스타베이션 완화는 RT 대신 "부하 줄이기"로 해결할 것:
#   - 주행 중 lane_view/monitor 끄기
#   - perception(특히 CPU에서 도는 sign onnxruntime) 경량화
# 그리고 wedge가 나도 control_node는 크래시하지 않도록 이미 방어코드가 들어가 있다.
exec "$@"
