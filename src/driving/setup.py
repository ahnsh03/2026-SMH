import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'driving'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # 워크스페이스 루트 config/ 를 패키지 share 로 설치 (lane_control.yaml 등)
        (os.path.join('share', package_name, 'config'), glob('../../config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='SMH Team',
    maintainer_email='inha.aim.01@gmail.com',
    description='제어 계층: LaneDetections → 경로선택/추종 → Control 명령 발행',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'control_node = driving.control_node:main',
            # 통합 주행노드(인지+제어 in-process, 인지→제어 토픽 미사용)
            'lane_drive_node = driving.lane_drive_node:main',
        ],
    },
)
