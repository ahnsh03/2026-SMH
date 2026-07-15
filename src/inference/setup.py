import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'inference'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # board 처럼 워크스페이스 루트 config/ 를 패키지 share 로 설치
        (os.path.join('share', package_name, 'config'), glob('../../config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='SMH Team',
    maintainer_email='inha.aim.01@gmail.com',
    description='인지 계층: 카메라 영상 → 차선/표지 인지 → LaneDetections 발행',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'inference_node = inference.inference_node:main',
        ],
    },
)
