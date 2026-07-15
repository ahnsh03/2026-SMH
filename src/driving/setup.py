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
        ],
    },
)
