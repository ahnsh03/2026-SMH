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
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py'),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='SEA-ME Team',
    maintainer_email='ahnsh03@inha.edu',
    description='Autonomous driving inference node for SEA:ME hackathon',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'inference_node = inference.inference_node:main',
            'lane_control_node = inference.lane_control_node:main',
            'lane_preview_node = inference.lane_preview_node:main',
            'hsv_mask_web_node = inference.hsv_mask_web_node:main',
        ],
    },
)
