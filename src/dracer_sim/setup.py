import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'dracer_sim'


def collect_data_files(source_subdir: str, install_subdir: str):
  files = []
  base = os.path.join(os.path.dirname(__file__), source_subdir)
  if not os.path.isdir(base):
    return files
  for root, _dirs, filenames in os.walk(base):
    if not filenames:
      continue
    rel = os.path.relpath(root, base)
    dest = (
      os.path.join('share', package_name, install_subdir, rel)
      if rel != '.'
      else os.path.join('share', package_name, install_subdir)
    )
    files.append((dest, [os.path.join(root, name) for name in filenames]))
  return files


setup(
  name=package_name,
  version='0.1.0',
  packages=find_packages(exclude=['test']),
  data_files=[
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
  ]
  + collect_data_files('launch', 'launch')
  + collect_data_files('urdf', 'urdf')
  + collect_data_files('worlds', 'worlds')
  + collect_data_files('models', 'models')
  + collect_data_files('config', 'config')
  + collect_data_files('assets', 'assets'),
  install_requires=['setuptools'],
  zip_safe=True,
  maintainer='SEA-ME Team',
  maintainer_email='ahnsh03@inha.edu',
  description='Gazebo simulator for D-Racer',
  license='Apache-2.0',
  tests_require=['pytest'],
  entry_points={
    'console_scripts': [
      'sim_control_bridge = dracer_sim.control_bridge:main',
      'sim_camera_republish = dracer_sim.camera_republish:main',
      'sim_battery_stub = dracer_sim.battery_stub:main',
      'sim_joystick_bridge = dracer_sim.joystick_bridge:main',
      'sim_robot_description_publisher = dracer_sim.robot_description_publisher:main',
      'sim_camera_preview = dracer_sim.camera_preview:main',
    ],
  },
)
