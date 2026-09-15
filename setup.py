import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'laundry_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'meshes'), glob('meshes/*')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cde3301a',
    maintainer_email='cde3301a@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'run_probe = laundry_control.main:main',
            'tof_sensor = laundry_control.tof_sensor:main',
            'scan_replay = laundry_control.scan_replay:main',
            'scan_recorder_node = laundry_control.scan_recorder_node:main',
            'gripper_node = laundry_control.gripper_node:main',
            'laundry_detect = laundry_control.laundry_detect_cli:main',
            'promote_baseline = laundry_control.promote_baseline:main',
            'retrieve = laundry_control.retrieve:main',
        ],
    },
)
