from pathlib import Path

from setuptools import find_packages, setup

package_name = 'sensor_engineering'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (str(Path('share') / package_name / 'launch'), [str(p) for p in Path('launch').glob('*.py')]),
        (str(Path('share') / package_name / 'worlds'), [str(p) for p in Path('worlds').iterdir() if p.is_file()]),
        (str(Path('share') / package_name / 'config'), [str(p) for p in Path('config').iterdir() if p.is_file()]),
    ],
    install_requires=['setuptools'],
    entry_points={
        'console_scripts': [
            'ground_truth_node = sensor_engineering.ground_truth_node:main',
            'fusion_node = sensor_engineering.fusion_node:main',
            'path_follower = sensor_engineering.path_follower:main',
            'fusion_dashboard = sensor_engineering.dashboard:main',
        ],
    },
)
