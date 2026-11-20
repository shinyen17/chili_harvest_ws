import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'chili_harvest'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'models'), glob('models/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Lee Shin Yen',
    maintainer_email='todo@todo.todo',
    description='Chili harvesting on the cocoabot arm',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'chili_perception = chili_harvest.chili_perception:main',
            'harvest_control = chili_harvest.harvest_control:main',
        ],
    },
)
