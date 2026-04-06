from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'custom_navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # ADD THIS LINE:
        # This copies all .pt files from your source directory into the install/share directory
        (os.path.join('share', package_name, 'yolo_models'), glob('custom_navigation/yolo_models/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yash',
    maintainer_email='yashc2025@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'perception_node = custom_navigation.perception_node:main',
            'navigation_brain_node = custom_navigation.navigation_brain_node:main',
            'test_navigation = custom_navigation.test_navigation:main',

        ],
    },
)
