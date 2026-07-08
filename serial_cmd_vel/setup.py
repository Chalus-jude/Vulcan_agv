from setuptools import find_packages, setup

package_name = 'serial_cmd_vel'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    
    data_files=[
        # Package index
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),

        # Package.xml
        ('share/' + package_name, ['package.xml']),

        # ✅ Launch files (FIXED)
        ('share/' + package_name + '/launch', [
            'launch/cartographer_s3.launch.py',
            'launch/nav2.launch.py',
            'launch/rplidar_s3.lua',   # ✅ IMPORTANT FIX
        ]),

        # ✅ Config files
        ('share/' + package_name + '/config', [
            'config/ekf.yaml',
            'config/nav2_params.yaml',
            'config/map.yaml',
            'config/map.pgm'
        ]),
    ],

    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nidar',
    maintainer_email='nidar@todo.todo',
    description='AMR serial + navigation package',
    license='TODO: License declaration',
    tests_require=['pytest'],

    entry_points={
        'console_scripts': [
            'serial_sub = serial_cmd_vel.serial_subscriber:main',
            'cmd_pub = serial_cmd_vel.cmd_publisher:main',
        ],
    },
)