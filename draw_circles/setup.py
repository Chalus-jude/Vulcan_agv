from setuptools import find_packages, setup

package_name = 'draw_circles'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='arunkumar',
    maintainer_email='marunkumar5767@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'draw_circle_node = draw_circles.draw_circle_node:main',
            'qr_scanner = draw_circles.qr_scanner_node:main',
        ],
    },
)
