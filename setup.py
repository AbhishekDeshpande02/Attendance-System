from setuptools import setup, find_packages

setup(
    name='attendance_model',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',
        'tensorflow',  # or any other dependencies your model needs
    ],
)
