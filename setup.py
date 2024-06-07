
import os

from setuptools import find_packages, setup

with open(os.path.join("skill_aware_panda_gym", "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="skill_aware_panda_gym",
    description="skill_aware_panda_gym",
    author="Loxs",
    author_email="1043694812@qq.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    package_data={"skill_aware_panda_gym": ["version.txt"]},
    version=__version__,
    install_requires=["gymnasium>=0.26", "pybullet", "numpy", "scipy","panda-gym"],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)