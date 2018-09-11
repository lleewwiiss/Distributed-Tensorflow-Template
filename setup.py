from setuptools import find_packages
from setuptools import setup

# add any packages required for your project e.g.
REQUIRED_PACKAGES = ["tensorflow>=1.8"]

setup(
    name="example model",
    version="0.1",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description="Example trainer package",
)
