# setup.py
from pathlib import Path

from setuptools import find_packages, setup


def read_requirements(path="requirements.txt"):
    lines = Path(path).read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


setup(
    name="soccer_one_twos_extractor",
    version="0.1.0",
    description="Progressive one–twos extractor from Opta F24 game data",
    packages=find_packages(exclude=("notebooks*", "tests*", "docs*")),
    install_requires=read_requirements(),  # ← loaded here
    python_requires=">=3.9",
    include_package_data=True,
)
