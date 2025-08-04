from setuptools import setup, find_packages

setup(
    name="multimodal_project",
    version="0.1",
    packages=find_packages(include=["src", "src.*", "models", "models.*"]),
)
