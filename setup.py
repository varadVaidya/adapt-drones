from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="adapt_drones",
    version="0.1.0",
    description="",
    author="Varad Vaidya <vaidyavarad2001@gmail.com>",
    packages=find_packages(),
    install_requires=requirements,
)
