from setuptools import setup, find_packages

setup(
    name="dag",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],
    description="dag is a Python package for Direction Acyclic Graph Modulization.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    author="Ludwig",
    author_email="yuzeliu@gmail.com",
    url=None,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)