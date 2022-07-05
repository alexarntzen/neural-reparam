from setuptools import setup

setup(
    name="neural-reparam",
    version="0.1",
    packages=["neural_reparam"],
    url="",
    license="",
    author="Alexander Johan Arntzen",
    author_email="alexander@alexarntzen.com",
    description="For master thesis project",
    install_requires=[
        "torch~=1.12.0",
        "numpy>=1.18.2",
        "matplotlib>=3.2.0",
        "scikit-learn>=0.24.1",
        "tqdm>=4.63.0",
        "pandas>=1.2.4",
    ],
)
