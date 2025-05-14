"""
Setup script for the SOWLv2 package.

SOWLv2 combines OWLv2 and SAM 2 for text-prompted object segmentation.
"""
from setuptools import setup, find_packages 

setup(
    name="sowlv2",
    version="1.0.0",
    description="SOWLv2: Text-prompted object segmentation using OWLv2 and SAM 2",
    author="Bolyos Csaba",
    author_email="bladeszasza@gmail.com",
    license="Apache-2.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.13.0",
        "transformers>=4.32.1",
        "sam2>=1.1.0",
        "opencv-python>=4.5.5.64",
        "numpy>=1.21.0",
        "Pillow>=9.0.0",
        "pyyaml>=6.0",
        "huggingface_hub>=0.15.0"
    ],
    entry_points={
        "console_scripts": [
            "sowlv2-detect = sowlv2.cli:main"
        ]
    }
)
