from setuptools import setup, find_packages

setup(
    name="hts-lib",
    version="1.0.0",
    description="Dynamic Hypernetwork with Dual-Residual Basis & Multi-Domain Continual Learning",
    author="Google DeepMind pair programming with User",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.20.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
