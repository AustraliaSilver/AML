from setuptools import setup, find_packages

setup(
    name="aml-framework",
    version="2.0.0",
    description="Architecture Mode Learning (AML) Framework: Meta-Generators (MGN) & Continual Hypernetworks (HtS)",
    author="AustraliaSilver",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        # Only needed by experiments/benchmark_baselines.py
        "baselines": ["scikit-learn>=1.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
