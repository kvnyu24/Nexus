from setuptools import setup, find_packages

setup(
    name="nexus-deep-learning",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.19.0",
        "pynvml>=11.0.0",
        "PyYAML>=5.4.1",
        "tqdm>=4.62.0",
        "pillow>=8.3.0",
        "scikit-learn>=0.24.0",
        "faiss-cpu>=1.7.0"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A modular deep learning library for implementing AI research papers",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nexus",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
)
