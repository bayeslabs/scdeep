import setuptools

VERSION = "0.0.1"

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="scdeep",
    version=VERSION,
    author="BayesLabs",
    author_email="contact@bayeslabs.co",
    description="single-cell analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bayeslabs/scdeep",
    packages=setuptools.find_packages(),
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'scanpy',
    ],
    python_requires='>=3.6',
)
