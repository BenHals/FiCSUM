import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ConceptFingerprint",
    version="0.0.2",
    author="Ben",
    author_email="",
    description="real-time lifelong machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/BenHals/ConceptFingerprint",
    packages=['ConceptFingerprint'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
