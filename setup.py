import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="galactic_wavelets",
    version="1.0",
    author="Bruno Régaldo-Saint Blancard",
    author_email="bregaldosaintblancard@flatironinstitute.org",
    description="Wavelet Scattering Transform for Galaxy Clustering Analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bregaldo/galactic_wavelets",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
