# Wavelet Scattering Transform for Galaxy Clustering Analysis

This package enables the computation of the Wavelet Scattering Tranform (WST) on Galaxy Catalogs as described in Regaldo-Saint Blancard+in prep.
Check out the examples/ folder for use cases.

This package is not GPU-accelerated yet, but might be in the future. In the meantime, for GPU-acceleration, we recommend using [Kymatio](https://github.com/kymatio/kymatio) which implements a 3D version of the WST.

## Install

```bash
conda install -c bccp nbodykit
pip install -r requirements.txt
pip install .
```
