# Wavelet Scattering Transform for 3D Cosmological Fields

⚠️ *This package is in an early development stage. While it provides useful functionality, some features may be incomplete or subject to change.*

This package provides tools to compute the **Wavelet Scattering Transform (WST)** on 3D fields, with specialized features for galaxy clustering analyses. Built on **[PyTorch](https://pytorch.org/)** for GPU acceleration and differentiability, it optionnally leverages [`nbodykit`](https://nbodykit.readthedocs.io/en/latest/) for processing galaxy catalogs. For applications non-specific to cosmology, the dependency on nbodykit is made optional. See installation instructions below.


## **Installation**

This package requires **Python 3.8 or newer** and **[PyTorch](https://pytorch.org/)** (version 2.0 or later). Dependencies specific to cosmological applications are optional.

### **Install Core Package**:

```bash
pip install .
```

### With Dependencies for Cosmological Applications

This includes a dependency on [`nbodykit`](https://nbodykit.readthedocs.io/en/latest/), which may be tricky to install. Refer to the official [installation instructions](https://nbodykit.readthedocs.io/en/latest/getting-started/install.html) for more details.

```
pip install .[cosmo]
```

## Usage

To get started, check out the [examples/](examples/) folder for detailed use cases.

## Contributing

We welcome contributions to improve this package! Whether you’d like to report a bug, suggest a new feature, or submit a pull request, please start by [opening an issue](https://github.com/bregaldo/galactic_wavelets/issues).

## Related References

1. Régaldo-Saint Blancard, B., *et al.* "Galaxy clustering analysis with SimBIG and the wavelet scattering transform", [*Physical Review D*, 109, 083535](https://doi.org/10.1103/PhysRevD.109.083535) (2024). ArXiv: [2310.15250](https://arxiv.org/abs/2310.15250).

2. Eickenberg, M., *et al.* "Wavelet Moments for Cosmological Parameter Estimation" (2022). ArXiv: [2204.07646](https://arxiv.org/abs/2204.07646).

## License

This project is licensed under the [BSD License](LICENSE).
