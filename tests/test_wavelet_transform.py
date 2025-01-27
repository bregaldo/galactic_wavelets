import torch

from galactic_wavelets.wavelet_transform import WaveletTransform3D


def test_wtransform_init():
    grid_size = (32, 32, 32)
    J = 3
    Q = 1
    kc = 3.14
    angular_width = None
    aliasing = True
    los = (0, 0, 1)
    use_mp = True
    device = "cpu"

    wtransform = WaveletTransform3D(
        grid_size=grid_size,
        J=J,
        Q=Q,
        kc=kc,
        angular_width=angular_width,
        aliasing=aliasing,
        los=los,
        use_mp=use_mp,
        device=device,
    )

    assert wtransform.grid_size == grid_size
    assert wtransform.J == J
    assert wtransform.Q == Q
    assert wtransform.kc == kc
    assert wtransform.angular_width == angular_width
    assert wtransform.aliasing == aliasing
    assert wtransform.los == los
    assert wtransform.use_mp == use_mp
    assert wtransform.device == torch.device(device)


def test_wtransform_zero_input():
    grid_size = (32, 32, 32)
    J, Q = 3, 1

    wtransform = WaveletTransform3D(
        grid_size=grid_size, J=J, Q=Q, angular_width=None, device="cpu"
    )

    wavelets = wtransform.get_wavelets()
    assert wavelets.shape == (J * Q, 1, *grid_size)

    x = torch.zeros(grid_size)
    y = wtransform(x)
    assert y.shape == (J * Q, 1, *grid_size)
    assert torch.allclose(y, torch.zeros_like(y))
