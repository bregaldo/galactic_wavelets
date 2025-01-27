import torch

from galactic_wavelets.scattering_operator import ScatteringOp


def test_scattering_op_init():
    grid_size = (32, 32, 32)
    J = 3
    Q = 1
    kc = 3.14
    angular_width = None
    aliasing = True
    use_mp = True
    device = "cpu"
    moments = [1 / 2, 1, 2]
    scattering = False

    scattering_op = ScatteringOp(
        grid_size=grid_size,
        J=J,
        Q=Q,
        kc=kc,
        angular_width=angular_width,
        aliasing=aliasing,
        moments=moments,
        scattering=scattering,
        use_mp=use_mp,
        device=device,
    )

    assert scattering_op.wt_op.grid_size == grid_size
    assert scattering_op.wt_op.J == J
    assert scattering_op.wt_op.Q == Q
    assert scattering_op.wt_op.kc == kc
    assert scattering_op.wt_op.angular_width == angular_width
    assert scattering_op.wt_op.aliasing == aliasing
    assert scattering_op.wt_op.use_mp == use_mp
    assert scattering_op.wt_op.device == torch.device(device)
    assert torch.allclose(scattering_op.moments_op.moments, torch.tensor(moments))
    assert scattering_op.scattering == scattering


def test_scattering_op_zero_input():
    grid_size = (32, 32, 32)
    J, Q = 3, 1
    moments = [1 / 2, 1, 2]

    scattering_op = ScatteringOp(
        grid_size=grid_size,
        J=J,
        Q=Q,
        angular_width=None,
        moments=moments,
        scattering=True,
        device="cpu",
    )

    x = torch.zeros(grid_size)
    s0, s1, s2 = scattering_op(x)
    assert s0.shape == (len(moments),)
    assert s1.shape == (len(moments), J * Q, 1)
    assert s2.shape == (len(moments), J * Q * (J * Q - 1) // 2, 1, 1)
    assert torch.allclose(s0, torch.zeros_like(s0))
    assert torch.allclose(s1, torch.zeros_like(s1))
    assert torch.allclose(s2, torch.zeros_like(s2))
