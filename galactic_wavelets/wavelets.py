import multiprocessing as mp
import os
from functools import partial

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def check_grid(grid_or_shape, to_torch=False, axes_weights=None):
    """Check or build a grid of given shape.
    A grid is an array of shape (n, k1, ..., kn) where n is the dimension of the grid,
    and k1, ..., kn is the size of the grid for every dimension.
    It stores the coordinates of the grid points.
    By default, these go from -ki//2 to -ki//2 + ki for every dimension.

    Inspired by quijote_scattering/filter_bank.py

    Args:
        grid_or_shape (tuple or array): Tuple describing the required shape or array corresponding to the grid.
        to_torch (bool, optional): Shall we build a torch tensor? Defaults to False.
        axes_weights (list or 1d array, optional): Weights of the coordinates for each dimension. Defaults to None (no weighting).

    Returns:
        array: Grid.
    """
    if isinstance(grid_or_shape, tuple):
        shape = grid_or_shape
        slices = [slice(-(s // 2), -(s // 2) + s) for s in shape]
        grid = np.mgrid[slices].astype(np.float32)

        # Axis weighting
        if axes_weights is not None:
            assert len(shape) == len(axes_weights)
            for i in range(len(shape)):
                grid[i] *= axes_weights[i]  # Physical space
    else:
        grid = grid_or_shape
        assert (
            grid.shape[0] == len(grid.shape) - 1
        )  # first dimension of grid must be ndim

    if isinstance(grid, np.ndarray) and to_torch:
        grid = torch.from_numpy(grid.astype("float32"))
    return grid


def gaussian(omega):
    """Gaussian function.

    Args:
        omega (float or array): Function argument.

    Returns:
        float or array: Bump values.
    """
    g = np.exp(-(omega**2))
    return g


def lanusse_scaling_function_3d(r, kc=np.pi):
    """Lanusse isotropic scaling function in physical space (see Lanusse+2012).
    This function is zero in (3D) Fourier space beyond kc.

    Args:
        r (array): Grid of r coordinates.
        kc (float, optional): Cutoff frequency. Defaults to np.pi.

    Returns:
        array: Values on the grid.
    """
    res = np.zeros_like(r)
    c = 1 / (8 * np.sqrt(2) * np.pi)
    mask = r != 0.0
    r_masked = r[mask]
    res[mask] = (
        c
        / r_masked
        * (
            -768
            * (
                kc * r_masked * np.cos(kc * r_masked / 4)
                - 4 * np.sin(kc * r_masked / 4)
            )
            * np.sin(kc * r_masked / 4) ** 3
            / (kc**3 * r_masked**5)
        )
    )
    res[np.logical_not(mask)] = c * kc**3 / 4
    return res


def lanusse_mother_wavelet_3d(r, kc=np.pi):
    """Lanusse isotropic wavelet (see Lanusse+2012).
    This wavelet is zero in (3D) Fourier space beyond 2*kc.

    Args:
        r (array): Grid of r coordinates.
        kc (float, optional): Cutoff frequency. Defaults to np.pi.

    Returns:
        array: Values on the grid.
    """
    return lanusse_scaling_function_3d(r, kc=2 * kc) - lanusse_scaling_function_3d(
        r, kc=kc
    )


def lanusse_fwavelet_3d(
    kc: float,
    grid_or_shape,
    fourier=True,
    axes_weights=None,
    aliasing=True,
    return_grid=False,
):
    """Wrapper to build a Lanusse isotropic wavelet (see Lanusse+2012) from a given grid of a given shape.
    Default behavior is to return the wavelet in Fourier space.
    Also build aliased wavelets to regularize the signal in physical space when kc is greater than pi/2.

    Args:
        kc (float): Cutoff frequency.
        grid_or_shape (tuple or array): Tuple describing the required shape or array corresponding to the grid.
        fourier (bool, optional): Shall we the wavelet in Fourier space? Defaults to True.
        axes_weights (list or 1d array, optional): Weights of the coordinates for each dimension. Defaults to None (no weighting).
        aliasing (bool, optional): Shall we build aliased wavelets when kc > pi/2? Defaults to True.
        return_grid (bool, optional): Shall we also return the grid? Defaults to False.

    Returns:
        array or tuple of arrays: Wavelet, pr tuple containing the wavelet and the grid if required.
    """
    grid = check_grid(grid_or_shape, axes_weights=axes_weights)

    # If aliasing is True might need to extend the grid
    if kc > np.pi / 2 and aliasing:  # Lanusse wavelet function is zero beyond 2*kc
        shape = grid.shape[1:]
        extend_factor = int(np.ceil(kc / (np.pi / 2)))
        grid = check_grid(
            tuple([extend_factor * i for i in shape]), axes_weights=axes_weights
        )
        kc_eff = kc / extend_factor
    else:
        kc_eff = kc

    # Checks
    ndim = grid.shape[0]
    shape = grid.shape[1:]
    if ndim != 3:
        print("Warning the Lanusse wavelet is designed for 3D data!")
    radii = np.linalg.norm(grid, axis=0)

    # Computation of the wavelet in physical space
    ws = np.fft.ifftshift(lanusse_mother_wavelet_3d(radii, kc=kc_eff))

    # Return it
    if fourier:
        ws = np.fft.fftn(ws, axes=(-1, -2, -3)).real.astype(np.float32)
    if return_grid:
        return ws, grid
    else:
        return ws


def angular_gaussians_bank(grid, orientations, angular_width=np.pi / 4):
    """Build a bank of angular gaussian functions in 3D Fourier space for an array of given orientations.

    Args:
        grid (array): Grid of coordinates.
        orientations (array): Orientation vectors. Array should be of dimension 4, where the last 3 dimensions correpond to the vector coordinates.
        angular_width (float, optional): Angular width of the gaussian function. Defaults to np.pi/4.

    Inspired by quijote_scattering/filter_bank.py

    Returns:
        array: Bank of gaussian functions.
    """
    shape = grid.shape[1:]
    n_orientations = orientations.shape[0]
    assert len(shape) == 3

    radii = np.linalg.norm(grid, axis=0)

    angular_gaussians = np.zeros((n_orientations,) + shape, dtype=np.float32)
    z_axis = np.array([0.0, 0.0, 1.0])
    for i, orientation in enumerate(orientations):
        orientation = orientation / np.linalg.norm(orientation)
        grid_tmp = grid

        if np.dot(orientation, z_axis) < 1 - 1e-5:
            # if orientation is not z-axis, rotate grid
            aux_axis = np.cross(orientation, z_axis)
            aux_axis = aux_axis / np.linalg.norm(aux_axis)
            aux_z = np.cross(orientation, aux_axis)
            basis = np.stack((orientation, aux_z, aux_axis), axis=1)
            angle = np.arccos(np.dot(orientation, z_axis))
            rot3db = np.array(
                [
                    [np.cos(-angle), -np.sin(-angle), 0],
                    [np.sin(-angle), np.cos(-angle), 0],
                    [0.0, 0.0, 1.0],
                ]
            )
            rot3d = basis @ rot3db @ basis.T
            assert np.linalg.norm(rot3d @ orientation - z_axis) < 1e-5

            grid_tmp = (grid.T @ rot3d.T).T

        _, _, z = grid_tmp

        elevation = np.arccos(
            np.clip(z / (radii + 1e-5), -1.0, 1.0)
        )  # Somehow, in specific circumstances, z / (radii + 1e-5) can be outside [-1, 1]

        angular_gaussians[i] = gaussian(elevation / angular_width)

    return np.fft.ifftshift(angular_gaussians, axes=(-1, -2, -3))


def lanusse_fwavelet_3d_bank_para(kcs, grid_or_shape, **kwargs):
    """Internal function, for parallelization purposes."""
    return lanusse_fwavelet_3d_bank(grid_or_shape, kc=kcs, **kwargs)


def lanusse_fwavelet_3d_bank(
    grid_or_shape,
    J=None,
    Q=1,
    kc=np.pi,
    orientations=None,
    angular_width=np.pi / 4,
    axes_weights=None,
    aliasing=True,
    use_mp=True,
):
    """Build a bank of oriented Lanusse wavelets and return them in Fourier space.
    These are built by multiplying in Fourier space a Lanusse isotropic wavelet (Lanusse+2012) and an angular gaussian function.

    There are two ways to call this function: either by providing a scalar kc, or by providing an array of kc values.
    In the first case, J and Q are used to define an array of kc values based on scalar kc, in the other one, these parameters are just ignored.
    This is notably of interest for parallel computation (when use_mp is True).

    Args:
        grid_or_shape (tuple or array): Tuple describing the required shape or array corresponding to the grid.
        J (int, optional): Number of j values (i.e. octaves). Defaults to None.
        Q (int, optional): Number of q values (i.e. scales per octave). Defaults to 1.
        kc (float or array, optional): Cutoff frequencies. Defaults to np.pi.
        orientations (array): Orientation vectors. Array should be of dimension 4, where the last 3 dimensions correpond to the vector coordinates. Defaults to None, meaning isotropic wavelets are built.
        angular_width (float, optional): Angular width of the gaussian function. Defaults to np.pi/4.
        axes_weights (list or 1d array, optional): Weights of the coordinates for each dimension. Defaults to None (no weighting).
        aliasing (bool, optional): Shall we build aliased wavelets when kc > pi/2? Defaults to True.
        use_mp (bool, optional): Shall we parallelize the computation of the bank of wavelets? Defaults to True.

    Returns:
        array: Bank of wavelets.
    """
    # Initial grid
    grid = check_grid(grid_or_shape, axes_weights=axes_weights)
    shape = grid.shape[1:]

    # Define kc values if not defined yet
    if J is None:
        J = int(np.log2(min(shape)) - 2)
    scalings = 2 ** -(np.arange(0, J * Q) / Q)
    if np.isscalar(kc):
        kcs = kc * scalings
    else:
        kcs = kc

    # Orientations check
    oriented = (
        orientations is not None and angular_width is not None
    )  # Are we building oriented wavelets or not?
    if oriented:
        assert orientations.ndim == 2
        assert orientations.shape[1] == 3
        n_orientations = orientations.shape[0]

    # Build array for the Fourier transform of the wavelets (float32 because real values are expected in Fourier space)
    if oriented:
        fwavelets = np.zeros((len(kcs), n_orientations) + shape, dtype=np.float32)
        # Axis weighting should modify orientations
        orientations_rescaled = np.copy(orientations)
        if axes_weights is not None:
            assert len(shape) == len(axes_weights)
            for i in range(len(shape)):
                orientations_rescaled[:, i] /= axes_weights[i]  # Fourier space
    else:
        fwavelets = np.zeros((len(kcs),) + shape, dtype=np.float32)

    # Build  wavelets
    if use_mp:
        build_bp_para_loc = partial(
            lanusse_fwavelet_3d_bank_para,
            grid_or_shape=grid,
            orientations=orientations,
            angular_width=angular_width,
            axes_weights=axes_weights,
            aliasing=aliasing,
            use_mp=False,
        )
        nb_processes = min(os.cpu_count(), len(kcs))
        work_list = np.array_split(kcs, nb_processes)
        pool = mp.Pool(processes=nb_processes)
        results = pool.map(build_bp_para_loc, work_list)
        cnt = 0
        for i in range(len(results)):
            fwavelets[cnt : cnt + results[i].shape[0]] = results[i]
            cnt += results[i].shape[0]
        pool.close()
    else:
        for i, kc in enumerate(kcs):
            # Get wavelet and grid
            fw, g = lanusse_fwavelet_3d(
                kc, grid, axes_weights=axes_weights, aliasing=aliasing, return_grid=True
            )

            if oriented:
                # Axis weighting for Fourier space
                if (
                    axes_weights is not None
                ):  # Need to compensate previous weighting in physical space
                    for j in range(len(shape)):
                        g[j] /= axes_weights[j] ** 2
                for j in range(len(shape)):  # To get frequencies
                    g[j] /= shape[j]
                angular_gaussians = angular_gaussians_bank(
                    g, orientations_rescaled, angular_width=angular_width
                )
                # Cancel previous weighting (in case we reuse the same grid later)
                for j in range(len(shape)):
                    g[j] *= shape[j]
                if axes_weights is not None:
                    for j in range(len(shape)):
                        g[j] *= axes_weights[j] ** 2

                fw = fw * angular_gaussians

            if aliasing and kc > np.pi / 2:
                extend_factor = int(np.ceil(kc / (np.pi / 2)))
                fw = fw.reshape(
                    fw.shape[:-3]
                    + (
                        extend_factor,
                        shape[0],
                        extend_factor,
                        shape[1],
                        extend_factor,
                        shape[2],
                    )
                ).sum(axis=(-6, -4, -2))

            fwavelets[i] = fw

    return fwavelets


def get_dodecahedron_vertices(half=False, los=None):
    """Return the vertices of a regular dodecahedron.

    Args:
        half (bool, optional): Shall we return the vertices of the half-space only? Defaults to False.
        los (array, optional): Line of sight vector. Defaults to None, corresponding to [0, 0, 1].

    Returns:
        array: Array of vertices.
    """
    phi = (1 + np.sqrt(5)) / 2

    # Taken from wikipedia
    vertices = []
    for v in [[1, 1, 1], [0, phi, 1 / phi], [1 / phi, 0, phi], [phi, 1 / phi, 0]]:
        vertices.append([v[0], v[1], v[2]])
        vertices.append([v[0], v[1], -v[2]])
        vertices.append([v[0], -v[1], v[2]])
        vertices.append([v[0], -v[1], -v[2]])
        vertices.append([-v[0], v[1], v[2]])
        vertices.append([-v[0], v[1], -v[2]])
        vertices.append([-v[0], -v[1], v[2]])
        vertices.append([-v[0], -v[1], -v[2]])
    vertices = np.unique(np.array(vertices), axis=0)

    # Rotation to make one vertex coincides with [0, 0, 1] (index for that vertex is 4)
    theta = np.arctan(1 / phi**2)
    r = R.from_euler("y", theta)
    vertices = r.apply(vertices) / np.sqrt(3)  # sqrt(3) to have unit norm

    if los is not None and np.dot(los, [0, 0, 1]) < 1 - 1e-5:
        losv = los / np.linalg.norm(los)
        zv = np.array([0, 0, 1])
        v = np.cross(zv, losv)

        s = np.linalg.norm(v)
        c = np.dot(zv, losv)

        mat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

        rot = R.from_matrix(np.identity(3) + mat + mat @ mat * ((1 - c) / s**2))

        vertices = rot.apply(vertices)

    # Vertices for half space only?
    if half:
        return vertices[
            np.dot(vertices, los if los is not None else np.array([0, 0, 1])) >= 0.0
        ]
    else:
        return vertices


def get_icosphere2_vertices(half=False, los=None):
    """Return the vertices of an geodesic icosahedron with subdivision frequency 2 (cf Python package icosphere).
       Line of sight vertex has index 12.

    Args:
        half (bool, optional): Shall we return the vertices of the half-space only? Defaults to False.
        los (array, optional): Line of sight vector. Defaults to None, corresponding to [0, 0, 1].

    Returns:
        array: Array of vertices.
    """
    vertices = np.array(
        [
            [0.0, 0.52573111, 0.85065081],
            [0.0, -0.52573111, 0.85065081],
            [0.52573111, 0.85065081, 0.0],
            [-0.52573111, 0.85065081, 0.0],
            [0.85065081, 0.0, 0.52573111],
            [-0.85065081, 0.0, 0.52573111],
            [-0.0, -0.52573111, -0.85065081],
            [-0.0, 0.52573111, -0.85065081],
            [-0.52573111, -0.85065081, -0.0],
            [0.52573111, -0.85065081, -0.0],
            [-0.85065081, -0.0, -0.52573111],
            [0.85065081, -0.0, -0.52573111],
            [0.0, 0.0, 1.0],
            [0.30901699, 0.80901699, 0.5],
            [-0.30901699, 0.80901699, 0.5],
            [0.5, 0.30901699, 0.80901699],
            [-0.5, 0.30901699, 0.80901699],
            [0.5, -0.30901699, 0.80901699],
            [-0.5, -0.30901699, 0.80901699],
            [-0.30901699, -0.80901699, 0.5],
            [0.30901699, -0.80901699, 0.5],
            [0.0, 1.0, 0.0],
            [0.80901699, 0.5, 0.30901699],
            [0.30901699, 0.80901699, -0.5],
            [0.80901699, 0.5, -0.30901699],
            [-0.80901699, 0.5, 0.30901699],
            [-0.30901699, 0.80901699, -0.5],
            [-0.80901699, 0.5, -0.30901699],
            [0.80901699, -0.5, 0.30901699],
            [1.0, 0.0, 0.0],
            [-0.80901699, -0.5, 0.30901699],
            [-1.0, 0.0, 0.0],
            [-0.0, 0.0, -1.0],
            [-0.30901699, -0.80901699, -0.5],
            [0.30901699, -0.80901699, -0.5],
            [-0.5, -0.30901699, -0.80901699],
            [0.5, -0.30901699, -0.80901699],
            [-0.5, 0.30901699, -0.80901699],
            [0.5, 0.30901699, -0.80901699],
            [0.0, -1.0, -0.0],
            [-0.80901699, -0.5, -0.30901699],
            [0.80901699, -0.5, -0.30901699],
        ]
    )

    if los is not None and np.dot(los, [0, 0, 1]) < 1 - 1e-5:
        losv = los / np.linalg.norm(los)
        zv = np.array([0, 0, 1])
        v = np.cross(zv, losv)

        s = np.linalg.norm(v)
        c = np.dot(zv, losv)

        mat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

        rot = R.from_matrix(np.identity(3) + mat + mat @ mat * ((1 - c) / s**2))

        vertices = rot.apply(vertices)

    # Vertices for half space only?
    if half:
        return vertices[
            np.dot(vertices, los if los is not None else np.array([0, 0, 1])) >= 0.0
        ]
    else:
        return vertices
