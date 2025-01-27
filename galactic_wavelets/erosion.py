import numpy as np
import torch
import torch.nn.functional as F


def mask_erosion(mask, kernel):
    """Mask erosion.

    Parameters
    ----------
    mask : _type_
        Initial geometry mask. True when pixels are to be removed.
    kernel : _type_
        Kernel used to extend input mask (in wraparound order).
    """
    mask = mask.astype(bool)
    kernel = kernel.astype(bool)
    kernel_support = np.argwhere(kernel)  # List of nonzeros pixels in the kernel

    if mask.ndim != kernel.ndim:
        raise Exception("mask and shape must have the same ndim!")
    for i in range(kernel.ndim):
        if mask.shape[i] < kernel.shape[i]:
            raise Exception("Invalid kernel shape!")

    mask_padding_shape = []
    kernel_padding_shape = []
    for i in range(kernel.ndim):
        M = kernel.shape[i]
        N = mask.shape[i]
        Kp, Km = 0, 0
        p_indices = kernel_support[:, i] < M / 2
        m_indices = kernel_support[:, i] >= M / 2
        if np.any(p_indices):
            Kp = np.max(kernel_support[p_indices, i])
        if np.any(m_indices):
            Km = M - np.min(kernel_support[m_indices, i])
        K = max(Kp, Km)

        mask_padding_shape.append((0, K))
        kernel_padding_shape.append(((N + K - M) // 2, N + K - M - (N + K - M) // 2))

    mask_padded = np.pad(
        mask, mask_padding_shape, constant_values=True
    )  # Pad with True
    kernel_padded = np.fft.ifftshift(
        np.pad(np.fft.fftshift(kernel), kernel_padding_shape, constant_values=False)
    )  # Pad with False

    return (
        np.fft.ifftn(np.fft.fftn(mask_padded) * np.fft.fftn(kernel_padded))[
            tuple([slice(0, N) for N in mask.shape])
        ].real
        > 0.5
    )


def mask_erosion_torch(mask, kernel):
    """Mask erosion.

    Parameters
    ----------
    mask : _type_
        Initial geometry mask. True when pixels are to be removed.
    kernel : _type_
        Kernel used to extend input mask (in wraparound order).
    """
    if mask.ndim != kernel.ndim:
        raise Exception("mask and shape must have the same ndim!")
    for i in range(kernel.ndim):
        if mask.shape[i] < kernel.shape[i]:
            raise Exception("Invalid kernel shape!")

    spatial_dims = tuple(range(-kernel.ndim, 0))
    if len(spatial_dims) == 1:
        conv = F.conv1d
    elif len(spatial_dims) == 2:
        conv = F.conv2d
    elif len(spatial_dims) == 3:
        conv = F.conv3d
    else:
        raise Exception("Invalid spatial dimension!")

    # Identify the support of the kernel and make a window with odd number of pixels/dim centered on it
    kernel_support = torch.argwhere(
        kernel
    )  # List of nonzeros pixels in the kernel of shape (nb_nonzero_pixels, ndim)
    kernel_new_shape = []
    for i in range(kernel.ndim):
        M = kernel.shape[i]
        Kp, Kn = torch.tensor(0), torch.tensor(0)
        p_indices = kernel_support[:, i] < M / 2
        n_indices = kernel_support[:, i] >= M / 2
        if torch.any(p_indices):
            Kp = torch.max(kernel_support[p_indices, i])
        if torch.any(n_indices):
            Kn = M - torch.min(kernel_support[n_indices, i])
        K = torch.max(Kp, Kn).item()  # The support of the kernel is included in [-K, K]
        kernel_new_shape.append(K)
    kernel = torch.fft.fftshift(kernel, dim=spatial_dims)
    kernel_reshaped = kernel[
        tuple(
            [
                slice(-K + M // 2, M // 2 + K + 1)
                for K, M in zip(kernel_new_shape, kernel.shape)
            ]
        )
    ]  # Make a window with odd number of pixels/dim centered on the support of the kernel
    mask_padded = F.pad(
        mask,
        sum(tuple([(K, K) for K in kernel_new_shape])[::-1], ()),
        mode="constant",
        value=True,
    )  # Pad with True

    kernel_time_reversal = torch.flip(
        kernel_reshaped, dims=spatial_dims
    )  # F.conv operations compute the cross-correlation intestead of the convolution, so we need to time-reverse the kernel

    mask_conv = conv(
        mask_padded.unsqueeze(0).float(),
        kernel_time_reversal.unsqueeze(0).unsqueeze(0).float(),
        padding="valid",
    )

    return mask_conv[0] > 0.5


def mask_erosion_para(kernels_indices, mask, kernels):
    """Wrapper for the parallelization of mask_erosion function.

    Parameters
    ----------
    kernels_indices : _type_
        Kernels selection.
    mask : _type_
        Initial geometry mask. True when pixels are to be removed.
    kernels : _type_
        Array of kernels used to extend input mask (in wraparound order).
    """
    ret = np.zeros(kernels_indices.shape + mask.shape, dtype=bool)
    for ret_index, kernel_index in enumerate(kernels_indices):
        ret[ret_index] = mask_erosion(mask, kernels[kernel_index])
    return ret


def mask_erosion_torch_para(kernels_indices, mask, kernels):
    """Wrapper for the parallelization of mask_erosion function.

    Parameters
    ----------
    kernels_indices : _type_
        Kernels selection.
    mask : _type_
        Initial geometry mask. True when pixels are to be removed.
    kernels : _type_
        Array of kernels used to extend input mask (in wraparound order).
    """
    ret = torch.zeros(kernels_indices.shape + mask.shape, dtype=bool).to(mask.device)
    for ret_index, kernel_index in enumerate(kernels_indices):
        print(ret_index, mask.shape, kernels[kernel_index].shape)
        ret[ret_index] = mask_erosion_torch(mask, kernels[kernel_index])
    return ret
