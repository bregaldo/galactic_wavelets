import numpy as np


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

    if kernel.ndim != kernel.ndim:
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

    mask_padded = np.pad(mask, mask_padding_shape, constant_values=True) # Pad with True
    kernel_padded = np.fft.ifftshift(np.pad(np.fft.fftshift(kernel), kernel_padding_shape, constant_values=False)) # Pad with False

    return np.fft.ifftn(np.fft.fftn(mask_padded) * np.fft.fftn(kernel_padded))[tuple([slice(0, N) for N in mask.shape])].real > 0.5

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
