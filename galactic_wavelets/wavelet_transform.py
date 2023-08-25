import torch
import numpy as np
import os
import multiprocessing as mp
from typing import Optional, Tuple, Literal, Union
from functools import partial

from .wavelets import lanusse_fwavelet_3d_bank, get_dodecahedron_vertices
from .erosion import mask_erosion_torch


class WaveletTransform3D(torch.nn.Module):
    def __init__(self,
                 grid_size: Tuple[int, int, int],
                 J: int,
                 Q: int = 1,
                 kc: float = np.pi,
                 angular_width: Optional[float] = None,
                 aliasing: bool = True,
                 los: Tuple[float, float, float] = (0, 0, 1),
                 grid_steps: Tuple[float, float, float] = (1, 1, 1),
                 erosion_threshold: Optional[float] = None,
                 survey_mask: Optional[torch.Tensor] = None,
                 use_mp: bool = True,
                 device: Union[str, int, torch.device] = "cpu"):
        super().__init__()

        # Wavelet parameters
        self.grid_size = grid_size
        self.grid_steps = grid_steps
        self.J = J
        self.Q = Q
        self.kc = kc
        self.angular_width = angular_width
        self.aliasing = aliasing
        self.los = los
        self.wavelets_domain = None # Whether wavelets are stored in Fourier or physical space
        self.register_buffer('wavelets', None) # Tensor of shape (J*Q, O, grid_size[0], grid_size[1], grid_size[2]) where O is the number of orientations (for oriented wavelets)

        # Wavelet transform erosion-related parameters
        self.erosion_threshold = erosion_threshold
        self.erosion = self.erosion_threshold is not None
        if survey_mask is None:
            # Neutral survey mask (False on voxels to be kept)
            survey_mask = torch.zeros(self.grid_size, dtype=bool)
        else:
            survey_mask = survey_mask.bool()
            assert survey_mask.shape == self.grid_size, "Survey mask must have the same shape as the grid!"
        self.register_buffer('survey_mask', survey_mask)
        self.register_buffer('wavelets_masks', None)

        # Multiprocessing
        self.use_mp = use_mp

        # Device
        self.device = device

        # Build wavelets
        self.build_wavelets()

        self.to(device)
    
    def to(self, *args, **kwargs):
        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = device
        super().to(*args, **kwargs)
    
    def get_wavelets(self, domain: Literal['fourier', 'physical'] = "fourier") -> torch.Tensor:
        """Return wavelets in the specified domain.
        """
        if self.wavelets is None:
            self.build_wavelets()

        if domain == "fourier":
            if self.wavelets_domain == "fourier":
                return self.wavelets
            else:
                return torch.fft.fftn(self.wavelets, dim=(-3, -2, -1))
        elif domain == "physical":
            if self.wavelets_domain == "fourier":
                return torch.fft.ifftn(self.wavelets, dim=(-3, -2, -1))
            else:
                return self.wavelets
        else:
            raise ValueError("Unknown domain: {}".format(domain))
    
    def build_wavelets(self) -> None:
        """Build and store wavelets in Fourier space.
        If self.angular_width is not None, then these are oriented wavelets with orientations corresponding to the vertices of a half regular dodecahedron.
        Else, these are isotropic wavelets.
        The wavelet design is inspired by Lanusse+2012 and Eickenberg+2022.
        Note that the wavelets can be scaled differently for each axis.
        """
        print("Computing wavelets...")

        if self.angular_width is not None:
            orientations = get_dodecahedron_vertices(half=True, los=self.los)
            self.orientations = orientations
            self.nb_orientations = len(orientations)
        else:
            self.orientations = None
            self.nb_orientations = 1 # Isotropic wavelets

        dmin = min(self.grid_steps)
        dx, dy, dz = self.grid_steps
        fwavelets = lanusse_fwavelet_3d_bank(self.grid_size,
                                             J=self.J,
                                             Q=self.Q,
                                             kc=self.kc,
                                             orientations=self.orientations, angular_width=self.angular_width, 
                                             axes_weights=[dx/dmin, dy/dmin, dz/dmin], 
                                             aliasing=self.aliasing,
                                             use_mp=self.use_mp)

        # These wavelets are real-valued in Fourier space + single precision to speed up computations
        self.wavelets = torch.from_numpy(fwavelets.real.astype(np.float32))
        self.wavelets_domain = "fourier"
        if self.angular_width is None:
            self.wavelets = self.wavelets.unsqueeze(1)

        print("Done!")

        assert self.wavelets.shape == (self.J*self.Q, self.nb_orientations, self.grid_size[0], self.grid_size[1], self.grid_size[2])

        self.to(self.device)
        self.compute_wavelet_masks()

    def compute_wavelet_masks(self) -> None:
        """Compute erosion masks for the wavelet transform.
        For each wavelet, we define an approximate support in physical space that is parametererized by self.erosion_threshold parameter.
        This approximate support is defined by the domain where |w| > self.erosion.theshold*max |w|.
        """
        if self.erosion:
            print("Computing masks for the wavelet transform...")

            assert self.survey_mask is not None, "Survey mask must be defined to compute wavelet masks!"

            # Define wavelet supports according to self.erosion_threshold parameter
            aw = torch.absolute(self.get_wavelets("physical"))
            aw_max = torch.amax(aw, dim=(-3, -2, -1), keepdim=True)
            w_supports = aw > self.erosion_threshold*aw_max
            
            # Mask per wavelet for erosion
            w_supports = w_supports.view((-1,) + self.grid_size)
            masks_per_w = torch.zeros(w_supports.shape, dtype=bool).to(self.device)
            for i in range(masks_per_w.shape[0]):
                masks_per_w[i] = mask_erosion_torch(self.survey_mask, w_supports[i])
            masks_per_w = masks_per_w.view((self.J*self.Q, self.nb_orientations) + masks_per_w.shape[-3:])
            self.wavelets_masks = masks_per_w

            print("Done!")

    def forward(self,
                x: torch.Tensor,
                jmin: int = 0,
                jmax: Optional[int] = None,
                ret_unmasked: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if jmax is None:
            jmax = self.J*self.Q
        xf = torch.fft.fftn(x, dim=(-3, -2, -1))
        wf = self.get_wavelets("fourier")[jmin:jmax]
        wx = torch.fft.ifftn(xf*wf, dim=(-3, -2, -1))

        wxu = wx
        if self.erosion:
            if ret_unmasked:
                wxu = wxu.clone()
            wx[self.wavelets_masks[jmin:jmax]] = torch.nan

        if ret_unmasked:
            return wx, wxu
        else:
            return wx
