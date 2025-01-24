import numpy as np
import scipy as scp
import os
import torch
from typing import Optional, Tuple, List, Union
from typing_extensions import Unpack

from .wavelet_transform import WaveletTransform3D


class MomentsOp(torch.nn.Module):
    def __init__(self, moments: List[float] = [1/2, 1, 2]):
        super().__init__()
        self.register_buffer("moments", torch.tensor(moments))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x ** self.moments.view((-1,) + (1,)*x.ndim)

class ScatteringOp(torch.nn.Module):
    """Wavelet Scattering Transform (WST) for fields.
    This class enables the computation of wavelet scattering transform coefficients from a 3D field.
    Coefficients are of the form:
        S_0(p) = <|x|^p>,
        S_1(l, p) = <|x*psi_l|^p>,
        S_2(l1, l2, p) = <||x*psi_l1|*psi_l2|^p>,
    where * stands for the convolution operation, x is the target field, and {psi_l} are a set of wavelets.
    These wavelets can be oriented. Their design is inspired by Lanusse+2012 and Eickenberg+2022.
    The wavelet transform can also be eroded with respect to the respective approximate supports of the wavelets (defined by |w| > erosion.theshold*max |w|).
    """
    
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
                 moments: List[float] = [1/2, 1, 2],
                 scattering: bool = False,
                 use_mp: bool = True,
                 device: Union[str, int, torch.device] = "cpu"):
        """Constructor.

        Args:
            grid_size (Tuple[int, int, int]): Grid size.
            J (int):  Wavelet bank parameter. Number of j values (i.e. octaves).
            Q (int, optional): Wavelet bank parameter. Number of q values (i.e. scales per octave). Defaults to 1.
            kc (float, optional): Wavelet bank parameter. Cutoff frequency of the mother wavelet. Defaults to np.pi.
            angular_width (Optional[float], optional): Wavelet bank parameter. Angular width of the gaussian function used to orient the wavelets. Defaults to None.
            aliasing (bool, optional): Whether to use aliased wavelets. Defaults to True.
            los (Tuple[float, float, float], optional): Line-of-sight for the orientation of wavelets. Defaults to (0, 0, 1).
            grid_steps (Tuple[float, float, float], optional): Grid steps. Defaults to (1, 1, 1).
            erosion_threshold (Optional[float], optional): Erosion threshold that controls the erosion of the wavelet transform according to the survey mask. Defaults to None.
            survey_mask (Optional[torch.Tensor], optional): Survey mask. Defaults to None.
            moments (List[float], optional): Exponents used to compute the WST coefficients. Defaults to [1/2, 1, 2].
            scattering (bool, optional): Whether to compute S_2 coefficients. Defaults to False.
            use_mp (bool, optional): Whether to use multiprocessing to compute the wavelets. Defaults to True.
            device (Union[str, int, torch.device], optional): Device. Defaults to "cpu".
        """
        super().__init__()

        # Wavelet transform
        self.wt_op = WaveletTransform3D(grid_size,
                                        J,
                                        Q=Q,
                                        kc=kc,
                                        angular_width=angular_width,
                                        aliasing=aliasing,
                                        los=los,
                                        grid_steps=grid_steps,
                                        erosion_threshold=erosion_threshold,
                                        survey_mask=survey_mask,
                                        use_mp=use_mp,
                                        device=device)

        # Additional operators
        self.abs_op = lambda x: torch.abs(x) # Modulus
        self.moments_op = MomentsOp(moments=moments) # Exponentiation
        self.mean_op = lambda x: x.nanmean(dim=(-3, -2, -1)) # Average over the mesh grid (ignoring NaNs)

        # Wavelet scattering parameters
        self.scattering = scattering

        self.to(device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Compute the WST coefficients of the input field.

        Args:
            x (torch.Tensor): Input field.

        Returns:
            tuple: Tuple containing the S_0 coefficients, S_1 coefficients, S_2 coefficients (optional), and various debug variables (optional).
        """

        print("Computing statistics...")

        wt_op, abs_op, moments_op, mean_op = self.wt_op, self.abs_op, self.moments_op, self.mean_op
        
        # Preliminary variables
        ax = abs_op(x)
        wtx, wtx_unmasked = wt_op(x, ret_unmasked=True)
        awtx = abs_op(wtx)
        if wtx is wtx_unmasked or not self.scattering: # To save memory
            awtx_unmasked = awtx
        else:
            del wtx
            awtx_unmasked = abs_op(wtx_unmasked)
            del wtx_unmasked

        # Moments S0
        print("Computing S0 coefficients...")
        s0_x = mean_op(moments_op(ax))
        del ax # To save memory

        # Moments S1
        print("Computing S1 coefficients...")
        s1_x = mean_op(moments_op(awtx))
        if awtx is not awtx_unmasked: del awtx # To save memory

        # Moments S2
        if self.scattering:
            print("Computing S2 coefficients...")
            J, Q = self.wt_op.J, self.wt_op.Q
            M = moments_op.moments.shape[0]
            L = self.wt_op.nb_orientations
            Jeff = J*Q

            s2_x = torch.zeros((M, Jeff*(Jeff-1)//2, L, L),
                               dtype=s1_x.dtype,
                               device=s1_x.device)
            cnt = 0
            for j1 in range(Jeff - 1):
                nb_scales = Jeff - 1 - j1
                for t1 in range(L):
                    s2_x[:, cnt:cnt + nb_scales, t1] = mean_op(moments_op(abs_op(wt_op(awtx_unmasked[j1, t1], jmin=j1+1))))
                cnt += Jeff - 1 - j1
        
        print("Done!")

        output = (s0_x, s1_x,)
        if self.scattering:
            output += (s2_x,)
        
        return output
