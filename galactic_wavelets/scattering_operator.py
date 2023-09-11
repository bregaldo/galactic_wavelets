import numpy as np
import scipy as scp
import os
import torch
from typing import Optional, Tuple, List, Union
from typing_extensions import Unpack

from nbodykit.source.catalog import ArrayCatalog
from nbodykit.lab import transform, FKPCatalog
from nbodykit.cosmology.cosmology import Cosmology
from .wavelet_transform import WaveletTransform3D
from .utils import fiducial_cosmology, get_nofz
from nbodykit.algorithms.convpower.fkp import get_compensation


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

class GalaxyCatalogScatteringOp(ScatteringOp):
    """Wavelet Scattering Transform (WST) for galaxy catalogs.
    This class enables the computation of wavelet scattering transform coefficients from a 3D galaxy catalog.
    """
    
    def __init__(self,
                 J: int,
                 box_size: Tuple[float, float, float] = (1000, 1000, 1000),
                 box_center: Tuple[float, float, float] = (0, 0, 0),
                 los_auto_detection: bool = True,
                 kmax: float = 0.5,
                 boxsize_powoftwomult: int = 1,
                 boxsize_auto_adjustment: bool = True,
                 FKP_weights: bool = False,
                 P0: float = 1e4,
                 **kwargs):
        """Constructor.

        Args:
            J (int): Wavelet bank parameter. Number of j values (i.e. octaves).
            box_size (Tuple[float, float, float], optional): Mesh physical size (in Mpc/h). Defaults to (1000, 1000, 1000).
            box_center (Tuple[float, float, float], optional): Mesh physical center (in Mpc/h). Defaults to (0, 0, 0).
            los_auto_detection (bool, optional): Auto-detection of the line of sight relying on box_center. Defaults to True.
            kmax (float, optional): If boxsize_autoadjustment is True, correspond to the smallest spatial frequency we want to probe in the survey. Defaults to 0.5.
            boxsize_powoftwomult (int, optional): If boxsize_autoadjustment is True, force self.grid_size elements to be multiples of 2^boxsize_powoftwomult. Defaults to 1.
            boxsize_auto_adjustment (bool, optional): Auto-adjustment of self.box_size and self.grid_size given kmax and boxsize_powoftwomult. Defaults to True.
            FKP_weights (bool, optional): Whether to define meshes with FKP weights (see self.create_mesh and Feldman+1994). Defaults to False.
            P0 (float, optional): Constant involved in the definition of FKP weights. Defaults to 1e4.
            **kwargs: Additional arguments to configure the wavelet transform (see ScatteringOp.__init__).

        """
        # For FKP weights
        self.FKP_weights = FKP_weights
        self.P0 = P0

        # Box parameters
        self.box_size = np.array(box_size, dtype=float)      # in Mpc
        self.box_center = np.array(box_center, dtype=float)  # in Mpc
        grid_size = kwargs.pop("grid_size") if "grid_size" in kwargs else (64, 64, 64)
        grid_steps = kwargs.pop("grid_steps") if "grid_steps" in kwargs else (1, 1, 1)
        los = kwargs.pop("los") if "los" in kwargs else (0, 0, 1)
        if los_auto_detection:
            n = np.linalg.norm(self.box_center)
            if n != 0.0:
                los = tuple(np.array(box_center) / n)
        self.dynamic_mask = False # Do we want to update the mask (and thus the masks per wavelet) every time we have discrepancies in the mask that is automatically generated from the input data

        # Auto-adjustment of box_size
        # Readjust box_size coordinates to make them multiples of scale_min/2 and multiples of 2^boxsize_powoftwomult
        if boxsize_auto_adjustment:
            scale_min = 2*np.pi / kmax
            for i in range(3):
                q = self.box_size[i]//(scale_min/2)
                for j in range(boxsize_powoftwomult + 1):
                    if q % 2**j != 0:
                        q = ((q // 2**j) + 1) * 2**j
                self.box_size[i] = q*scale_min/2
            grid_size = tuple([int(np.rint(2*self.box_size[i]/scale_min)) for i in range(3)])
            grid_steps = (scale_min/2, scale_min/2, scale_min/2)
            print(f"Auto-adjustement of box_size changed box_size to {self.box_size} and grid_size to {grid_size}.")
        
        super().__init__(grid_size, J, los=los, grid_steps=grid_steps, **kwargs)

    def create_mesh(self,
                    galaxies: ArrayCatalog,
                    randoms: Optional[ArrayCatalog] = None,
                    survey_geometry: bool = True,
                    cosmo: Optional[Cosmology] = None,
                    normalize: bool = True,
                    ret_shot_noise: bool = False,
                    ret_norm: bool = False,
                    ret_ngal: bool = False,
                    ret_alpha: bool = False) -> Tuple[Union[torch.Tensor, float]]:
        """Build a regular mesh from a galaxy catalog.
        The size of the mesh and origin of coordinates are fixed by self.grid_size, self.box_size, and self.box_center variables.

        If survey_geometry is False, the mesh just corresponds to a number density field of galaxies minus its mean, assuming periodic boundary conditions.
        galaxies is expected to already include a key of cartesian positions called "Position".

        If survey_geometry is True, the mesh is w_{c,g}(r)*n_g(r) - alpha*w_{c,r}*n_r(r) if a random catalog is provided, or just w_{c,g}(r)*n_g(r) is randoms is None.
        Here w_{c,g/r} describes completeness weights, n_g and n_r are number density fields corresponding to the galaxy and random catalog, respectively, and alpha is a normalization factor.
        If required (when self.FKP_weights is True), n_g and n_r are also multiplied by FKP weights (Feldman+1994), i.e. 1/(1+self.P0*nbar_{g/r}(r)).
        Objects positions are expected to be given in RA/DEC/Z coordinates. These are converted into cartesian coordinates for a given cosmology.
        Default behavior is now to normalize this field by dividing it by a global normalization factor equal to \sqrt{\sum alpha*nbar_r*w_{c,r}*w_{FKP, r}**2} (see Scoccimarro+2015 Eq. (49) and Feldman+1994 Eq. (2.1.7)).

        Number density fields are computed from point clouds using Triangular Shaped Cloud mass assignment scheme.

        Args:
            galaxies (ArrayCatalog): Catalog of galaxies.
            randoms (Optional[ArrayCatalog], optional): Catalog of random points describing the geometry. Defaults to None.
            survey_geometry (bool, optional): Should we take into account a specific survey geometry, or just use periodic boundary conditions? Defaults to True.
            cosmo (Optional[Cosmology], optional): Choice of cosmology. Defaults to None (get SimBIG fiducial cosmology).
            normalize (bool, optional): Whether we normalize the field similarly to the FKP field. Only coded for survey geometry. Defaults to True.
            ret_shot_noise (bool, optional): Return shot noise estimate. This should be consistent with nbodykit FFTPower/ConvolvedFFTPower estimates. Defaults to False.
            ret_norm (bool, optional): Return normalization factor. This should be consistent with nbodykit ConvolvedFFTPower normalization factor. Defaults to False.
            ret_ngal (bool, optional): Return weighted number of galaxies. Defaults to False.
            ret_alpha (bool, optional): Return alpha. Defaults to False.

        Returns:
            tuple: Tuple containing the computed mesh, as well as the shot noise estimate and the normalization factor is required.
        """
        print("Create mesh from catalog...")

        # Initialization
        shot_noise = 0.0
        norm = 1.0
        alpha = None
        ngal = 0.0

        grid_size = self.wt_op.grid_size

        # Make copies of the input catalogues
        galaxies = galaxies.copy()
        if randoms is not None:
            randoms = randoms.copy()

        # Survey geometry or not
        if not survey_geometry:
            # paint galaxies to mesh
            mesh_catalog = galaxies.to_mesh(resampler='tsc',
                                            Nmesh=grid_size,
                                            BoxSize=self.box_size,  
                                            compensated=True,
                                            position='Position')
            deltan = mesh_catalog.compute(Nmesh=grid_size)
            shot_noise = deltan.attrs['shotnoise']
            #TODO: ngal = ?
            #TODO: normalization =?
        else:
            # Get fiducial cosmology if not provided
            if cosmo is None:
                cosmo =  fiducial_cosmology()

            # RA, DEC, Z -> 3D
            galaxy_positions = transform.SkyToCartesian(galaxies['RA'], galaxies['DEC'], galaxies['Z'], cosmo=cosmo)
            if randoms is not None:
                randoms_positions = transform.SkyToCartesian(randoms['RA'], randoms['DEC'], randoms['Z'], cosmo=cosmo)
            
            # make 3D field
            ng_nofz = get_nofz(np.array(galaxies['Z']), galaxies.attrs['fsky'], cosmo)
            nbar_galaxies = ng_nofz(np.array(galaxies['Z']))
            if randoms is not None:
                nbar_randoms = ng_nofz(np.array(randoms['Z']))
            
            # FKP weights
            if self.FKP_weights:
                fkp_weight_galaxies = 1. / (1. + self.P0*nbar_galaxies)
                if randoms is not None:
                    fkp_weight_randoms = 1. / (1. + self.P0*nbar_randoms)
            else:
                fkp_weight_galaxies = np.ones(len(nbar_galaxies))
                if randoms is not None:
                    fkp_weight_randoms = np.ones(len(nbar_randoms))
            
            galaxies['Position'] = galaxy_positions
            galaxies['WEIGHT_FKP'] = fkp_weight_galaxies
            galaxies['NZ'] = nbar_galaxies
            if randoms is not None:
                randoms['Position'] = randoms_positions
                randoms['WEIGHT_FKP'] = fkp_weight_randoms
                randoms['NZ'] = nbar_randoms

            fkp_galaxies = FKPCatalog(galaxies, randoms)
            mesh_catalog = fkp_galaxies.to_mesh(Nmesh=grid_size,
                                                nbar='NZ',
                                                fkp_weight='WEIGHT_FKP',
                                                comp_weight='Weight',
                                                resampler='tsc',
                                                BoxSize=self.box_size,
                                                BoxCenter=self.box_center)

            # paint the 1st FKP density field to the mesh (paints: data - alpha*randoms, essentially)
            deltan_noncomp = mesh_catalog.compute(Nmesh=grid_size)
  
            # Shot noise computation (inspired from nbodykit ConvolvedFFTPower.shotnoise method)
            alpha = deltan_noncomp.attrs['alpha'] # ratio of data to randoms
            if randoms is not None:
                shot_noise = ((galaxies['Weight']**2*fkp_weight_galaxies**2).sum() + alpha**2*(randoms['Weight']**2*fkp_weight_randoms**2).sum()).compute()
            else:
                shot_noise = (galaxies['Weight']**2*fkp_weight_galaxies**2).sum().compute()

            # Normalization (for comparison with nbodykit ConlvovedFFTPower output)
            if randoms is not None:
                norm = np.sqrt((alpha*nbar_randoms*randoms['Weight']*fkp_weight_randoms**2).sum().compute())
            else:
                norm = 1.0
            shot_noise /= norm**2

            # FFT 1st density field and compensate the resampler transfer kernel
            deltan_noncomp_f = deltan_noncomp.r2c()
            deltan_noncomp_f.apply(out=Ellipsis, **get_compensation(mesh_catalog))
            deltan = deltan_noncomp_f.c2r()

            if normalize:
                deltan /= norm

            ngal = galaxies['Weight'].sum().compute()

        # the real-space grid
        if survey_geometry:
            offset = mesh_catalog.attrs['BoxCenter'] + 0.5*mesh_catalog.pm.BoxSize/mesh_catalog.pm.Nmesh
        else:
            offset = 0.5*mesh_catalog.pm.BoxSize/mesh_catalog.pm.Nmesh
        xgrid = [xx.real + offset[ii] for ii, xx in enumerate(deltan.slabs.optx)]
        
        # Should be consistent:
        #assert self.BoxSize == np.array(mesh_catalog.attrs['BoxSize'])
        #assert self.BoxCenter == np.array(mesh_catalog.attrs['BoxCenter'])

        # Convert pmesh objects into numpy arrays
        if survey_geometry: deltan_noncomp = deltan_noncomp.preview().real.copy()
        deltan = deltan.preview().real.copy()

        # pmesh order seems to be [X, Y, Z]
        x = xgrid[0][:, 0, 0].copy()
        y = xgrid[1][0, :, 0].copy()
        z = xgrid[2][0, 0, :].copy()
        xargs = np.argsort(x)
        yargs = np.argsort(y)
        zargs = np.argsort(z)
        x.sort()
        y.sort()
        z.sort()

        # Reordering of the numpy arrays
        if survey_geometry: deltan_noncomp = ((deltan_noncomp[xargs, :, :])[:, yargs, :])[:, :, zargs]
        deltan = ((deltan[xargs, :, :])[:, yargs, :])[:, :, zargs]

        # Spatial steps of the grid
        dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
        # Might want to make sure that the grid is regular (it is down to the numerical precision for Quijote/nbodykit catalogues)
        #print(np.unique(x[1:] - x[:-1]), np.unique(y[1:] - y[:-1]), np.unique(z[1:] - z[:-1]))
        #assert len(np.unique(x[1:] - x[:-1])) == 1 # Safety
        #assert len(np.unique(y[1:] - y[:-1])) == 1 # Safety
        #assert len(np.unique(z[1:] - z[:-1])) == 1 # Safety
        grid_steps = self.wt_op.grid_steps
        if not np.allclose(grid_steps, (dx, dy, dz)): # Check if the grid steps have changed significantly
            self.wt_op.grid_steps = (dx, dy, dz)
            if self.wt_op.wavelets is not None:
                print("Box geometry changed, need to reload the wavelets.")
            self.wt_op.build_wavelets()

        # Mask of the survey (True for cells out of the survey domain)
        if survey_geometry and self.wt_op.erosion:
            survey_mask = deltan_noncomp == 0
            if self.wt_op.survey_mask is None or (not np.array_equal(self.wt_op.survey_mask, survey_mask) and self.dynamic_mask):
                self.wt_op.survey_mask = survey_mask
                self.wt_op.compute_wavelet_masks()

        # Need to subtract the mean if pbc
        if not survey_geometry:
            deltan -= deltan.mean()
        
        print("Done!")

        output = (torch.from_numpy(deltan.astype(np.float32)).to(self.wt_op.device),) # Single precision
        if ret_shot_noise:
            output += (shot_noise,)
        if ret_norm:
            output += (norm,)
        if ret_ngal:
            output += (ngal,)
        if ret_alpha:
            output += (alpha,)
        return output
    
    def forward(self,
                galaxies: ArrayCatalog,
                **kwargs) -> Tuple[Union[torch.Tensor, float], ...]:
        """Compute the WST coefficients from a 3D galaxy catalog.
           Additional arguments are passed to self.create_mesh.

        Args:
            galaxies (catalog): Catalog of galaxies
            kwargs: See self.create_mesh.
        Returns:
            tuple: Tuple containing the S_0 coefficients, S_1 coefficients, S_2 coefficients (optional), shot noise (optional), normalization factor (optional).
        """
        # Build mesh
        ret = self.create_mesh(galaxies, **kwargs)
        F = ret[0] # 3D tensor representing the field

        # Compute WST coefficients
        output = super().forward(F)
        
        # Return the coefficients and potential additional variables
        return output + ret[1:]
