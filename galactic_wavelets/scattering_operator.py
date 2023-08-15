import numpy as np
import scipy as scp
import os
import multiprocessing as mp
from functools import partial

from nbodykit.lab import transform, FKPCatalog
from .wavelets import lanusse_fwavelet_3d_bank, get_dodecahedron_vertices, get_icosphere2_vertices
from .erosion import mask_erosion, mask_erosion_para
from simbig import util as UT
from nbodykit.algorithms.convpower.fkp import get_compensation


class GalaxyCatalogScatteringOp:
    """Wavelet Scattering Transform (WST) for galaxy catalogs.
    This class enables the computation of wavelet scattering transform coefficients from a 3D galaxy catalog.
    Coefficients are of the form:
        S_0(p) = <|x|^p>,
        S_1(l, p) = <|x*psi_l|^p>,
        S_2(l1, l2, p) = <||x*psi_l1|*psi_l2|^p>,
    where * stands for the convolution operation, x is a mesh built from a given galaxy catalog (see self.create_mesh method), and {psi_l} are a set of wavelets.
    These wavelets can be oriented. Their design is inspired by Lanusse+2012 and Eickenberg+2022.
    The wavelet transform can also be eroded with respect to the respective approximate supports of the wavelets (defined by |w| > erosion.theshold*max |w|).
    """
    
    def __init__(self, J, Q=1, kc=np.pi, angular_width=None, aliasing=True,
                 erosion_threshold=None, fwavelets=None, wavelet_masks=None,
                 moments=(1/2, 1, 2),
                 scattering=False,
                 Ngrid=(256, 256, 256), BoxSize=1000.0, BoxCenter=[0.0, 0.0, 0.0], los=None,
                 los_auto_detection=True,
                 kmax=0.5,
                 boxsize_powoftwomult=1,
                 boxsize_auto_adjustment=True,
                 FKP_weights=False,
                 P0=1e4,
                 use_mp=True):
        """Constructor.
        Configure the wavelet parameters (J, Q, kc, angular_width, aliasing, erosion_threshold, fwavelets, wavelet_masks),
        the coefficients that we want to compute (moments, scattering),
        and the mesh grid defined from galaxy catalogs (Ngrid, BoxSize, BoxCenter, los, los_auto_detection, kmax, boxsize_powoftwomult, boxsize_auto_adjustment, FKP_weights, P0).

        If wavelets need to be recomputed, it is safer to create a new object.

        Args:
            J (int): Wavelet bank parameter. Number of j values (i.e. octaves).
            Q (int, optional): Wavelet bank parameter. Number of q values (i.e. scales per octave). Defaults to 1.
            kc (float, optional): Wavelet bank parameter. Cutoff frequency of the mother wavelet. Defaults to np.pi.
            angular_width (float, optional): Wavelet bank parameter. Angular width of the gaussian function used to orient the wavelets. Defaults to None.
            aliasing (bool, optional): Shall we build aliased wavelets when kc > pi/2? Defaults to True.
            erosion_threshold (float, optional): Erosion threshold that controls the erosion of the wavelet transform according to the survey geometry. Defaults to None.
            fwavelets (array, optional): Bank of wavelets in Fourier space. Defaults to None, i.e. will be computed.
            wavelet_masks (array, optional): Erosion masks for each wavelet. Defaults to None, i.e. will be computed.
            moments (tuple, optional): Tuple of exponenets used to compute the WST coefficients. Defaults to (1/2, 1, 2).
            scattering (bool, optional): Shall we compute S_2 coefficients? Defaults to False.
            Ngrid (tuple, optional): Tuple of length 3 describing the mesh grid size on which galaxy catalogs are mapped to. Defaults to (256, 256, 256).
            BoxSize (float or array, optional): Float or array of length 3 describing the mesh physical size (in Mpc/h). Defaults to 1000.0.
            BoxCenter (list, optional): Array of length 3 corresponding to the mesh physical center (in Mpc/h). Defaults to [0.0, 0.0, 0.0].
            los (array, optional): Array of length 3 defining a line of sight for the orientation of the wavelets. Defaults to None.
            los_auto_detection (bool, optional): Auto-detection of the line of sight relying on BoxCenter. Defaults to True.
            kmax (float, optional): If boxsize_autoadjustment is True, correspond to the smallest spatial frequency we want to probe in the survey. Defaults to 0.5.
            boxsize_powoftwomult (int, optional): If boxsize_autoadjustment is True, force self.Ngrid elements to be multiples of 2^boxsize_powoftwomult. Defaults to 1.
            boxsize_auto_adjustment (bool, optional): Auto-adjustment of self.BoxSize and self.Ngrid given kmax and boxsize_powoftwomult. Defaults to True.
            FKP_weights (bool, optional): Shall we define meshes involving FKP weights (see self.create_mesh and Feldman+1994)? Defaults to False.
            P0 (float, optional): Constant involved in the definition of FKP weights. Defaults to 1e4.
            use_mp (bool, optional): Shall we use multiprocessing to compute the wavelets and theirs masks? Defaults to True.
        """
        # Wavelet parameters
        self.J = J
        self.Q = Q
        self.kc = kc
        self.angular_width = angular_width
        self.aliasing = aliasing # Aliasing
        self.orientations = None
        self.nb_orientations = 0 # Number of orientations (as if isotropic for now)

        # Wavelet transform erosion related parameters
        self.fwavelets = fwavelets
        self.wavelet_masks = wavelet_masks
        self.erosion_threshold = erosion_threshold
        self.erosion = self.erosion_threshold is not None

        # Wavelet-based statistics parameters
        self.moments = np.array(moments)
        self.scattering = scattering

        # For FKP weights
        self.FKP_weights = FKP_weights
        self.P0 = P0

        # Multiprocessing
        self.use_mp = use_mp # Use multiprocessing when possible

        # Box parameters
        self.BoxSize = BoxSize # in Mpc
        self.BoxCenter = BoxCenter # in Mpc
        self.los = los
        if los_auto_detection:
            n = np.linalg.norm(self.BoxCenter)
            if n != 0.0:
                self.los = self.BoxCenter / n
            else:
                self.los = np.array([1.0, 0.0, 0.0])
        self.Ngrid = Ngrid
        self.dx, self.dy, self.dz = 1.0, 1.0, 1.0 # default values (in Mpc)
        self.dmin = min([self.dx, self.dy, self.dz])
        self.survey_mask = None # (True for cells out of the survey domain)
        self.dynamic_mask = False # Do we want to update the mask (and thus the masks per wavelet) every time we have discrepancies in the mask that is automatically generated from the input data

        # Auto-adjustment of the BoxSize
        # Readjust BoxSize coordinates to make them multiples of scale_min/2 and multiples of 2^boxsize_powoftwomult
        if boxsize_auto_adjustment:
            scale_min = 2*np.pi / kmax
            for i in range(3):
                q = self.BoxSize[i]//(scale_min/2)
                for j in range(boxsize_powoftwomult + 1):
                    if q % 2**j != 0:
                        q = ((q // 2**j) + 1) * 2**j
                self.BoxSize[i] = q*scale_min/2
            self.Ngrid = (int(np.rint(2*self.BoxSize[0]/scale_min)), int(np.rint(2*self.BoxSize[1]/scale_min)), int(np.rint(2*self.BoxSize[2]/scale_min)))
            print(f"Auto-adjustement of the BoxSize changed BoxSize to {self.BoxSize} and Ngrid to {self.Ngrid}.")
    
    def __call__(self, galaxies, *args, **kwargs):
        return self.forward(galaxies, *args, **kwargs)

    def build_wavelets(self):
        """Build and store wavelets in Fourier space.
        If self.angular_width is not None, then these are oriented wavelets with orientations corresponding to the vertices of a half regular dodecahedron.
        Else, these are isotropic wavelets.
        The wavelet design is inspired by Lanusse+2012 and Eickenberg+2022.
        """
        print("Computing wavelets...")

        # Build wavelets (might be scaled differently for each axis)
        if self.angular_width is not None:
            orientations = get_dodecahedron_vertices(half=True, los=self.los)
            self.orientations = orientations
            self.nb_orientations = len(orientations)
        fwavelets = lanusse_fwavelet_3d_bank(self.Ngrid, J=self.J, Q=self.Q, kc=self.kc,
                                             orientations=self.orientations, angular_width=self.angular_width, 
                                             axes_weights=[self.dx/self.dmin, self.dy/self.dmin, self.dz/self.dmin], 
                                             aliasing=self.aliasing, use_mp=self.use_mp)

        self.fwavelets = fwavelets.real.astype(np.float32) # These wavelets are real-valued in Fourier space + single precision to speed up computations

        print("Done!")

    def compute_wavelet_masks(self):
        """Compute erosion masks for the wavelet transform.
        For each wavelet, we define an approximate support in physical space that is parametererized by self.erosion_threshold parameter.
        This approximate support is defined by the domain where |w| > self.erosion.theshold*max |w|.
        """
        print("Computing masks for the wavelet transform...")

        if self.fwavelets is None:
            self.build_wavelets()

        # Define wavelet supports according to self.erosion_threshold parameter
        wavelets = np.fft.ifftn(self.fwavelets, axes=(-1, -2, -3)).real
        wavelets_renormalized = wavelets.copy()
        for i in range(self.J*self.Q):
            wavelets_renormalized[i] *= 2**(3*i/self.Q)
        wavelets_max = np.absolute(wavelets_renormalized).max()
        wavelets_supports = np.absolute(wavelets_renormalized) > self.erosion_threshold*wavelets_max
        
        # Mask per wavelet for erosion
        if self.nb_orientations != 0:
            wavelets_supports = np.reshape(wavelets_supports, (-1,) + wavelets_supports.shape[-3:])
        masks_per_wavelet = np.zeros(wavelets_supports.shape, dtype=bool)
        if self.use_mp:
            mask_erosion_para_partial = partial(mask_erosion_para, mask=self.survey_mask, kernels=wavelets_supports)
            work = np.arange(masks_per_wavelet.shape[0])
            nb_processes = min(os.cpu_count(), len(work))
            work_list = np.array_split(work, nb_processes)
            pool = mp.Pool(processes=nb_processes)
            results = pool.map(mask_erosion_para_partial, work_list)
            cnt = 0
            for i in range(len(results)):
                masks_per_wavelet[cnt: cnt + results[i].shape[0]] = results[i]
                cnt += results[i].shape[0]
            pool.close()
        else:
            for i in range(masks_per_wavelet.shape[0]):
                masks_per_wavelet[i] = mask_erosion(self.survey_mask, wavelets_supports[i])
        if self.nb_orientations != 0:
            masks_per_wavelet = np.reshape(masks_per_wavelet, (self.J*self.Q, self.nb_orientations) + masks_per_wavelet.shape[-3:])
        self.wavelet_masks = ~masks_per_wavelet
        self.wavelet_masks_vol_fraction = self.wavelet_masks.sum(axis=(-3, -2, -1)) / (self.Ngrid[0]*self.Ngrid[1]*self.Ngrid[2])

        print("Done!")

    def create_mesh(self, galaxies, randoms=None, survey_geometry=True, cosmo=None, normalize=True, ret_shot_noise=False, ret_norm=False, ret_ngal=False, ret_alpha=False):
        """Build a regular mesh from a galaxy catalog.
        The size of the mesh and origin of coordinates are fixed by self.Ngrid, self.BoxSize, and self.BoxCenter variables.

        If survey_geometry is False, the mesh just corresponds to a number density field of galaxies minus its mean, assuming periodic boundary conditions.
        galaxies is expected to already include a key of cartesian positions called "Position".

        If survey_geometry is True, the mesh is w_{c,g}(r)*n_g(r) - alpha*w_{c,r}*n_r(r) if a random catalog is provided, or just w_{c,g}(r)*n_g(r) is randoms is None.
        Here w_{c,g/r} describes completeness weights, n_g and n_r are number density fields corresponding to the galaxy and random catalog, respectively, and alpha is a normalization factor.
        If required (when self.FKP_weights is True), n_g and n_r are also multiplied by FKP weights (Feldman+1994), i.e. 1/(1+self.P0*nbar_{g/r}(r)).
        Objects positions are expected to be given in RA/DEC/Z coordinates. These are converted into cartesian coordinates for a given cosmology.
        Default behavior is now to normalize this field by dividing it by a global normalization factor equal to \sqrt{\sum alpha*nbar_r*w_{c,r}*w_{FKP, r}**2} (see Scoccimarro+2015 Eq. (49) and Feldman+1994 Eq. (2.1.7)).

        Number density fields are computed from point clouds using Triangular Shaped Cloud mass assignment scheme.

        Args:
            galaxies (catalog): Catalog of galaxies.
            randoms (catalog, optional): Catalog of random points describing the geometry. Defaults to None.
            survey_geometry (bool, optional): Should we take into account a specific survey geometry, or just use periodic boundary conditions? Defaults to True.
            cosmo (nbodykit.cosmology.cosmology.Cosmology, optional): Choice of cosmology. Defaults to None (get SimBIG fiducial cosmology).
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

        # Make copies of the input catalogues
        galaxies = galaxies.copy()
        if randoms is not None:
            randoms = randoms.copy()

        # Survey geometry or not
        if not survey_geometry:
            # paint galaxies to mesh
            mesh_catalog = galaxies.to_mesh(resampler='tsc', Nmesh=self.Ngrid, BoxSize=self.BoxSize,  
                    compensated=True, position='Position')
            deltan = mesh_catalog.compute(Nmesh=self.Ngrid)
            shot_noise = deltan.attrs['shotnoise']
            #TODO: ngal = ?
            #TODO: normalization =?
        else:
            # Get fiducial cosmology if not provided
            if cosmo is None:
                cosmo =  UT.fiducial_cosmology()

            # RA, DEC, Z -> 3D
            galaxy_positions = transform.SkyToCartesian(galaxies['RA'], galaxies['DEC'], galaxies['Z'], cosmo=cosmo)
            if randoms is not None:
                randoms_positions = transform.SkyToCartesian(randoms['RA'], randoms['DEC'], randoms['Z'], cosmo=cosmo)
            
            # make 3D field
            ng_nofz = UT.get_nofz(np.array(galaxies['Z']), galaxies.attrs['fsky'], cosmo=cosmo)
            nbar_galaxies = ng_nofz(np.array(galaxies['Z']))
            if randoms is not None:
                nbar_randoms = ng_nofz(np.array(randoms['Z']))
            
            # FKP weights
            if self.FKP_weights:
                fkp_weight_galaxies = 1. / (1. + self.P0 * nbar_galaxies)
                if randoms is not None:
                    fkp_weight_randoms = 1. / (1. + self.P0 * nbar_randoms)
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
            mesh_catalog = fkp_galaxies.to_mesh(Nmesh=self.Ngrid, nbar='NZ', fkp_weight='WEIGHT_FKP', 
                                        comp_weight='Weight', resampler='tsc', BoxSize=self.BoxSize, BoxCenter=self.BoxCenter)

            # paint the 1st FKP density field to the mesh (paints: data - alpha*randoms, essentially)
            deltan_noncomp = mesh_catalog.compute(Nmesh=self.Ngrid)
  
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

            ngal = galaxies['Weight'].sum()

        # the real-space grid
        if survey_geometry:
            offset = mesh_catalog.attrs['BoxCenter'] + 0.5*mesh_catalog.pm.BoxSize / mesh_catalog.pm.Nmesh
        else:
            offset = 0.5*mesh_catalog.pm.BoxSize / mesh_catalog.pm.Nmesh
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
        if self.dx != dx or self.dy != dy or self.dz != dz:
            self.dx = dx; self.dy = dy; self.dz = dz; self.dmin = min([self.dx, self.dy, self.dz])
            if self.fwavelets is not None:
                print("Box geometry changed, need to reload the wavelets.")
            self.build_wavelets()

        # Mask of the survey (True for cells out of the survey domain)
        if survey_geometry and self.erosion:
            survey_mask = deltan_noncomp == 0
            if self.survey_mask is None or (not np.array_equal(self.survey_mask, survey_mask) and self.dynamic_mask):
                self.survey_mask = survey_mask
                self.compute_wavelet_masks()
                self.survey_mask_vol_fraction = np.logical_not(self.survey_mask).sum(axis=(-3, -2, -1)) / (self.Ngrid[0]*self.Ngrid[1]*self.Ngrid[2])

        # Need to subtract the mean if pbc
        if not survey_geometry:
            deltan -= deltan.mean()
        
        print("Done!")

        output = (deltan.astype(np.float32),) # Single precision
        if ret_shot_noise:
            output += (shot_noise,)
        if ret_norm:
            output += (norm,)
        if ret_ngal:
            output += (ngal,)
        if ret_alpha:
            output += (alpha,)
        return output
    
    def forward(self, galaxies, randoms=None, survey_geometry=True, cosmo=None,
                ret_shot_noise=False, ret_norm=False, ret_ngal=False, ret_alpha=False,
                return_mask_averaged=False, return_full_averaged=False,
                return_abs_filtered_fields=False, return_masked_abs_filtered_fields=False):
        """Compute the WST coefficients associated with a 3D galaxy catalog.

        Args:
            galaxies (catalog): Catalog of galaxies
            randoms (catalog, optional): Catalog of random points describing the geometry. Defaults to None.
            survey_geometry (bool, optional): Should we take into account a specific survey geometry, or just use periodic boundary conditions? Defaults to True.
            cosmo (nbodykit.cosmology.cosmology.Cosmology, optional): Choice of cosmology. Defaults to None (get SimBIG fiducial cosmology).
            ret_shot_noise (bool, optional): Return shot noise estimate. This should be consistent with nbodykit FFTPower/ConvolvedFFTPower estimates. Defaults to False.
            ret_norm (bool, optional): Return normalization factor. This should be consistent with nbodykit ConvolvedFFTPower normalization factor. Defaults to False.
            return_mask_averaged (bool, optional): For debug, return the S_1 coefficients as computed after erosion. Defaults to False.
            return_full_averaged (bool, optional): For debug, return the S_1 coefficients as computed without erosion. Defaults to False.
            return_abs_filtered_fields (bool, optional): For debug, return the modulus of the wavelet transform before erosion. Defaults to False.
            return_masked_abs_filtered_fields (bool, optional): For debug, return the modulus of the wavelet transform after erosion. Defaults to False.
            ret_ngal (bool, optional): Return weighted number of galaxies. Defaults to False.
            ret_alpha (bool, optional): Return alpha. Defaults to False.

        Returns:
            tuple: Tuple containing the S_0 coefficients, S_1 coefficients, S_2 coefficients (optional), shot noise (optional), normalization factor (optional), and various debug variables (optional).
        """
        # Build mesh and get the output
        ret = self.create_mesh(galaxies, randoms=randoms, survey_geometry=True, cosmo=cosmo, ret_shot_noise=ret_shot_noise, ret_norm=ret_norm, ret_ngal=ret_ngal, ret_alpha=ret_alpha)
        i = 0
        deltan = ret[i]
        if ret_shot_noise:
            shot_noise = ret[i + 1]
            i += 1
        if ret_norm:
            norm = ret[i + 1]
            i +=1
        if ret_ngal:
            ngal = ret[i + 1]
            i +=1
        if ret_alpha:
            alpha = ret[i + 1]
            i +=1
        
        # Make sure wavelets and masks (when needed) are loaded
        mask_wt = survey_geometry and self.erosion # Shall we do erosion for this catalog?
        if self.fwavelets is None:
            self.build_wavelets()
        if mask_wt and self.wavelet_masks is None:
            self.compute_wavelet_masks()

        print("Computing statistics...")
        
        # Wavelet transform
        fdeltan = scp.fft.fftn(deltan, axes=(-3, -2, -1), workers=-1).astype(np.complex64)
        ffiltered = fdeltan * self.fwavelets
        filtered = scp.fft.ifftn(ffiltered, axes=(-3, -2, -1), workers=-1).astype(np.complex64)
        
        # Modulus of the WT
        abs_filtered = np.abs(filtered)

        # Moments S0
        print("Computing S0 coefficients...")
        absdeltan = np.absolute(deltan)
        wavelet_moments_S0 = np.zeros((self.moments.shape[0]), dtype=np.float32)
        for i in range(self.moments.shape[0]):
             wavelet_moments_S0[i] = (absdeltan ** self.moments[i]).mean(axis=(-3, -2, -1))
        if mask_wt: wavelet_moments_S0 /= self.survey_mask_vol_fraction # Normalization correction

        # Moments S1
        print("Computing S1 coefficients...")
        if self.nb_orientations != 0:
            abs_filtered = np.reshape(abs_filtered, (-1,) + abs_filtered.shape[-3:])
            if mask_wt:
                self.wavelet_masks = np.reshape(self.wavelet_masks, (-1,) + self.wavelet_masks.shape[-3:])
                self.wavelet_masks_vol_fraction = np.reshape(self.wavelet_masks_vol_fraction, (-1,))
        wavelet_moments_S1 = np.zeros((self.moments.shape[0], abs_filtered.shape[0]), dtype=np.float32)
        if not mask_wt:
            modwt = abs_filtered
        else:
            modwt = abs_filtered * self.wavelet_masks
        for i in range(self.moments.shape[0]):
             wavelet_moments_S1[i] = (modwt ** self.moments[i]).mean(axis=(-3, -2, -1))
        if mask_wt: wavelet_moments_S1 /= self.wavelet_masks_vol_fraction # Normalization correction
        if self.nb_orientations != 0:
            abs_filtered = np.reshape(abs_filtered, (self.J*self.Q, self.nb_orientations) + abs_filtered.shape[-3:])
            wavelet_moments_S1 = np.reshape(wavelet_moments_S1, (self.moments.shape[0], self.J*self.Q, self.nb_orientations))
            if mask_wt:
                self.wavelet_masks = np.reshape(self.wavelet_masks, (self.J*self.Q, self.nb_orientations) + self.wavelet_masks.shape[-3:])
                self.wavelet_masks_vol_fraction = np.reshape(self.wavelet_masks_vol_fraction, (self.J*self.Q, self.nb_orientations))

        # Moments S2
        if self.scattering:
            print("Computing S2 coefficients...")
            abs_filtered_f = scp.fft.fftn(abs_filtered, axes=(-3, -2, -1), workers=-1).astype(np.complex64)
            cnt = 0
            if self.nb_orientations != 0:
                wavelet_moments_S2 = np.zeros((self.moments.shape[0], self.J*self.Q * (self.J*self.Q - 1) // 2, self.nb_orientations, self.nb_orientations), dtype=np.float32)
                for j1 in range(self.J*self.Q - 1):
                    nb_scales = self.J*self.Q - 1 - j1
                    for t1 in range(self.nb_orientations):
                        curr_field_f = abs_filtered_f[j1, t1] * self.fwavelets[j1 + 1:]
                        curr_field = scp.fft.ifftn(curr_field_f, axes=(-3, -2, -1), workers=-1).astype(np.complex64)
                        curr_field = np.abs(curr_field)
                        if mask_wt:
                            curr_field *= self.wavelet_masks[j1 + 1:]
                        for i in range(self.moments.shape[0]):
                            wavelet_moments_S2[i, cnt:cnt + nb_scales, t1] = (curr_field ** self.moments[i]).mean(axis=(-3, -2, -1))
                        if mask_wt: wavelet_moments_S2[i, cnt:cnt + nb_scales, t1] /= self.wavelet_masks_vol_fraction[j1 + 1:] # Normalization correction
                    cnt += self.J*self.Q - 1 - j1
            else:
                wavelet_moments_S2 = np.zeros((self.moments.shape[0], self.J*self.Q * (self.J*self.Q - 1) // 2), dtype=np.float32)
                for j1 in range(self.J*self.Q - 1):
                    nb_scales = self.J*self.Q - 1 - j1
                    curr_field_f = abs_filtered_f[j1] * self.fwavelets[j1 + 1:]
                    curr_field = scp.fft.ifftn(curr_field_f, axes=(-3, -2, -1), workers=-1).astype(np.complex64)
                    curr_field = np.abs(curr_field)
                    if mask_wt:
                        curr_field *= self.wavelet_masks[j1 + 1:]
                    for i in range(self.moments.shape[0]):
                        wavelet_moments_S2[i, cnt:cnt + nb_scales] = (curr_field ** self.moments[i]).mean(axis=(-3, -2, -1))
                    if mask_wt: wavelet_moments_S2[i, cnt:cnt + nb_scales] /= self.wavelet_masks_vol_fraction[j1 + 1:] # Normalization correction
                    cnt += self.J*self.Q - 1 - j1
        
        print("Done!")

        output = (wavelet_moments_S0, wavelet_moments_S1,)
        if self.scattering:
            output += (wavelet_moments_S2,)
        if ret_shot_noise:
            output += (shot_noise,)
        if ret_norm:
            output += (norm,)
        if ret_ngal:
            output += (ngal,)
        if ret_alpha:
            output += (alpha,)

        # For debug
        if return_mask_averaged:
            abs_filtered_masked = abs_filtered * self.wavelet_masks
            masked_average = np.zeros((self.moments.shape[0], abs_filtered.shape[0]))
            for i in range(self.moments.shape[0]):
                masked_average[i] = (abs_filtered_masked ** self.moments[i]).mean(axis=(-3, -2, -1)) / self.wavelet_masks_vol_fraction
            output = output + (masked_average,)
        if return_full_averaged:
            full_average = np.zeros((self.moments.shape[0], abs_filtered.shape[0]))
            for i in range(self.moments.shape[0]):
                full_average[i] = (abs_filtered ** self.moments[i]).mean(axis=(-3, -2, -1)) / self.wavelet_masks_vol_fraction
            output = output + (full_average,)
        if return_abs_filtered_fields:
            output = output + (abs_filtered,)
        if return_masked_abs_filtered_fields:
            output = output + (abs_filtered * self.wavelet_masks,)
        
        return output

class ScatteringOp:
    """Wavelet Scattering Transform (WST) for fields.
    This class enables the computation of wavelet scattering transform coefficients from a 3D density field.
    Coefficients are of the form:
        S_0(p) = <|x|^p>,
        S_1(l, p) = <|x*psi_l|^p>,
        S_2(l1, l2, p) = <||x*psi_l1|*psi_l2|^p>,
    where * stands for the convolution operation, x is the target field, and {psi_l} are a set of wavelets.
    These wavelets can be oriented. Their design is inspired by Lanusse+2012 and Eickenberg+2022.
    The wavelet transform can also be eroded with respect to the respective approximate supports of the wavelets (defined by |w| > erosion.theshold*max |w|).
    """
    
    def __init__(self, J, Q=1, kc=np.pi, angular_width=None, aliasing=True,
                 erosion_threshold=None, fwavelets=None, wavelet_masks=None,
                 moments=(1/2, 1, 2),
                 scattering=False,
                 Ngrid=(256, 256, 256), BoxSize=1000.0, BoxCenter=[0.0, 0.0, 0.0], los=None,
                 los_auto_detection=True,
                 use_mp=True):
        """Constructor.
        Configure the wavelet parameters (J, Q, kc, angular_width, aliasing, erosion_threshold, fwavelets, wavelet_masks),
        the coefficients that we want to compute (moments, scattering),
        and the mesh grid defined from galaxy catalogs (Ngrid, BoxSize, BoxCenter, los, los_auto_detection, kmax, boxsize_powoftwomult, boxsize_auto_adjustment, FKP_weights, P0).

        If wavelets need to be recomputed, it is safer to create a new object.

        Args:
            J (int): Wavelet bank parameter. Number of j values (i.e. octaves).
            Q (int, optional): Wavelet bank parameter. Number of q values (i.e. scales per octave). Defaults to 1.
            kc (float, optional): Wavelet bank parameter. Cutoff frequency of the mother wavelet. Defaults to np.pi.
            angular_width (float, optional): Wavelet bank parameter. Angular width of the gaussian function used to orient the wavelets. Defaults to None.
            aliasing (bool, optional): Shall we build aliased wavelets when kc > pi/2? Defaults to True.
            erosion_threshold (float, optional): Erosion threshold that controls the erosion of the wavelet transform according to the survey geometry. Defaults to None.
            fwavelets (array, optional): Bank of wavelets in Fourier space. Defaults to None, i.e. will be computed.
            wavelet_masks (array, optional): Erosion masks for each wavelet. Defaults to None, i.e. will be computed.
            moments (tuple, optional): Tuple of exponenets used to compute the WST coefficients. Defaults to (1/2, 1, 2).
            scattering (bool, optional): Shall we compute S_2 coefficients? Defaults to False.
            Ngrid (tuple, optional): Tuple of length 3 describing the mesh grid size on which galaxy catalogs are mapped to. Defaults to (256, 256, 256).
            BoxSize (float or array, optional): Float or array of length 3 describing the mesh physical size (in Mpc/h). Defaults to 1000.0.
            BoxCenter (list, optional): Array of length 3 corresponding to the mesh physical center (in Mpc/h). Defaults to [0.0, 0.0, 0.0].
            los (array, optional): Array of length 3 defining a line of sight for the orientation of the wavelets. Defaults to None.
            los_auto_detection (bool, optional): Auto-detection of the line of sight relying on BoxCenter. Defaults to True.
            use_mp (bool, optional): Shall we use multiprocessing to compute the wavelets and theirs masks? Defaults to True.
        """
        # Wavelet parameters
        self.J = J
        self.Q = Q
        self.kc = kc
        self.angular_width = angular_width
        self.aliasing = aliasing # Aliasing
        self.orientations = None
        self.nb_orientations = 0 # Number of orientations (as if isotropic for now)

        # Wavelet transform erosion related parameters
        self.fwavelets = fwavelets
        self.wavelet_masks = wavelet_masks
        self.erosion_threshold = erosion_threshold
        self.erosion = self.erosion_threshold is not None

        # Wavelet-based statistics parameters
        self.moments = np.array(moments)
        self.scattering = scattering

        # Multiprocessing
        self.use_mp = use_mp # Use multiprocessing when possible

        # Box parameters
        self.BoxSize = BoxSize # in Mpc
        self.BoxCenter = BoxCenter # in Mpc
        self.los = los
        if los_auto_detection:
            n = np.linalg.norm(self.BoxCenter)
            if n != 0.0:
                self.los = self.BoxCenter / n
            else:
                self.los = np.array([1.0, 0.0, 0.0])
        self.Ngrid = Ngrid
        self.dx, self.dy, self.dz = 1.0, 1.0, 1.0 # default values (in Mpc)
        self.dmin = min([self.dx, self.dy, self.dz])
        self.survey_mask = None # (True for cells out of the survey domain)
    
    
    def __call__(self, galaxies, *args, **kwargs):
        return self.forward(galaxies, *args, **kwargs)

    def build_wavelets(self):
        """Build and store wavelets in Fourier space.
        If self.angular_width is not None, then these are oriented wavelets with orientations corresponding to the vertices of a half regular dodecahedron.
        Else, these are isotropic wavelets.
        The wavelet design is inspired by Lanusse+2012 and Eickenberg+2022.
        """
        print("Computing wavelets...")

        # Build wavelets (might be scaled differently for each axis)
        if self.angular_width is not None:
            orientations = get_dodecahedron_vertices(half=True, los=self.los)
            self.orientations = orientations
            self.nb_orientations = len(orientations)
        fwavelets = lanusse_fwavelet_3d_bank(self.Ngrid, J=self.J, Q=self.Q, kc=self.kc,
                                             orientations=self.orientations, angular_width=self.angular_width, 
                                             axes_weights=[self.dx/self.dmin, self.dy/self.dmin, self.dz/self.dmin], 
                                             aliasing=self.aliasing, use_mp=self.use_mp)

        self.fwavelets = fwavelets.real.astype(np.float32) # These wavelets are real-valued in Fourier space + single precision to speed up computations

        print("Done!")

    def compute_wavelet_masks(self):
        """Compute erosion masks for the wavelet transform.
        For each wavelet, we define an approximate support in physical space that is parametererized by self.erosion_threshold parameter.
        This approximate support is defined by the domain where |w| > self.erosion.theshold*max |w|.
        """
        print("Computing masks for the wavelet transform...")

        if self.fwavelets is None:
            self.build_wavelets()

        # Define wavelet supports according to self.erosion_threshold parameter
        wavelets = np.fft.ifftn(self.fwavelets, axes=(-1, -2, -3)).real
        wavelets_renormalized = wavelets.copy()
        for i in range(self.J*self.Q):
            wavelets_renormalized[i] *= 2**(3*i/self.Q)
        wavelets_max = np.absolute(wavelets_renormalized).max()
        wavelets_supports = np.absolute(wavelets_renormalized) > self.erosion_threshold*wavelets_max
        
        # Mask per wavelet for erosion
        if self.nb_orientations != 0:
            wavelets_supports = np.reshape(wavelets_supports, (-1,) + wavelets_supports.shape[-3:])
        masks_per_wavelet = np.zeros(wavelets_supports.shape, dtype=bool)
        if self.use_mp:
            mask_erosion_para_partial = partial(mask_erosion_para, mask=self.survey_mask, kernels=wavelets_supports)
            work = np.arange(masks_per_wavelet.shape[0])
            nb_processes = min(os.cpu_count(), len(work))
            work_list = np.array_split(work, nb_processes)
            pool = mp.Pool(processes=nb_processes)
            results = pool.map(mask_erosion_para_partial, work_list)
            cnt = 0
            for i in range(len(results)):
                masks_per_wavelet[cnt: cnt + results[i].shape[0]] = results[i]
                cnt += results[i].shape[0]
            pool.close()
        else:
            for i in range(masks_per_wavelet.shape[0]):
                masks_per_wavelet[i] = mask_erosion(self.survey_mask, wavelets_supports[i])
        if self.nb_orientations != 0:
            masks_per_wavelet = np.reshape(masks_per_wavelet, (self.J*self.Q, self.nb_orientations) + masks_per_wavelet.shape[-3:])
        self.wavelet_masks = ~masks_per_wavelet
        self.wavelet_masks_vol_fraction = self.wavelet_masks.sum(axis=(-3, -2, -1)) / (self.Ngrid[0]*self.Ngrid[1]*self.Ngrid[2])

        print("Done!")
    
    def forward(self, x, survey_geometry=True,
                return_mask_averaged=False, return_full_averaged=False,
                return_abs_filtered_fields=False, return_masked_abs_filtered_fields=False):
        """Compute the WST coefficients associated with a 3D galaxy catalog.

        Args:
            galaxies (catalog): Catalog of galaxies
            survey_geometry (bool, optional): Should we take into account a specific survey geometry, or just use periodic boundary conditions? Defaults to True.
            return_mask_averaged (bool, optional): For debug, return the S_1 coefficients as computed after erosion. Defaults to False.
            return_full_averaged (bool, optional): For debug, return the S_1 coefficients as computed without erosion. Defaults to False.
            return_abs_filtered_fields (bool, optional): For debug, return the modulus of the wavelet transform before erosion. Defaults to False.
            return_masked_abs_filtered_fields (bool, optional): For debug, return the modulus of the wavelet transform after erosion. Defaults to False.

        Returns:
            tuple: Tuple containing the S_0 coefficients, S_1 coefficients, S_2 coefficients (optional), and various debug variables (optional).
        """
        deltan = x
        
        # Make sure wavelets and masks (when needed) are loaded
        mask_wt = survey_geometry and self.erosion # Shall we do erosion?
        if self.fwavelets is None:
            self.build_wavelets()
        if mask_wt and self.wavelet_masks is None:
            self.compute_wavelet_masks()

        print("Computing statistics...")
        
        # Wavelet transform
        fdeltan = scp.fft.fftn(deltan, axes=(-3, -2, -1), workers=-1).astype(np.complex64)
        ffiltered = fdeltan * self.fwavelets
        filtered = scp.fft.ifftn(ffiltered, axes=(-3, -2, -1), workers=-1).astype(np.complex64)
        
        # Modulus of the WT
        abs_filtered = np.abs(filtered)

        # Moments S0
        print("Computing S0 coefficients...")
        absdeltan = np.absolute(deltan)
        wavelet_moments_S0 = np.zeros((self.moments.shape[0]), dtype=np.float32)
        for i in range(self.moments.shape[0]):
             wavelet_moments_S0[i] = (absdeltan ** self.moments[i]).mean(axis=(-3, -2, -1))
        if mask_wt: wavelet_moments_S0 /= self.survey_mask_vol_fraction # Normalization correction

        # Moments S1
        print("Computing S1 coefficients...")
        if self.nb_orientations != 0:
            abs_filtered = np.reshape(abs_filtered, (-1,) + abs_filtered.shape[-3:])
            if mask_wt:
                self.wavelet_masks = np.reshape(self.wavelet_masks, (-1,) + self.wavelet_masks.shape[-3:])
                self.wavelet_masks_vol_fraction = np.reshape(self.wavelet_masks_vol_fraction, (-1,))
        wavelet_moments_S1 = np.zeros((self.moments.shape[0], abs_filtered.shape[0]), dtype=np.float32)
        if not mask_wt:
            modwt = abs_filtered
        else:
            modwt = abs_filtered * self.wavelet_masks
        for i in range(self.moments.shape[0]):
             wavelet_moments_S1[i] = (modwt ** self.moments[i]).mean(axis=(-3, -2, -1))
        if mask_wt: wavelet_moments_S1 /= self.wavelet_masks_vol_fraction # Normalization correction
        if self.nb_orientations != 0:
            abs_filtered = np.reshape(abs_filtered, (self.J*self.Q, self.nb_orientations) + abs_filtered.shape[-3:])
            wavelet_moments_S1 = np.reshape(wavelet_moments_S1, (self.moments.shape[0], self.J*self.Q, self.nb_orientations))
            if mask_wt:
                self.wavelet_masks = np.reshape(self.wavelet_masks, (self.J*self.Q, self.nb_orientations) + self.wavelet_masks.shape[-3:])
                self.wavelet_masks_vol_fraction = np.reshape(self.wavelet_masks_vol_fraction, (self.J*self.Q, self.nb_orientations))

        # Moments S2
        if self.scattering:
            print("Computing S2 coefficients...")
            abs_filtered_f = scp.fft.fftn(abs_filtered, axes=(-3, -2, -1), workers=-1).astype(np.complex64)
            cnt = 0
            if self.nb_orientations != 0:
                wavelet_moments_S2 = np.zeros((self.moments.shape[0], self.J*self.Q * (self.J*self.Q - 1) // 2, self.nb_orientations, self.nb_orientations), dtype=np.float32)
                for j1 in range(self.J*self.Q - 1):
                    nb_scales = self.J*self.Q - 1 - j1
                    for t1 in range(self.nb_orientations):
                        curr_field_f = abs_filtered_f[j1, t1] * self.fwavelets[j1 + 1:]
                        curr_field = scp.fft.ifftn(curr_field_f, axes=(-3, -2, -1), workers=-1).astype(np.complex64)
                        curr_field = np.abs(curr_field)
                        if mask_wt:
                            curr_field *= self.wavelet_masks[j1 + 1:]
                        for i in range(self.moments.shape[0]):
                            wavelet_moments_S2[i, cnt:cnt + nb_scales, t1] = (curr_field ** self.moments[i]).mean(axis=(-3, -2, -1))
                        if mask_wt: wavelet_moments_S2[i, cnt:cnt + nb_scales, t1] /= self.wavelet_masks_vol_fraction[j1 + 1:] # Normalization correction
                    cnt += self.J*self.Q - 1 - j1
            else:
                wavelet_moments_S2 = np.zeros((self.moments.shape[0], self.J*self.Q * (self.J*self.Q - 1) // 2), dtype=np.float32)
                for j1 in range(self.J*self.Q - 1):
                    nb_scales = self.J*self.Q - 1 - j1
                    curr_field_f = abs_filtered_f[j1] * self.fwavelets[j1 + 1:]
                    curr_field = scp.fft.ifftn(curr_field_f, axes=(-3, -2, -1), workers=-1).astype(np.complex64)
                    curr_field = np.abs(curr_field)
                    if mask_wt:
                        curr_field *= self.wavelet_masks[j1 + 1:]
                    for i in range(self.moments.shape[0]):
                        wavelet_moments_S2[i, cnt:cnt + nb_scales] = (curr_field ** self.moments[i]).mean(axis=(-3, -2, -1))
                    if mask_wt: wavelet_moments_S2[i, cnt:cnt + nb_scales] /= self.wavelet_masks_vol_fraction[j1 + 1:] # Normalization correction
                    cnt += self.J*self.Q - 1 - j1
        
        print("Done!")

        output = (wavelet_moments_S0, wavelet_moments_S1,)
        if self.scattering:
            output += (wavelet_moments_S2,)

        # For debug
        if return_mask_averaged:
            abs_filtered_masked = abs_filtered * self.wavelet_masks
            masked_average = np.zeros((self.moments.shape[0], abs_filtered.shape[0]))
            for i in range(self.moments.shape[0]):
                masked_average[i] = (abs_filtered_masked ** self.moments[i]).mean(axis=(-3, -2, -1)) / self.wavelet_masks_vol_fraction
            output = output + (masked_average,)
        if return_full_averaged:
            full_average = np.zeros((self.moments.shape[0], abs_filtered.shape[0]))
            for i in range(self.moments.shape[0]):
                full_average[i] = (abs_filtered ** self.moments[i]).mean(axis=(-3, -2, -1)) / self.wavelet_masks_vol_fraction
            output = output + (full_average,)
        if return_abs_filtered_fields:
            output = output + (abs_filtered,)
        if return_masked_abs_filtered_fields:
            output = output + (abs_filtered * self.wavelet_masks,)
        
        return output
