import numpy as np 
import nbodykit.lab as NBlab
from astropy.stats import scott_bin_width
from scipy.interpolate import InterpolatedUnivariateSpline

from nbodykit.cosmology.cosmology import Cosmology


def get_nofz(z: np.ndarray,
             fsky: float,
             cosmo: Cosmology) -> InterpolatedUnivariateSpline:
    ''' calculate nbar(z) given redshift values and f_sky (sky coverage
    fraction)

    Function taken from the SimBIG repository (https://github.com/changhoonhahn/simbig).

    Parameters
    ----------
    z : array like
        array of redshift values 
    fsky : float 
        sky coverage fraction  
    cosmo : cosmology object 
        cosmology to calculate comoving volume of redshift bins 

    Returns
    -------
    number density at input redshifts: nbar(z) 

    Notes
    -----
    * based on nbdoykit implementation 

    '''
    # calculate nbar(z) for each galaxy 
    _, edges = scott_bin_width(z, return_bins=True)

    dig = np.searchsorted(edges, z, "right")
    N = np.bincount(dig, minlength=len(edges)+1)[1:-1]

    R_hi = cosmo.comoving_distance(edges[1:]) # Mpc/h
    R_lo = cosmo.comoving_distance(edges[:-1]) # Mpc/h

    dV = (4./3.) * np.pi * (R_hi**3 - R_lo**3) * fsky

    nofz = InterpolatedUnivariateSpline(0.5*(edges[1:] + edges[:-1]), N/dV, ext='const')
    
    return nofz


def fiducial_cosmology() -> Cosmology:
    ''' hardcoded fiducial cosmology. This is equivalent to the fiducial cosmology 
    of Quijote. This cosmology is meant to be used for calculating galaxy observables.

    Function taken from the SimBIG repository (https://github.com/changhoonhahn/simbig).

    Returns
    -------
    cosmo : nbodykit.lab.cosmology object
        cosmology object with the fiducial cosmology 
    '''
    # Om, Ob, h, ns, s8 = Halos.Quijote_fiducial_cosmo()
    Om, Ob, h, ns, s8 = 0.3175, 0.049, 0.6711, 0.9624, 0.834

    cosmo = NBlab.cosmology.Planck15.clone(
                h=h,
                Omega0_b=Ob,
                Omega0_cdm=Om - Ob,
                m_ncdm=None,
                n_s=ns)
    return cosmo
