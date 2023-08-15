import shutil
import gzip
import urllib
import sys, os
from nbodykit.lab import FITSCatalog, transform, cosmology


def print_download_progress(count, block_size, total_size):
    """
    Code taken from https://nbodykit.readthedocs.io/en/latest/cookbook/boss-dr12-data.html
    """
    pct_complete = float(count * block_size) / total_size
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()

def download_boss_data(download_dir):
    """
    Download the FITS data needed for this notebook to the specified directory.

    Code taken from: https://nbodykit.readthedocs.io/en/latest/cookbook/boss-dr12-data.html

    Parameters
    ----------
    download_dir : str
        the data will be downloaded to this directory
    """

    urls = ['https://data.sdss.org/sas/dr12/boss/lss/galaxy_DR12v5_LOWZ_South.fits.gz',
            'https://data.sdss.org/sas/dr12/boss/lss/random0_DR12v5_LOWZ_South.fits.gz']
    filenames = ['galaxy_DR12v5_LOWZ_South.fits', 'random0_DR12v5_LOWZ_South.fits']

    # download both files
    for i, url in enumerate(urls):

        # the download path
        filename = url.split('/')[-1]
        file_path = os.path.join(download_dir, filename)
        final_path = os.path.join(download_dir, filenames[i])

        # do not re-download
        if not os.path.exists(final_path):
            print("Downloading %s" % url)

            # Check if the download directory exists, otherwise create it.
            if not os.path.exists(download_dir):
                os.makedirs(download_dir)

            # Download the file from the internet.
            file_path, _ = urllib.request.urlretrieve(url=url,
                                                      filename=file_path,
                                                      reporthook=print_download_progress)

            print()
            print("Download finished. Extracting files.")

            # unzip the file
            with gzip.open(file_path, 'rb') as f_in, open(final_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(file_path)
            print("Done.")
        else:
            print("Data has already been downloaded.")

def load_boss_data():
    """
    Code taken from https://nbodykit.readthedocs.io/en/latest/cookbook/boss-dr12-data.html
    """
    download_boss_data("data/")

    # initialize the FITS catalog objects for data and randoms
    data = FITSCatalog("data/galaxy_DR12v5_LOWZ_South.fits")
    randoms = FITSCatalog("data/random0_DR12v5_LOWZ_South.fits")

    # select a redshift slice
    ZMIN = 0.15
    ZMAX = 0.43

    # slice the randoms
    valid = (randoms['Z'] > ZMIN)&(randoms['Z'] < ZMAX)
    randoms = randoms[valid]

    # slice the data
    valid = (data['Z'] > ZMIN)&(data['Z'] < ZMAX)
    data = data[valid]

    # the fiducial BOSS DR12 cosmology
    cosmo = cosmology.Cosmology(h=0.676).match(Omega0_m=0.31)

    # add Cartesian position column
    data['Position'] = transform.SkyToCartesian(data['RA'], data['DEC'], data['Z'], cosmo=cosmo)
    randoms['Position'] = transform.SkyToCartesian(randoms['RA'], randoms['DEC'], randoms['Z'], cosmo=cosmo)

    # completeness weights
    randoms['WEIGHT'] = 1.0
    data['WEIGHT'] = data['WEIGHT_SYSTOT'] * (data['WEIGHT_NOZ'] + data['WEIGHT_CP'] - 1.0)

    return data, randoms, cosmo
