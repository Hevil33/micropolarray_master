import os
import numpy as np
import pandas as pd
from scipy import constants
from pathlib import Path
import time
from micropolarray.cameras import PolarCam


def make_abs_and_create_dir_old(filenames: str):
    cwd = os.getcwd()
    turned_to_list = False
    if type(filenames) is not list:
        turned_to_list = True
        filenames = [filenames]
    # Make path absolute, create dirs if not existing
    for idx, filename in enumerate(filenames):
        if not os.path.isabs(filename):
            filename = cwd + os.path.sep + filename
            filenames[idx] = filename
        parent_dir = os.path.sep.join(filename.split(os.path.sep)[:-1])
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
    if turned_to_list:
        return filenames[0]
    else:
        return filenames


def make_abs_and_create_dir(filename: str):
    path = Path(filename)
    print(path)

    if not path.is_absolute():  # suppose it is in cwd
        path = path.joinpath(Path().cwd(), path)

    if path.suffix:
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
    else:
        if not path.exists():
            path.mkdir()
    return str(path)


def sigma_DN(pix_DN):
    gain = 6.93
    sigma_DN = np.sqrt(gain * pix_DN) / gain
    return sigma_DN


def fix_data(data: np.array, min, max):
    if not (min and max):
        return data
    fixed_data = data.copy()
    fixed_data = np.where(fixed_data > min, fixed_data, min)
    fixed_data = np.where(fixed_data < max, fixed_data, max)
    return fixed_data


def mean_minus_std(data: np.array, stds_n: int = 1) -> float:
    return np.mean(data) - stds_n * np.std(data)


def mean_plus_std(data: np.array, stds_n: int = 1) -> float:
    return np.mean(data) + stds_n * np.std(data)


def normalize2pi(angles_list):
    if type(angles_list) is not list:
        angles_list = [
            angles_list,
        ]
    for i, angle in enumerate(angles_list):
        while angle > 90:
            angle -= 180
        while angle <= -90:
            angle += 180
        angles_list[i] = angle

    return angles_list


# timer decorator
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(
            f"Function {func.__name__} took {round(end - start, 4)} ns to run"
        )
        return result

    return wrapper


def align_keywords_and_data(header, data, sun_center, platescale, binning=1):
    """Fixes antarticor keywords and data to reflect each other.

    Args:
        header (dict): fits file header
        data (ndarray): data as np array
        platescale (float): plate scale in arcsec/pixel
        binning (int, optional): binning applied to image. Defaults to 1 (no binning).

    Returns:
        header, data: new fixed header and data
    """
    data = np.rot90(data, k=-1)
    data = np.flip(data, axis=0)

    header["NAXIS1"] = data.shape[0]
    header["NAXIS2"] = data.shape[1]
    height = header["NAXIS1"]
    width = header["NAXIS2"]
    rotation_angle = -10  # degrees
    if binning > 1:
        platescale = platescale * binning

    header["DATE-OBS"] = header["DATE-OBS"] + "T" + header["TIME-OBS"]

    header["WCSNAME"] = "helioprojective-cartesian"
    # header["DSUN_OBS"] = 1.495978707e11
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    header["CDELT1"] = platescale
    header["CDELT2"] = platescale
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    header["CRVAL1"] = 0
    header["CRVAL2"] = 0
    header["CROTA2"] = rotation_angle
    y, x = sun_center
    # if year == 2021:
    #    y, x, _ = PolarCam().occulter_pos_2021
    # elif year == 2022:
    #    y, x, _ = PolarCam().occulter_pos_last
    relative_y = y / 1952
    relative_x = x / 1952
    sun_x = int(width * relative_x)
    sun_y = int(height * relative_y)

    # one changes because of rotation, the other because of jhelioviewer representation
    header["CRPIX1"] = height - sun_y  # y, checked
    header["CRPIX2"] = width - sun_x  # x, checked

    return header, data


def get_Bsun_units(
    diffuser_I: float,
    texp_image: float = 1.0,
    texp_diffuser: float = 1.0,
) -> float:
    """Returns the conversion unit for expressing brightness in units of sun brightness. Usage is
    data [units of B_sun] = data[DN] * get_Bsun_units(mean_Bsun_brightness, texp_image, texp_diffuser)

    Args:
        mean_sun_brightness (float): diffuser mean in DN.
        texp_image (float, optional): image exposure time. Defaults to 1.0.
        texp_diffuser (float, optional): diffuser exposure time. Defaults to 1.0.

    Returns:
        float: Bsun units conversion factor
    """
    diffusion_solid_angle = 1.083 * 1.0e-5
    diffuser_transmittancy = 0.28
    Bsun_unit = (
        diffusion_solid_angle
        * diffuser_transmittancy
        * texp_diffuser
        / texp_image
    )
    Bsun_unit = (
        (1.0 / texp_image)
        * diffuser_transmittancy
        * diffusion_solid_angle
        / (diffuser_I / texp_diffuser)
    )

    return Bsun_unit