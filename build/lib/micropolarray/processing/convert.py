import numpy as np
import multiprocessing as mp
from astropy.io import fits
import sys
import tqdm
import datetime
import pytz
from logging import critical


def three_bytes_to_two_ints(filecontent):
    """Needed for parallelization, this will be run by each thread for a slice of the original array.

    Returns:
        np.array: array of saved data
    """
    newarr = np.zeros(shape=int(len(filecontent) * 2 / 3))
    position = 0
    for i in range(0, len(filecontent), 3):
        binsum = "".join([bin(j)[2:].zfill(8) for j in filecontent[i : i + 3]])
        newarr[position] = int(binsum[0:12], 2)
        newarr[position + 1] = int(binsum[16:24] + binsum[12:16], 2)
        position += 2
    return newarr


def nparr_from_binary(filename):
    """Converts a PolarCam binary file into a numpy array. Bytes are saved like this

     - 24 bit (3 bytes)
         1             |   3                |     2
    111111111111       | 1111               | 11111111
     - 2 numbers
    First number 12bit | Second number (little endian) 8+4=12 bit

    Args:
        filename (str): name of the file to be converted

    Raises:
        ValueError: file lenghts is indivisible by the number of chunks requested to parallelize operations

    Returns:
        np.array: array of data from file
    """
    with open(filename, mode="rb") as file:
        filecontent = file.read()  #  serial representation
    image_dimension = 1952
    newarr = np.zeros(shape=image_dimension * image_dimension)
    chunks_n = 32
    chunk_size = len(filecontent) / chunks_n
    if chunk_size % 1 or (chunk_size / 3) % 1:
        raise ValueError("Indivisible by chunks")
    chunk_size = int(chunk_size)
    splitted = np.array(
        [
            filecontent[i * chunk_size : (i + 1) * chunk_size]
            for i in range(chunks_n)
        ]
    )
    with mp.Pool(processes=chunks_n) as p:
        result = p.map(three_bytes_to_two_ints, splitted)
    newarr = np.array(result).reshape((1952, 1952))
    return newarr


def convert_set(filenames, new_filename):
    """Sums a set of filenames and converts them to one fits file.

    Args:
        filenames (list): list of file names to be summed before being converted
        new_filename (str): new .fits file name
    """
    if new_filename.split(".")[-1] != "fits":
        raise ValueError(
            "Trying to save a .fits file to .bin, check new filename"
        )
    if type(filenames) is not list:
        filenames = [
            filenames,
        ]
    images_n = len(filenames)
    arr = np.zeros(shape=(1952, 1952))
    for filename in tqdm.tqdm(filenames):
        arr += nparr_from_binary(filename) / images_n
    hdu = fits.PrimaryHDU(data=arr)
    date_and_time = datetime.datetime.now(
        tz=pytz.timezone("Australia/Perth")
    ).strftime("%Y-%m-%dT%H:%M:%S%z")
    hdu.header["CREATED"] = (
        str(date_and_time),
        "Datetime conversion from bin to fits file (Dome C timezone).",
    )
    hdu.writeto(new_filename, overwrite=True)
