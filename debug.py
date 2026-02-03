import sys
import time
import tracemalloc
from glob import glob
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg
from scipy.ndimage import median_filter
from skimage.filters.thresholding import threshold_multiotsu, threshold_otsu

import micropolarray as ml
from micropolarray.processing.demodulation_errors import *
from micropolarray.processing.image_cleaning import remove_outliers_simple
from micropolarray.processing.linear_roi import DDA, linear_roi, linear_roi_from_polar
from astropy.io import fits
import tracemalloc
from astropy import units as u
from pympler import asizeof
from memory_profiler import profile


# @profile
def main():
    tracemalloc.start()

    fname = "/home/herve/dottorato/antarticor/herve/campagna_2022/results/2021_12_11/corona_0/corona.fits"
    fname = "/home/herve/dottorato/mexicor/eclipse_2024/eclipse_sequence_2024/sample_image_w4096_h3000_t2000_n4_f0_20240408T181639.fits"

    with fits.open(fname) as hdu:
        ...
        # data = hdu[0].data
        # print(type(data[0, 0]))

    demodulator = ml.Demodulator(
        "/home/herve/dottorato/cormag/2023_flight/post_flight_calibration/polarimetria/demo_matrices_computation/demo_matrices_correctangles/notilt/1"
    )
    image = ml.MicropolImage(fname)
    # image = image.demodulate(demodulator)

    # print((asizeof.asizeof(image) * u.byte).to(u.megabyte))
    # current, peak = tracemalloc.get_traced_memory()
    # print(current)

    for key, val in image.__dict__.items():
        ...
        # print(key, val, (asizeof.asizeof(val) * u.byte).to(u.megabyte))

    image = None
    print("Done")


if __name__ == "__main__":
    main()
