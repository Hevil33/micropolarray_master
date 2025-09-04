from __future__ import annotations

import datetime
import os
import sys
from logging import info, warning
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from micropolarray.processing.rebin import standard_rebin
from micropolarray.processing.shift import shift
from micropolarray.utils import _make_abs_and_create_dir, fix_data


class Image:
    """Basic image class. Can be initialized from a filename, a filenames list, a numpy array or another Image instance. If multiple filenames are provided, will perform the mean of them unless averageimages is False."""

    def __init__(
        self,
        initializer: str | np.ndarray | Image,
        averageimages: bool = True,
    ):
        self.header = None
        if type(initializer) is str or type(initializer) is list:
            self._init_image_from_file(initializer, averageimages)
        elif type(initializer) is np.ndarray:
            self._init_image_from_data(initializer)
        elif type(initializer) is Image:
            self._init_image_from_image(initializer)

    def _init_image_from_data(self, input_data: np.array) -> None:
        self.data = np.array(input_data)
        self.filename = "image.fits"

    def _init_image_from_file(self, filenames, averageimages) -> None:
        filenames_list = [filenames] if type(filenames) is not list else filenames

        filenames_len = len(filenames_list)
        if filenames_len == 0:
            raise NameError("Can't load files, empty filenames list.")
        datetimes = [0] * filenames_len

        print_info_message = True
        for idx, filename in enumerate(filenames_list):
            if ".raw" in filenames_list:
                raise NameError(
                    "Can't load a raw file, convert it to fits first."
                )  # to fix
            with fits.open(filename) as hul:
                if idx == 0:
                    combined_data = hul[0].data / (
                        1 + int(averageimages) * (filenames_len - 1)
                    )
                    self.header = hul[0].header
                else:
                    if print_info_message:
                        if averageimages:
                            info(f"Averaging {filenames_len} images...")
                        else:
                            info(f"Summing {filenames_len} images...")
                        print_info_message = False
                    combined_data = combined_data + (
                        hul[0].data / (1 + int(averageimages) * (filenames_len - 1))
                    )  # divide by 1 if summing, either n if averaging

                hul.verify("fix")

                try:  # standard format
                    datetimes[idx] = datetime.datetime.strptime(
                        hul[0].header["DATE-OBS"].split(".")[0],
                        "%Y-%m-%dT%H:%M:%S",
                    )
                except ValueError:  # antarticor format
                    datetimes[idx] = datetime.datetime.strptime(
                        hul[0].header["DATE-OBS"]
                        + "-"
                        + hul[0].header["TIME-OBS"].split(".")[0],
                        "%Y-%m-%d-%H:%M:%S",
                    )
                except KeyError:
                    pass

        datetimes = [datetime for datetime in datetimes if datetime != 0]
        if len(datetimes) == 0:
            datetimes = [0]
        self.data = np.array(combined_data)

        if filenames_len > 1:
            self.header["SUMOF"] = (
                filenames_len,
                "Number of files the images is the sum of.",
            )
            datetimes = sorted(datetimes)
            if not datetimes[0] == 0:
                self.header["STRT-OBS"] = (
                    str(datetimes[0]),
                    "Date and time of first image.",
                )
                self.header["END-OBS"] = (
                    str(datetimes[-1]),
                    "Date and time of last image.",
                )
            tempfilename = filenames[0].split(os.path.sep)
            tempfilename[-1] = "SUM_" + tempfilename[-1]
            self.filename = os.path.sep.join(tempfilename)
        else:
            self.filename = filenames[0]

    def _init_image_from_image(self, image: Image):
        self.data = image.data.copy()
        self.filename = image.filename

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    # def data(self, input_data = np.array):
    def data(self, input_data: np.array):
        """Set image data and derived polarization informations, and
        consequently change header."""
        self._data = input_data
        self.height, self.width = input_data.shape
        if self.header is None:
            self.header = fits.PrimaryHDU(data=input_data).header
        else:
            self.header["NAXIS1"] = input_data.shape[1]
            self.header["NAXIS2"] = input_data.shape[0]

    def shift(self, y: int, x: int, missing: float = 0) -> Image:
        """Shifts image by y, x pixels and fills with 0 the remaining space. Positive numbers for up/right shift and negative for down/left shift.

        Args:
            y (int): vertical shift in pix
            x (int): horizontal shift in pix
            missing (float, optional): value used for filling missing values. Defaults to 0

        Returns:
            Image: shifted image copied from the original
        """
        # newdata = shift(self.data, y, x)
        newdata = shift(self.data, y, x, missing)
        newimage = Image(newdata)
        return newimage

    def rebin(self, binning: int) -> Image:
        """Rebins the image, binned each binningxbinning. Sum bins by default.

        Args:
            binning (int): binning to perform. A value of n will be translated in a nxn binning.

        Raises:
            ValueError: negative binning provided

        Returns:
            Image: copy of the input image, rebinned.
        """
        if binning <= 0:
            raise ValueError(f"Negative binning {binning}x{binning}")
        rebinned_image = Image(self)
        rebinned_data = standard_rebin(
            np.array(rebinned_image.data, dtype=float),
            binning,
        )
        rebinned_image.data = rebinned_data

        return rebinned_image

    # ----------------------------------------------------------------
    # -------------------------- SAVING ------------------------------
    # ----------------------------------------------------------------

    def save_as_fits(self, filename: str, fixto: str[float, float] = None):
        """Saves image as fits with current header

        Args:
            filename (str): output filename
            fixto (str[float, float], optional): Set a maximum and minimum range for the data. Defaults to None.

        Raises:
            ValueError: filename does not end with ".fits"
        """
        filepath = Path(filename)
        if filepath.suffix != ".fits":
            raise ValueError(
                f"filename ends with '{filepath.suffix}'filename must be a valid file name, not folder."
            )
        if not filepath.is_absolute():  # suppose it is in cwd
            filepath = Path().cwd() / filepath
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if fixto:
            data = fix_data(self.data, *fixto)
        else:
            data = self.data
        hdu = fits.PrimaryHDU(
            data=data,
            header=self.header,
            do_not_scale_image_data=True,
            uint=False,
        )
        hdu.writeto(filepath, overwrite=True)
        info(f'Image successfully saved to "{filepath}".')

    def save_as_raw(self, filename: str):
        """Saves the image as a raw binary file

        Args:
            filename (str): output filename

        Raises:
            ValueError: filename does not end with ".raw"
        """
        print(self.data.shape)
        if Path(filename).suffix != ".raw":
            raise ValueError("Filename must have .raw extension")
        self.data.astype("int16").tofile(filename)

    # ----------------------------------------------------------------
    # ------------------------------ SHOW ----------------------------
    # ----------------------------------------------------------------

    def show(self, cmap="Greys_r", vmin=None, vmax=None, **kwargs) -> tuple:
        """Shows the image data

        Args:
            cmap (str, optional): figure colorbar. Defaults to "Greys_r".
            vmin (_type_, optional): Minimum value to plot. Defaults to None.
            vmax (_type_, optional): Maximum value to plot. Defaults to None.

        Returns:
            tuple: fig, ax tuple as returned by matplotlib.pyplot.subplots
        """
        data_to_plot = self.data
        data_ratio = data_to_plot.shape[0] / data_to_plot.shape[1]
        # image_fig, imageax = plt.subplots(figsize=(9, 9))
        image_fig, imageax = plt.subplots(dpi=200)
        if vmin is None:
            vmin = np.min(data_to_plot)
        if vmax is None:
            vmax = np.max(data_to_plot)
        pos = imageax.imshow(data_to_plot, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        imageax.set_title(
            f"Image data (avrg {np.mean(data_to_plot, where=np.where(data_to_plot>0, True, False)):3.2f}+-{np.std(data_to_plot, where=np.where(data_to_plot>0, True, False)):3.2f})",
            color="black",
        )
        imageax.set_xlabel("x")
        imageax.set_ylabel("y")
        image_fig.colorbar(pos, ax=imageax, label="DN", fraction=data_ratio * 0.05)

        return image_fig, imageax

    def show_histogram(self, split_pols=False, **kwargs) -> tuple:
        """Print the histogram of the flattened image data

        Args:
            **kwargs (int, optional): arguments to pass to numpy.histogram(), like bins and range.

        Returns:
            tuple: fig, ax tuple as returned by matplotlib.pyplot.subplots
        """
        fig, ax = plt.subplots(dpi=200, constrained_layout=True)
        histo = np.histogram(self.data, **kwargs)
        ax.stairs(*histo, label="Total histogram")
        ax.set_title("Image histogram", color="black")
        ax.set_xlabel("Signal [DN]")
        ax.set_ylabel("Number of pixels")
        plt.legend()

        return fig, ax

    def __add__(self, second) -> Image:
        if type(self) is type(second):
            newdata = self.data + second.data
        else:
            newdata = self.data + second
        return Image(newdata)

    def __sub__(self, second) -> Image:
        if type(self) is type(second):
            newdata = self.data - second.data
        else:
            newdata = self.data - second
        return Image(newdata)

    def __mul__(self, second) -> Image:
        if type(self) is type(second):
            newdata = self.data * second.data
        else:
            newdata = self.data * second
        return Image(newdata)

    def __truediv__(self, second) -> Image:
        if type(self) is type(second):
            newdata = np.divide(self.data, second.data, where=second.data != 0)
        else:
            newdata = np.divide(self.data, second, where=second != 0)
        return Image(newdata)
