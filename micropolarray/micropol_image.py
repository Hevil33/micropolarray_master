from __future__ import annotations

import sys
from dataclasses import dataclass
from logging import debug, error, info, warning
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy
from astropy.io import fits
from PIL import Image as PILImage

from micropolarray.cameras import Camera, PolarCam
from micropolarray.image import Image
from micropolarray.polarization_functions import AoLP, DoLP, pB
from micropolarray.processing.chen_wan_liang_calibration import _ifov_jitcorrect
from micropolarray.processing.congrid import congrid
from micropolarray.processing.demodulation import Demodulator
from micropolarray.processing.demosaic import (
    demosaic,
    merge_polarizations,
    split_polarizations,
)
from micropolarray.processing.nrgf import roi_from_polar
from micropolarray.processing.rebin import micropolarray_rebin
from micropolarray.processing.shift import shift_micropol
from micropolarray.utils import (
    _make_abs_and_create_dir,
    fix_data,
    mean_minus_std,
    mean_plus_std,
    timer,
)


@dataclass
class PolParam:
    """Auxiliary class for polarization parameters.

    Members:
        ID (str): parameter identifier
        data (np.array): parameter image as numpy 2D array
        title (str): brief title of the parameter, useful for plotting
        measure_unit (str): initial measure units of the parameter
        fix_data (bool): controls whether data has to be constrained to [0, 4096] interval (not implemented yet)
    """

    ID: str
    data: np.ndarray
    title: str
    measure_unit: str
    fix_data: bool = False


DEFAULT_ANGLES_DIC = None  # sets the micropolarizer orientations with a dictionary {angle : position in superpix 1->3}


def set_default_angles(angles_dic: dict):
    """Sets the default micropolarizer orientations for images.

    Args:
        angles_dic (dict): dictionary {value : pos} where value is the angle in degrees from -90 to 90 and pos is the pixel position in superpixel, from 0 to 3 (position [y, x], fast index x)
    """
    global DEFAULT_ANGLES_DIC
    DEFAULT_ANGLES_DIC = angles_dic


class MicropolImage(Image):
    """Micro-polarizer array image class. Can be initialized from a 2d array, a list of 1 or more file names (use the boolean keyword averageimages to select if sum or average is taken) or another MicropolImage. Dark and flat micropolarray images can also be provided to automatically correct the result."""

    first_call = True  # Avoid repeating messages

    def __init__(
        self,
        initializer: str | np.ndarray | list | MicropolImage,
        angle_dic: dict = None,
        dark: MicropolImage = None,
        flat: MicropolImage = None,
        averageimages: bool = True,
    ):
        self._initialize_internal_variables()
        if angle_dic is None:
            global DEFAULT_ANGLES_DIC
            if DEFAULT_ANGLES_DIC is None:
                if MicropolImage.first_call:
                    warning(
                        f"Micropolarizer orientation dictionary defaults to {PolarCam().angle_dic}, set it via set_default_angles(camera)\n"
                    )
                MicropolImage.first_call = False
                DEFAULT_ANGLES_DIC = PolarCam().angle_dic
            angle_dic = DEFAULT_ANGLES_DIC
        self.angle_dic = angle_dic

        if type(initializer) is list and len(initializer) > 1:
            self._num_of_images = len(initializer)
        else:
            self._num_of_images = 1
        super().__init__(
            initializer=initializer, averageimages=averageimages
        )  # Call generic Image() constructor

        if type(initializer) is MicropolImage:
            self._import_image_parameters(initializer)
        else:
            self.Stokes_vec = self._get_theo_Stokes_vec_components()

        # Apply corrections if needed
        if dark is not None:
            self.subtract_dark(dark=dark)
        if flat is not None:
            self.correct_flat(flat=flat)

    def _initialize_internal_variables(self):
        self._data = None
        self._is_demodulated = False
        self._is_demosaiced = False
        self._binning = 1
        self._flat_subtracted = False
        self._dark_subtracted = False
        self.demosaiced_images = None

    def _import_image_parameters(self, image: MicropolImage):
        self.data = image.data
        self.header = image.header
        self.angle_dic = image.angle_dic
        self.Stokes_vec = image.Stokes_vec
        self._is_demodulated = image._is_demodulated
        self._is_demosaiced = image._is_demosaiced
        self._binning = image._binning
        self._dark_subtracted = image._dark_subtracted
        self._flat_subtracted = image._flat_subtracted
        self.demosaiced_images = image.demosaiced_images

    # ----------------------------------------------------------------
    # -------------------- POLARIZATION PROPERTIES -------------------
    # ----------------------------------------------------------------

    @property
    def I(self) -> PolParam:
        return PolParam("I", self.Stokes_vec[0], "Stokes I", "DN", fix_data=False)

    @property
    def Q(self) -> PolParam:
        return PolParam("Q", self.Stokes_vec[1], "Stokes Q", "DN", fix_data=False)

    @property
    def U(self) -> PolParam:
        return PolParam("U", self.Stokes_vec[2], "Stokes U", "DN", fix_data=False)

    @property
    def pB(self) -> PolParam:
        return PolParam(
            "pB",
            pB(self.Stokes_vec),
            "Polarized brightness",
            "DN",
            fix_data=False,
        )

    @property
    def AoLP(self) -> PolParam:
        return PolParam(
            "AoLP",
            AoLP(self.Stokes_vec),
            "Angle of Linear Polarization",
            "rad",
            fix_data=False,
        )

    @property
    def DoLP(self) -> PolParam:
        return PolParam(
            "DoLP",
            DoLP(self.Stokes_vec),
            "Degree of Linear Polarization",
            "",
            fix_data=False,
        )

    # @property
    # def polparam_list(self) -> list:
    #    return [self.I, self.Q, self.U, self.pB, self.AoLP, self.DoLP]

    @property
    def single_pol_subimages(self):
        return split_polarizations(self.data)

    # removed to reduce memory usage
    """ 
    @property
    def pol0(self) -> PolParam:
        return PolParam(
            "0",
            self.single_pol_subimages[self.angle_dic[0]],
            "0 deg orientation pixels",
            "DN",
            fix_data=False,
        )

    @property
    def pol45(self) -> PolParam:
        return PolParam(
            "45",
            self.single_pol_subimages[self.angle_dic[45]],
            "45 deg orientation pixels",
            "DN",
            fix_data=False,
        )

    @property
    def pol_45(self) -> PolParam:
        return PolParam(
            "-45",
            self.single_pol_subimages[self.angle_dic[-45]],
            "-45 deg orientation pixels",
            "DN",
            fix_data=False,
        )

    @property
    def pol90(self) -> PolParam:
        return PolParam(
            "90",
            self.single_pol_subimages[self.angle_dic[90]],
            "90 deg orientation pixels",
            "DN",
            fix_data=False,
        )
    """

    def _get_single_pols_as_polparam_list(self):
        """Returns the single polarization subimages as a list of PolParam objects, used internally for plots and saving.

        Returns:
            list: list of ordered single polarizations as PolParam list
        """
        return [
            PolParam(
                f"{int(angle):2d}",
                self.single_pol_subimages[self.angle_dic[angle]],
                f"{int(angle):2d} deg orientation pixels",
                "DN",
                fix_data=False,
            )
            for angle in [0, 45, 90, -45]
        ]

    # ----------------------------------------------------------------
    # ---------------------- STOKES COMPONENTS -----------------------
    # ----------------------------------update_dim------------------------------

    def _update_data_and_Stokes(self, newdata: np.ndarray = None):
        if newdata is not None:
            self.data = newdata
        self.Stokes_vec = self._get_theo_Stokes_vec_components()

    def demodulate(
        self,
        demodulator: Demodulator,
        demosaicing: bool = False,
    ) -> MicropolImage:
        """Returns a MicropolImage with polarization parameters calculated from the demodulation tensor provided.

        Args:
            demodulator (Demodulator): Demodulator object containing the demodulation tensor components (see processing.new_demodulation)
            demosaicing (bool, optional): wether to apply demosaicing to the image or not. Set it to False if demodulation matrices have half the dimension of the image. Defaults to True.

        Raises:
            ValueError: raised if image and demodulator do not have the same dimension, for example in case of different binning

        Returns:
            MicropolImage: copy of the input imagreturn e with I, Q, U, pB, DoLP, AoLP calculated from the demodulation tensor.
        """

        info("Demodulating...")
        demodulated_image = MicropolImage(self)
        demodulated_image.Stokes_vec = demodulated_image._get_Stokes_from_demodulator(
            demodulator, demosaicing
        )
        demodulated_image._is_demodulated = True

        info("Image correctly demodulated")
        return demodulated_image

    def _get_theo_Stokes_vec_components(self) -> np.array:
        """
        Computes stokes vector components from four polarized images at four angles, angle_dic describes the coupling between
        poled_images_array[i] <--> angle_dic[i]
        Return:
            stokes vector, shape=(3, poled_images.shape[1], poled_images.shape[0])
        """
        if self._is_demosaiced:
            subimages = self.demosaiced_images
        else:
            subimages = self.single_pol_subimages
        I = 0.5 * np.sum(subimages, axis=0)
        Q = subimages[self.angle_dic[0]] - subimages[self.angle_dic[90]]
        U = subimages[self.angle_dic[45]] - subimages[self.angle_dic[-45]]

        S = np.array([I, Q, U], dtype=float)
        return S

    def _get_Stokes_from_demodulator(
        self,
        demodulator: Demodulator,
        demosaicing: bool,
    ) -> np.array:
        """
        Computes stokes vector components from four polarized images at four angles, angle_dic describes the coupling between
        poled_images_array[i] <--> angle_dic[i]. Calculates:

        I = M_00 * I_1 + M_01 * I_2 + M_02 * I_3 + M_03 * I_4
        Q = M_10 * I_1 + M_11 * I_2 + M_12 * I_3 + M_13 * I_4
        U = M_20 * I_1 + M_21 * I_2 + M_22 * I_3 + M_23 * I_4

        Return:
            stokes vector, shape=(3, poled_images.shape[1], poled_images.shape[0])
        """
        # Corrected with demodulation matrices, S.shape = (4, n, m)
        num_of_malus_parameters = 3  # 3 multiplication params
        pixels_in_superpix = 4
        fit_found_flags = demodulator.fit_found_flags

        if demosaicing:
            self.demosaic()
            splitted_pols = self.demosaiced_images
            mij = np.ones(
                shape=(
                    demodulator.mij.shape[0],
                    demodulator.mij.shape[1],
                    demodulator.mij.shape[2] * 2,
                    demodulator.mij.shape[3] * 2,
                ),
                dtype=float,
            )
            for i in range(num_of_malus_parameters):
                for j in range(pixels_in_superpix):
                    demo_component = demodulator.mij[i, j]
                    mij[i, j, :, :] = merge_polarizations(
                        np.repeat(demo_component[np.newaxis, :, :], 4, axis=0)
                    )  # repeat necessary to avoid numba problems
            fit_found_flags = merge_polarizations(
                np.array(4 * [demodulator.fit_found_flags.astype(float)])
            )  # idk why but fit_found_flags is >f8 as default
        else:
            mij = demodulator.mij
            splitted_pols = self.single_pol_subimages

        """
        print(mij.shape)
        print(splitted_pols.shape)

        pixel_values = np.array(
            [splitted_pol for splitted_pol in splitted_pols],
            dtype=float,
        )
        if (mij[0, 0].shape[0] != pixel_values[0].shape[0]) or (
            mij[0, 0].shape[1] != pixel_values[0].shape[1]
        ):
            raise ValueError(
                f"demodulation matrix {mij[0,0].shape} and images {pixel_values[0].shape} have different shapes. Check that binning is correct."
            )  # sanity check

        T_ij = np.zeros(
            shape=(
                num_of_malus_parameters,
                pixels_in_superpix,
                *splitted_pols[0].shape,
            )
        )
        for i in range(num_of_malus_parameters):
            for j in range(pixels_in_superpix):
                temp_tij = np.multiply(
                    mij[i, j, :, :], pixel_values[j, :, :]
                )  # Matrix product
                T_ij[i, j, :, :] = temp_tij

        I = T_ij[0, 0] + T_ij[0, 1] + T_ij[0, 2] + T_ij[0, 3]
        Q = T_ij[1, 0] + T_ij[1, 1] + T_ij[1, 2] + T_ij[1, 3]
        U = T_ij[2, 0] + T_ij[2, 1] + T_ij[2, 2] + T_ij[2, 3]
        """

        I, Q, U = np.matmul(
            mij,
            np.expand_dims(splitted_pols, axis=0),
            axes=[(-4, -3), (-3, -4), (-4, -3)],
        )[
            :, 0
        ]  # expand dims needed for matmul, output [3, 1, y, x]

        S = np.array([I, Q, U], dtype=float)

        # use theo stokes where fit wasn't found
        if demodulator.fit_found_flags is not None:
            theo_S = self.Stokes_vec
            S = np.where(fit_found_flags == 1.0, S, theo_S)

        return S

    def subtract_dark(self, dark: MicropolImage) -> MicropolImage:
        """Correctly subtracts the input dark image from the image

        Args:
            dark (MicropolImage): dark to subtract

        Returns:
            MicropolImage: copy of input image with dark subtracted
        """
        self.data = self.data - dark.data
        self.data = np.where(self.data >= 0, self.data, 0)  # Fix data
        self.Stokes_vec = self._get_theo_Stokes_vec_components()
        self._dark_subtracted = True
        return self

    def correct_flat(self, flat: MicropolImage) -> MicropolImage:
        """Normalizes the flat and uses it to correStokes_vecct the image.

        Args:
            flat (MicropolImage): flat image, does not need to be normalized.

        Returns:
            MicropolImage: copy of input image corrected by flat
        """
        normalized_flat = flat.data / np.mean(flat.data)

        self.data = np.divide(
            self.data,
            normalized_flat,
            where=normalized_flat != 0.0,
        )

        # self.data = np.where(self.data >= 0, self.data, 0)
        # self.data = np.where(self.data < 4096, self.data, 4096)
        self.Stokes_vec = self._get_theo_Stokes_vec_components()
        self._flat_subtracted = True
        return self

    def correct_ifov(self) -> MicropolImage:
        """Corrects differences in single pixels fields of view inside each superpixel

        Returns:
            MicropolImage: image with data corrected for field of view differences
        """
        corrected_data = self.data.copy()
        corrected_data = _ifov_jitcorrect(self.data, self.height, self.width)
        self._update_data_and_Stokes(corrected_data)
        return self

    # ----------------------------------------------------------------
    # ------------------------------ SHOW ----------------------------
    # ----------------------------------------------------------------

    def show_with_pol_params(self, cmap="Greys_r") -> tuple:
        """Returns a tuple containing figure and axis of the plotted
        data, and figure and axis of polarization parameters (3x2
        subplots). User must callplt.show after this is called.

        Args:
            cmap (str, optional): colormap string. Defaults to "Greys_r".

        Returns:
            tuple: a (figure, axis, figure, axis) couple same as
            matplotlib.pyplot.subplots for the image data and another for
            the six polarization parameters
        """
        data_ratio = self.data.shape[0] / self.data.shape[1]
        image_fig, imageax = plt.subplots(dpi=200, constrained_layout=True)

        avg = np.mean(self.data)
        stdev = np.std(self.data)
        mappable = imageax.imshow(
            self.data,
            cmap=cmap,
            vmin=avg - stdev,
            vmax=avg + stdev,
        )
        if avg < 1.0e5:
            format = "3.2f"
        else:
            format = ".1e"
        imageax.set_title(
            f"Image data (avrg {avg:{format}}+-{stdev:{format}})",
            color="black",
        )
        imageax.set_xlabel("x [px]")
        imageax.set_ylabel("y [px]")
        image_fig.colorbar(
            mappable, ax=imageax, label="[DN]", fraction=data_ratio * 0.05
        )
        stokes_fig, stokesax = self.show_pol_params(cmap=cmap)

        return image_fig, imageax, stokes_fig, stokesax

    def show_pol_params(self, cmap="Greys_r", figsize=None, **kwargs) -> tuple:
        """Returns a tuple containing figure and axis of polarization parameters (3x2 subplots). User must call plt.show after this is called.

        Args:
            cmap (str, optional): colormap string. Defaults to "Greys_r".

        Returns:
            tuple: a (figure, axis) couple same as
            matplotlib.pyplot.subplots for the six polarization parameters
        """
        data_ratio = self.data.shape[0] / self.data.shape[1]
        if figsize is None:
            figsize = (14, 6)
        stokes_fig, stokesax = plt.subplots(
            2, 3, figsize=figsize, constrained_layout=True, **kwargs
        )
        stokesax = stokesax.ravel()
        polparam_list = [self.I, self.Q, self.U, self.pB, self.AoLP, self.DoLP]
        for parameter, axis in zip(polparam_list, stokesax):
            avg = np.mean(parameter.data)
            stdev = np.std(parameter.data)
            mappable_stokes = axis.imshow(
                parameter.data,
                cmap=cmap,
                vmin=avg - stdev,
                vmax=avg + stdev,
            )
            if avg < 1.0e5:
                format = "3.2f"
            else:
                format = ".1e"
            axis.set_title(
                parameter.title + f" (avrg {avg:{format}}+-{stdev:{format}})",
                color="black",
            )
            axis.set_xlabel("x [px]")
            axis.set_ylabel("y [px]")
            stokes_fig.colorbar(
                mappable_stokes,
                ax=axis,
                label=parameter.measure_unit,
                fraction=data_ratio * 0.05,
            )

        return stokes_fig, stokesax

    def show_single_pol_images(self, cmap="Greys_r", **kwargs):
        """Plots the four polarizations images.

        Args:
            cmap (str, optional): colormap for the plot. Defaults to "Greys_r".
            **kwargs: arguments passed to matplotlib.pyplot.imshow.

        Returns:
            tuple: a (figure, axis) couple same as matplotlib.pyplot.subplots
        """
        data_ratio = self.data.shape[0] / self.data.shape[1]
        fig, ax = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
        ax = ax.ravel()
        # polslist = [self.pol0, self.pol45, self.pol90, self.pol_45]
        polslist = self._get_single_pols_as_polparam_list()
        for pol, axis in zip(polslist, ax):
            mappable = axis.imshow(pol.data, cmap=cmap, **kwargs)
            axis.set_title(pol.title)
            axis.set_xlabel("x [px]")
            axis.set_ylabel("y [px]")
            fig.colorbar(
                mappable,
                ax=axis,
                label=pol.measure_unit,
                fraction=data_ratio * 0.05,
                pad=0.01,
            )
        return fig, ax

    def show_demo_images(self, cmap="Greys_r", vmin=None, vmax=None, **kwargs):
        """Plots the four demosaiced images.

        Args:
            cmap (str, optional): colormap for the plot. Defaults to "Greys_r".
            **kwargs: arguments passed to matplotlib.pyplot.imshow.

        Returns:
            tuple: a (figure, axis) couple same as matplotlib.pyplot.subplots
        """
        if not self._is_demosaiced:
            error("Image is not yet demosaiced.")
        data_ratio = self.data.shape[0] / self.data.shape[1]
        fig, ax = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
        ax = ax.ravel()
        demo_images_list = self.demosaiced_images
        for i, single_demo_ax in enumerate(ax):
            if vmin is None:
                this_vmin = mean_minus_std(demo_images_list[i])
            if vmax is None:
                this_vmax = mean_plus_std(demo_images_list[i])
            mappable = single_demo_ax.imshow(
                demo_images_list[i],
                cmap=cmap,
                vmin=this_vmin,
                vmax=this_vmax,
                **kwargs,
            )
            single_demo_ax.set_title(
                f"Demosaiced image {list(self.angle_dic.keys())[list(self.angle_dic.values()).index(i)]}"
            )
            single_demo_ax.set_xlabel("x [px]")
            single_demo_ax.set_ylabel("y [px]")
            fig.colorbar(
                mappable,
                ax=single_demo_ax,
                label="DN",
                fraction=data_ratio * 0.05,
                pad=0.01,
            )
        return fig, ax

    def show_pol_param(
        self, polparam: str, cmap="Greys_r", vmin=None, vmax=None, **kwargs
    ):
        """Plots a single polarization parameter given as input

        Args:
            polparam (str): image PolParam containing the parameter to plot. Can be one among [I, Q, U, pB, AoLP, DoLP]
            cmap (str, optional): colormap for the plot. Defaults to "Greys_r".
            **kwargs: arguments passed to matplotlib.pyplot.imshow.

        Returns:
            tuple: a (figure, axis) couple same as matplotlib.pyplot.subplots
        """
        polparam = getattr(self, polparam)
        data_ratio = self.data.shape[0] / self.data.shape[1]
        fig, ax = plt.subplots(dpi=200)
        if vmin is None:
            vmin = mean_minus_std(polparam.data)
        if vmax is None:
            vmax = mean_plus_std(polparam.data)
        mappable = ax.imshow(
            polparam.data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )
        ax.set_title(polparam.title)
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px]")
        fig.colorbar(
            mappable,
            ax=ax,
            label=polparam.measure_unit,
            fraction=data_ratio * 0.05,
        )
        return fig, ax

    def show_histogram(self, split_pols: bool = True, **kwargs):
        """Print the histogram of the flattened image data

        Args:
            split_pols (bool, optional): Whether to overplot histograms of same family pixels separately. Defaults to False.
            **kwargs (int, optional): arguments to pass to numpy.histogram(), like bins and range.
        Returns:
            tuple: fig, ax tuple as returned by matplotlib.pyplot.subplots
        """

        fig, ax = super().show_histogram(**kwargs)
        if split_pols:
            for i, single_pol_subimage in enumerate(self.single_pol_subimages):
                subhist = np.histogram(single_pol_subimage, **kwargs)
                ax.stairs(*subhist, label=f"pixel {i}")
        return fig, ax

    # ----------------------------------------------------------------
    # -------------------------- SAVING ------------------------------
    # ----------------------------------------------------------------

    def save_single_pol_images(
        self, filename: str, fixto: list[float, float] = None
    ) -> None:
        """Saves the four polarized images as fits files

        Args:
            filename (str): filename of the output image. The four images will be saved as filename_POLXX.fits
            fixto (list[float, float], optional): set the minimum and maximum value for the output images. Defaults to None.

        Raises:
            ValueError: an invalid file name is provided
        """
        # polslist = [self.pol0, self.pol45, self.pol90, self.pol_45]
        polslist = self._get_single_pols_as_polparam_list()
        filepath = Path(_make_abs_and_create_dir(filename))
        if filepath.suffix != ".fits":
            raise ValueError("filename must be a valid file name, not folder.")
        group_filepath = filepath.joinpath(filepath.parent, filepath.stem)
        for single_pol in polslist:
            hdr = self.header.copy()
            hdr["POL"] = (single_pol.ID, "Micropolarizer orientation")
            if fixto:
                data = fix_data(single_pol.data, *fixto)
            else:
                data = single_pol.data
            hdu = fits.PrimaryHDU(
                data=data,
                header=hdr,
                do_not_scale_image_data=True,
                uint=False,
            )
            filename_with_ID = str(
                group_filepath.joinpath(
                    str(group_filepath) + "POL" + str(single_pol.ID) + ".fits"
                )
            )
            hdu.writeto(filename_with_ID, overwrite=True)
        info(f'All params successfully saved to "{filename}"')

    def save_param_as_fits(
        self,
        polparam: str,
        filename: str,
        fixto: list[float, float] = None,
    ) -> None:
        """Saves chosen polarization parameter as a fits file

        Args:
            polparam (str): polarization parameter to save. Can be one among [I, Q, U, pB, AoLP, DoLP]
            filename (str): filename of the output image.
            fixto (list[float, float], optional): set the minimum and maximum value for the output images. Defaults to None.

        Raises:
            ValueError: filename is not a valid .fits file
        """
        polparam = getattr(self, polparam)
        filepath = Path(_make_abs_and_create_dir(filename))
        if filepath.suffix != ".fits":
            raise ValueError("filename must be a valid file name, not folder.")
        hdr = self.header.copy()
        hdr["PARAM"] = (str(polparam.title), "Polarization parameter")
        hdr["UNITS"] = (str(polparam.measure_unit), "Measure units")
        if fixto:
            data = fix_data(polparam.data, *fixto)
        else:
            data = polparam.data
        hdu = fits.PrimaryHDU(
            data=data,
            header=hdr,
            do_not_scale_image_data=True,
            uint=False,
        )
        filename_with_ID = str(
            filepath.joinpath(
                filepath.parent, filepath.stem + "_" + polparam.ID + ".fits"
            )
        )

        # filename = _make_abs_and_create_dir(filename)
        # filename_with_ID = (
        #    filename.split(".")[-2] + "_" + polparam.ID + ".fits"
        # )

        hdu.writeto(filename_with_ID, overwrite=True)
        info(f'"{filename_with_ID}" {polparam.ID} successfully saved')

    def save_all_pol_params_as_fits(self, filename: str) -> None:
        """Saves the image and all polarization parameters as fits file with the same name

        Args:
            filename (str): filename of the output image. Will be saved as filename_[I, Q, U, pB, AoLP, DoLP].fits

        Raises:
            ValueError: filename is not a valid .fits file
        """
        filepath = Path(filename)
        if filepath.suffix != ".fits":
            raise ValueError("filename must be a valid file name, not folder.")
        filepath = Path(_make_abs_and_create_dir(filename))
        group_filename = str(filepath.joinpath(filepath.parent, filepath.stem))
        polparam_list = [self.I, self.Q, self.U, self.pB, self.AoLP, self.DoLP]
        for param in polparam_list:
            hdr = self.header.copy()
            hdr["PARAM"] = (str(param.title), "Polarization parameter")
            hdr["UNITS"] = (str(param.measure_unit), "Measure units")
            if param.fix_data:
                data = fix_data(param.data)
            else:
                data = param.data
            hdu = fits.PrimaryHDU(
                data=data,
                header=hdr,
                do_not_scale_image_data=True,
                uint=False,
            )
            filename_with_ID = group_filename + "_" + param.ID + ".fits"
            hdu.writeto(filename_with_ID, overwrite=True)
        info(f'All params successfully saved to "{group_filename}"')

    def save_demosaiced_images_as_fits(
        self, filename: str, fixto: list[float, float] = None
    ) -> None:
        """Saves the four demosaiced images as fits files

        Args:
            filename (str): filename of the output image. The four images will be saved as filename_POLXX.fits
            fixto (list[float, float], optional): set the minimum and maximum value for the output images. Defaults to None.

        Raises:
            ValueError: an invalid file name is provided
        """
        if not self._is_demosaiced:
            raise ValueError("Demosaiced images not yet calculated.")
        imageHdr = self.header.copy()
        filepath = Path(filename)
        if not filepath.suffix:
            raise ValueError("filename must be a valid file name, not folder.")
        filepath = Path(_make_abs_and_create_dir(filename))
        group_filename = str(filepath.joinpath(filepath.parent, filepath.stem))
        for i, demo_image in enumerate(self.demosaiced_images):
            POL_ID = list(self.angle_dic.keys())[list(self.angle_dic.values()).index(i)]
            imageHdr["POL"] = (int(POL_ID), "Micropolarizer orientation")
            if fixto:
                data = fix_data(demo_image, *fixto)
            else:
                data = demo_image
            hdu = fits.PrimaryHDU(
                data=data,
                header=imageHdr,
                do_not_scale_image_data=True,
                uint=False,
            )
            new_filename = group_filename + "_POL" + str(POL_ID) + ".fits"
            hdu.writeto(new_filename, overwrite=True)
        info(f'Demosaiced images successfully saved to "{group_filename}_POLX.fits"')

    # ----------------------------------------------------------------
    # -------------------- DATA MANIPULATION -------------------------
    # ----------------------------------------------------------------
    def demosaic(self, demosaic_mode="adjacent") -> MicropolImage:
        """Returns a demosaiced copy of the image with updated polarization parameters. Demoisacing is done IN PLACE and
        using the THEORETICAL MATRIX. If demodulation and demosaicing are required, please use demodulate(demosaic=True)

        Args:
            demosaic_mode (str, optional): demosaicing mode (see processing.demosaic). Defaults to "adjacent".

        Returns:
            MicropolImage: demosaiced image
        """

        self.demosaiced_images = demosaic(self.data, option=demosaic_mode)
        self._is_demosaiced = True
        self.Stokes_vec = self._get_theo_Stokes_vec_components()

        return self

    def rebin(self, binning: int) -> MicropolImage:
        """Rebins the micropolarizer array image, binned each
        binningxbinning. Sum bins by default.

        Args:
            binning (int): binning to perform. A value of n will be translated in a nxn binning.

        Raises:
            ValueError: negative binning provided

        Returns:
            MicropolImage: copy of the input image, rebinned.
        """
        if binning <= 0:
            raise ValueError(f"Negative binning {binning}x{binning}")
        rebinned_image = MicropolImage(self)
        rebinned_data = micropolarray_rebin(
            np.array(rebinned_image.data, dtype=float),
            binning,
        )

        """
        new_stokes = []
        for i in range(3):
            new_stokes.append(
                standard_rebin(
                    np.array(self.Stokes_vec[i], dtype=float),
                    binning,
                )
            )
        
        rebinned_image.Stokes_vec = np.array(new_stokes, dtype=float)
        rebinned_image.data = rebinned_data
        """

        rebinned_image._update_data_and_Stokes(rebinned_data)

        return rebinned_image

    def congrid(self, newdim_y: int, newdim_x: int) -> MicropolImage:
        """Reshapes a MicropolImage into any new lenght and width. This is done separately for each pixel family.

        Args:
            newdim_y (int): new height
            newdim_x (int): new width

        Returns:
            MicropolImage: image with reshaped data.
        """
        # Trim to nearest superpixel
        if (newdim_y % 2) or (newdim_x % 2):
            while newdim_y % 2:
                newdim_y = newdim_y - 1
            while newdim_x % 2:
                newdim_x = newdim_x - 1
            warning(
                f"New dimension was incompatible with superpixels. Trimmed to ({newdim_y}, {newdim_x})"
            )
        new_subdims = [int(newdim_y / 2), int(newdim_x / 2)]
        congridded_pol_images = np.zeros(shape=(4, *new_subdims), dtype=float)
        for subimage_i, pol_subimage in enumerate(self.single_pol_subimages):
            congridded_pol_images[subimage_i] = congrid(pol_subimage, new_subdims)
        newdata = merge_polarizations(congridded_pol_images)
        newimage = MicropolImage(self)
        newimage.data = newdata
        newimage.Stokes_vec = [
            congrid(stokes_component, [newdim_y, newdim_x])
            for stokes_component in self.Stokes_vec
        ]
        return newimage

    def rotate(self, angle: float) -> MicropolImage:
        """Rotates an image of angle degrees, counter-clockwise."""

        single_pols = self.single_pol_subimages
        for i in range(4):
            image = PILImage.fromarray(single_pols[i])
            image = image.rotate(angle)
            single_pols[i] = np.asarray(image, dtype=float)
        data = merge_polarizations(single_pols)

        Stokes_vec = self.Stokes_vec
        for i in range(3):
            image = PILImage.fromarray(Stokes_vec[i])
            image = image.rotate(angle)
            Stokes_vec[i] = np.asarray(image, dtype=float)
        newimage = MicropolImage(self)
        newimage.data = data
        newimage.Stokes_vec = Stokes_vec

        return newimage

    def mask_occulter(
        self,
        y: int = PolarCam().occulter_pos_last[0],
        x: int = PolarCam().occulter_pos_last[1],
        r: int = PolarCam().occulter_pos_last[2],
        overoccult: int = 0,
        fill: float = 0.0,
    ) -> None:
        """Masks occulter for all image parameters

        Args:
            y (int, optional): Occulter y position. Defaults to PolarCam().occulter_pos_last[0].
            x (int, optional): Occulter x position. Defaults to PolarCam().occulter_pos_last[1].
            r (int, optional): Occulter radius. Defaults to PolarCam().occulter_pos_last[2].
            overoccult (int, optional): Pixels to overoccult. Defaults to 0.
            camera (_type_, optional): Camera image type. Defaults to PolarCam().

        Returns:
            None
        """
        # y, x, r = camera.occulter_pos_last

        r = r + overoccult

        self.data = roi_from_polar(
            self.data, (y, x), [r, 2 * np.max([self.height, self.width])], fill=fill
        )
        if self._is_demosaiced:
            self.demosaiced_images = [
                roi_from_polar(
                    data, (y, x), (r, 2 * np.max([self.height, self.width])), fill=fill
                )
                for data in self.demosaiced_images
            ]

        if not self._is_demosaiced:
            y = int(y / 2)
            x = int(x / 2)
            r = int(r / 2)
        for i in range(3):
            self.Stokes_vec[i] = roi_from_polar(
                self.Stokes_vec[i],
                (y, x),
                [r, 2 * np.max(self.Stokes_vec[i].shape)],
                include_superpixels=False,
                fill=fill,
            )

    def shift(self, y: int, x: int, missing: float = 0) -> MicropolImage:
        """Shifts image by y, x pixels and fills with 0 the remaining space. Positive numbers for up/right shift and negative for down/left shift. Image is split into polarizations, each one is shifted, then they are merged again.

        Args:
            y (int): vertical shift in pix
            x (int): horizontal shift in pix
            missing (float, optional): value used for filling missin values. Defaults to 0.

        Returns:
            MicropolImage: shifted image copied from the original
        """
        # newdata = shift(self.data, y, x)
        newdata = shift_micropol(self.data, y, x, missing)
        newimage = MicropolImage(self)
        newimage._update_data_and_Stokes(newdata)

        return newimage

    def clean_hot_pixels(self, flagged_hot_pix_map: MicropolImage):
        """Returns a copy of the image with gaussian smeared pixels where flagged_hot_pix_map == 1.

        Args:
            flagged_hot_pix_map (MicropolImage): hot pixels map.

        Returns:
            MicropolImage: copy of the original image, gaussian smeared where flagged_hot_pix_map == 1
        """
        subimages = self.single_pol_subimages
        blurred_subimages = np.array(
            [scipy.ndimage.median_filter(subimage, size=2) for subimage in subimages]
        )
        flagged_subimages = flagged_hot_pix_map.single_pol_subimages
        subimages = np.where(flagged_subimages == 1, blurred_subimages, subimages)

        newimage = MicropolImage(self)
        newimage._update_data_and_Stokes(merge_polarizations(subimages))
        return newimage

    # ----------------------------------------------------------------
    # ------------------------ OVERLOADING ---------------------------
    # ----------------------------------------------------------------
    def __add__(self, second) -> MicropolImage:
        if type(self) is type(second):
            newdata = self.data + second.data
            newimage = MicropolImage(self)
            newimage._update_data_and_Stokes(newdata)
        else:
            newdata = self.data + second
            newimage = MicropolImage(newdata, angle_dic=self.angle_dic)

        newimage.header = self.header
        return newimage

        # CHANGED 2024/12/21: keep the header of the first image
        if type(self) is type(second):
            newdata = self.data + second.data
            newimage = MicropolImage(self)
            newimage._update_data_and_Stokes(newdata)
            return newimage
        else:
            newdata = self.data + second
            return MicropolImage(newdata, angle_dic=self.angle_dic)

    def __sub__(self, second) -> MicropolImage:
        if type(self) is type(second):
            newdata = self.data - second.data
            newimage = MicropolImage(self)
            newimage._update_data_and_Stokes(newdata)
        else:
            newdata = self.data - second
            newimage = MicropolImage(newdata, angle_dic=self.angle_dic)

        newimage.header = self.header
        return newimage

        # CHANGED 2024/12/21: keep the header of the first image
        if type(self) is type(second):
            newdata = self.data - second.data
            newimage = MicropolImage(self)
            newimage._update_data_and_Stokes(newdata)
            return newimage
        else:
            newdata = self.data - second
            return MicropolImage(newdata, angle_dic=self.angle_dic)

    def __mul__(self, second) -> MicropolImage:
        if type(self) is type(second):
            newdata = self.data * second.data
            newimage = MicropolImage(self)
            newimage._update_data_and_Stokes(newdata)
        else:
            newdata = self.data * second
            newimage = MicropolImage(newdata, angle_dic=self.angle_dic)

        newimage.header = self.header
        return newimage

        # CHANGED 2024/12/21: keep the header of the first image
        if type(self) is type(second):
            newdata = self.data * second.data
            newimage = MicropolImage(self)
            newimage._update_data_and_Stokes(newdata)
            return newimage
        else:
            newdata = self.data * second
            return MicropolImage(newdata, angle_dic=self.angle_dic)

    def __truediv__(self, second) -> MicropolImage:
        if type(self) is type(second):
            newdata = np.divide(self.data, second.data, where=second.data != 0.0)
            newimage = MicropolImage(self)
            newimage._update_data_and_Stokes(newdata)
        else:
            newdata = np.divide(self.data, second, where=second != 0.0)
            newimage = MicropolImage(newdata, angle_dic=self.angle_dic)

        newimage.header = self.header
        return newimage

        # CHANGED 2024/12/21: keep the header of the first image
        if type(self) is type(second):
            newdata = np.divide(self.data, second.data, where=second.data != 0.0)
            newimage = MicropolImage(self)
            newimage._update_data_and_Stokes(newdata)
            return newimage
        else:
            # newdata = np.where(second != 0, self.data / second, 4096)
            newdata = np.divide(self.data, second, where=second != 0.0)
            return MicropolImage(newdata, angle_dic=self.angle_dic)


# provide shorter aliases
# PolarcamImage = MicropolImage
# MicropolImage = MicropolImage
