from typing import Tuple

import numpy as np

from scipy.constants import c as clight
from scipy.constants import e as qe
from scipy.interpolate import interp1d

import xobjects as xo
import xfields as xf
from ..element_with_slicer import ElementWithSlicer
from .convolution import _ConvData


class WakeTracker(ElementWithSlicer):
    """
    An object handling many WakeField instances as a single beam element.

    Parameters
    ----------
    components : xfields.WakeField
        List of wake fields.
    zeta_range : Tuple
        Zeta range for each bunch used in the underlying slicer.
    num_slices : int
        Number of slices per bunch used in the underlying slicer.
    bunch_spacing_zeta : float
        Bunch spacing in meters.
    filling_scheme: np.ndarray
        List of zeros and ones representing the filling scheme. The length
        of the array is equal to the number of slots in the machine and each
        element of the array holds a one if the slot is filled or a zero
        otherwise.
    bunch_numbers: np.ndarray
        List of the bunches indicating which slots from the filling scheme are
        used (not all the bunches are used when using multi-processing)
    num_turns : int
        Number of turns which are consiered for the multi-turn wake.
    circumference: float
        Machine length in meters.
    log_moments: list
        List of moments logged in the slicer.
    _flatten: bool
        Use flattened wakes
    """

    def __init__(self, components,
                 zeta_range=None,  # These are [a, b] in the paper
                 num_slices=None,  # Per bunch, this is N_1 in the paper
                 bunch_spacing_zeta=None,  # This is P in the paper
                 filling_scheme=None,
                 bunch_numbers=None,
                 num_turns=1,
                 circumference=None,
                 log_moments=None,
                 _flatten=False):

        self.components = components
        self.pipeline_manager = None

        all_slicer_moments = []
        for cc in self.components:
            assert not hasattr(cc, 'moments_data') or cc.moments_data is None
            all_slicer_moments += cc.source_moments

        self.all_slicer_moments = list(set(all_slicer_moments))

        super().__init__(
            slicer_moments=all_slicer_moments,
            log_moments=log_moments,
            zeta_range=zeta_range,  # These are [a, b] in the paper
            num_slices=num_slices,  # Per bunch, this is N_1 in the paper
            bunch_spacing_zeta=bunch_spacing_zeta,  # This is P in the paper
            filling_scheme=filling_scheme,
            bunch_numbers=bunch_numbers,
            num_turns=num_turns,
            circumference=circumference,
            _flatten=False,
            with_compressed_profile=True
        )

        self._initialize_moments(
            zeta_range=zeta_range,  # These are [a, b] in the paper
            num_slices=num_slices,  # Per bunch, this is N_1 in the paper
            bunch_spacing_zeta=bunch_spacing_zeta,  # This is P in the paper
            filling_scheme=filling_scheme,
            num_turns=num_turns,
            circumference=circumference)

        self._flatten = _flatten
        all_slicer_moments = list(set(all_slicer_moments))

    def init_pipeline(self, pipeline_manager, element_name, partners_names):

        super().init_pipeline(pipeline_manager=pipeline_manager,
                              element_name=element_name,
                              partners_names=partners_names)

    def track(self, particles):

        # Find first active particle to get beta0
        if particles.state[0] > 0:
            beta0 = particles.beta0[0]
        else:
            i_first = np.where(particles.state > 0)[0][0]
            beta0 = particles.beta0[i_first]

        # Build _conv_data if necessary
        for cc in self.components:
            if (hasattr(cc, '_conv_data') and cc._conv_data is not None
                and cc._conv_data.wakefield is self):
                continue

            cc._conv_data = _ConvData(component=cc, wakefield=self,
                                            _flatten=self._flatten)
            cc._conv_data._initialize_conv_data(_flatten=self._flatten,
                                                moments_data=self.moments_data,
                                                beta0=beta0)

        # Use common slicer from parent class to measure all moments
        super().track(particles)

        for wf in self.components:
            wf._conv_data.track(particles,
                     i_bunch_particles=self.i_bunch_particles,
                     i_slice_particles=self.i_slice_particles,
                     moments_data=self.moments_data)

    @property
    def zeta_range(self):
        return (self.slicer.zeta_centers[0] - self.slicer.dzeta / 2,
                self.slicer.zeta_centers[-1] + self.slicer.dzeta / 2)

    @property
    def num_slices(self):
       return self.slicer.num_slices

    @property
    def bunch_spacing_zeta(self):
        return self.slicer.bunch_spacing_zeta

    @property
    def filling_scheme(self):
        assert len(self.slicer.filled_slots) == 1, (
            'Only single bunch mode is supported for now')
        assert self.slicer.filled_slots[0] == 0, (
            'Only single bunch mode is supported for now')
        return None

    @property
    def bunch_numbers(self):
        return self.slicer.bunch_numbers

    @property
    def num_turns(self):
        return self.moments_data.num_turns

    @property
    def circumference(self):
        return self.moments_data.circumference

    def __add__(self, other):

        if other == 0:
            return self

        assert isinstance(other, WakeTracker)

        new_components = self.components + other.components

        # Check consistency
        xo.assert_allclose(self.zeta_range, other.zeta_range, atol=1e-12, rtol=0)
        xo.assert_allclose(self.num_slices, other.num_slices, atol=0, rtol=0)
        if self.bunch_spacing_zeta is None:
            assert other.bunch_spacing_zeta is None, (
                'Bunch spacing zeta is not consistent')
        else:
            xo.assert_allclose(self.bunch_spacing_zeta, other.bunch_spacing_zeta, atol=1e-12, rtol=0)
        if self.filling_scheme is None:
            assert other.filling_scheme is None, (
                'Filling scheme is not consistent')
        else:
            xo.assert_allclose(self.filling_scheme, other.filling_scheme, atol=0, rtol=0)
        xo.assert_allclose(self.bunch_numbers, other.bunch_numbers, atol=0, rtol=0)
        xo.assert_allclose(self.num_turns, other.num_turns, atol=0, rtol=0)
        xo.assert_allclose(self.circumference, other.circumference, atol=0, rtol=0)

        return WakeTracker(
                 components=new_components,
                 zeta_range=self.zeta_range,
                 num_slices=self.num_slices,
                 bunch_spacing_zeta=self.bunch_spacing_zeta,
                 filling_scheme=self.filling_scheme,
                 bunch_numbers=(self.bunch_numbers if self.filling_scheme else None),
                 num_turns=self.num_turns,
                 circumference=self.circumference
        )

    def __radd__(self, other):
        return self.__add__(other)