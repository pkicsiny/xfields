from scipy.constants import e as qe
import xobjects as xo
import xtrack as xt

from ..general import _pkg_root

"""
18/08/21: add slicing by index
"""

class CPToIP3D(xt.BeamElement):

    _xofields = {
        'x_centroid': xo.Float64,
        'px_centroid': xo.Float64,
        'y_centroid': xo.Float64,
        'py_centroid': xo.Float64,
        'z_centroid': xo.Float64,
        'is_sliced': xo.Int64,
        'slice_id': xo.Int64,
    }

    def update(self, **kwargs):
        for kk in kwargs.keys():
            if not hasattr(self, kk):
                raise NameError(f'Unknown parameter: {kk}')
            setattr(self, kk, kwargs[kk])


srcs = []
srcs.append(_pkg_root.joinpath('headers/constants.h'))
srcs.append('#define NOFIELDMAP') #TODO Remove this workaound
srcs.append(_pkg_root.joinpath('beam_elements/beambeam_src/cptoip3d.h'))

CPToIP3D.XoStruct.extra_sources = srcs
