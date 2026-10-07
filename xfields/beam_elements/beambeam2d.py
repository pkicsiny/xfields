# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np

import xobjects as xo
import xtrack as xt


class BeamBeamBiGaussian2D(xt.BeamElement):

    _xofields = {

        'scale_strength': xo.Float64,

        'ref_shift_x': xo.Float64,
        'ref_shift_y': xo.Float64,

        'other_beam_shift_x': xo.Float64,
        'other_beam_shift_y': xo.Float64,

        'post_subtract_px': xo.Float64,
        'post_subtract_py': xo.Float64,

        # TODO this could become other_beam_q0, other_beam_beta0 (to be done also in 6D)
        'other_beam_q0': xo.Float64,
        'other_beam_beta0': xo.Float64,

        'other_beam_num_particles': xo.Float64,

        'other_beam_Sigma_11': xo.Float64,
        'other_beam_Sigma_13': xo.Float64,
        'other_beam_Sigma_33': xo.Float64,

        'min_sigma_diff': xo.Float64,

        'use_gl':        xo.Int32,
        'num_gl_points': xo.Int32,
        'U2':            xo.Float64[:],
        'ONE_MINUS_U2':  xo.Float64[:],
        'HALF_U2':       xo.Float64[:],
        'U_W':           xo.Float64[:], 
        'gl_A':          xo.Float64[:],
        'gl_B':          xo.Float64[:],
        'gl_Wx':         xo.Float64[:],
        'gl_Wy':         xo.Float64[:],
        'pref':          xo.Float64,
    }

    _extra_c_sources= [
       '#include "xfields/beam_elements/beambeam_src/beambeam2d.h"',
    ]

    def __init__(self,
                    scale_strength=1.,

                    other_beam_q0=0,
                    other_beam_beta0=1,

                    other_beam_num_particles=0,

                    other_beam_Sigma_11=1,
                    other_beam_Sigma_13=0,
                    other_beam_Sigma_33=1,

                    ref_shift_x=0,
                    ref_shift_y=0,

                    other_beam_shift_x=0,
                    other_beam_shift_y=0,

                    post_subtract_px=0,
                    post_subtract_py=0,

                    min_sigma_diff=1e-10,

                    use_gl=0,
                    num_gl_points=128,
                    tanh_cc=20,

                    config_for_update=None,

                    **kwargs):

        """
        2D beam-beam element in the soft-Gaussian approximation.

        Args:
             scale_strength (float): Used to scale beam-beam force strength. Scales `other_beam_q0`.
             other_beam_q0 (float): Charge sign of opposing beam. -1 for electrons, +1 for protons or positrons.
             other_beam_beta0 (float): Relativistic beta of the opposing beam.
             other_beam_num_particles (float): Number of real charges in the opposing bunch. 
             other_beam_Sigma_{13}{13} (float): Statistical moments (variances and covariance of x (=1) and y (=3)) of the opposing bunch, in the unboosted accelerator frame. 
             ref_shift_{xy} (float): Closed orbit shift, subtracted from each macroparticle before collision and added back after.
             other_beam_shift_{x,y} (float): Closed orbit shift of the opposing beam. Subtracted and from the opposing slice centroids before collision and added back after.
             post_subtract_{px,py} (float): Additional quantity that is subtracted after the beam-beam collision. Used e.g. for dipole kick.
             min_sigma_diff (float): Round beam kick (~2x faster) is used instead of elliptical kick, if `fabs(sigma_x-sigma_y) < min_sigma_diff`.
             config_for_update (xfields.ConfigForUpdateBeamBeamBiGaussian3D): Used for (quasi-)strong-strong beam-beam and `None` for weak-strong. See documentation of xfields.ConfigForUpdateBeamBeamBiGaussian3D.
        """


        if '_xobject' in kwargs.keys():
            self.xoinitialize(**kwargs)
            return

        # Collective mode (pipeline update)
        if config_for_update is not None:

            self.config_for_update = config_for_update
            self.iscollective = True
            self.track_WS = self.track # weak-strong tracking once the properties of the element have been updated
            self.track = self._track_collective # switch to specific track method

            self.moments = None
            self.partner_moments = np.zeros((1+2+3), dtype=float)
            other_beam_num_particles = 0.0
            other_beam_Sigma_11 = 0.0
            other_beam_Sigma_13 = 0.0
            other_beam_Sigma_33 = 0.0
        else:
            self.config_for_update = None

        params = self._handle_init_old_interface(kwargs)

        # GL
        self._allocate_xobject(num_gl_points, **kwargs)
        #self.xoinitialize(**kwargs)

        if self.iscollective:
            if not isinstance(self._buffer.context, xo.ContextCpu):
                raise NotImplementedError(
                    'BeamBeamBiGaussian3D only works with CPU context for now')

        # Handle old interface
        if 'other_beam_num_particles' in params.keys(): other_beam_num_particles = params['other_beam_num_particles']
        if 'other_beam_q0' in params.keys(): other_beam_q0 = params['other_beam_q0']
        if 'other_beam_beta0' in params.keys(): other_beam_beta0 = params['other_beam_beta0']
        if 'other_beam_shift_x' in params.keys(): other_beam_shift_x = params['other_beam_shift_x']
        if 'other_beam_shift_y' in params.keys(): other_beam_shift_y = params['other_beam_shift_y']
        if 'other_beam_Sigma_11' in params.keys(): other_beam_Sigma_11 = params['other_beam_Sigma_11']
        if 'other_beam_Sigma_33' in params.keys(): other_beam_Sigma_33 = params['other_beam_Sigma_33']
        if 'post_subtract_px' in params.keys(): post_subtract_px = params['post_subtract_px']
        if 'post_subtract_py' in params.keys(): post_subtract_py = params['post_subtract_py']

        # Coupling between transverse planes
        if other_beam_Sigma_13 is None:
            other_beam_Sigma_13 = 0

        if np.abs(other_beam_Sigma_13) > 0:
            raise NotImplementedError("Coupled case not tested yet.")

        self.other_beam_num_particles = other_beam_num_particles

        self.other_beam_Sigma_11 = other_beam_Sigma_11
        self.other_beam_Sigma_13 = other_beam_Sigma_13
        self.other_beam_Sigma_33 = other_beam_Sigma_33

        self.other_beam_q0 = other_beam_q0

        self.other_beam_beta0 = other_beam_beta0

        self.ref_shift_x = ref_shift_x
        self.ref_shift_y = ref_shift_y

        self.other_beam_shift_x = other_beam_shift_x
        self.other_beam_shift_y = other_beam_shift_y

        self.post_subtract_px = post_subtract_px
        self.post_subtract_py = post_subtract_py

        self.min_sigma_diff = min_sigma_diff

        self.scale_strength = scale_strength

        #GL
        self.use_gl = use_gl
        self.init_gl_tables(num_gl_points, tanh_cc)
        self.num_gl_points = num_gl_points

    def _handle_init_old_interface(self, kwargs):

        params = {}

        if 'n_particles' in kwargs.keys():
            params['other_beam_num_particles'] = kwargs['n_particles']
            del kwargs['n_particles']

        if 'q0' in kwargs.keys():
            params['other_beam_q0'] = kwargs['q0']
            del kwargs['q0']

        if 'beta0' in kwargs.keys():
            params['other_beam_beta0'] = kwargs['beta0']
            del kwargs['beta0']

        if 'mean_x' in kwargs.keys():
            params['other_beam_shift_x'] = kwargs['mean_x']
            del kwargs['mean_x']

        if 'mean_y' in kwargs.keys():
            params['other_beam_shift_y'] = kwargs['mean_y']
            del kwargs['mean_y']

        if 'sigma_x' in kwargs.keys():
            params['other_beam_Sigma_11'] = kwargs['sigma_x']**2
            del kwargs['sigma_x']

        if 'sigma_y' in kwargs.keys():
            params['other_beam_Sigma_33'] = kwargs['sigma_y']**2
            del kwargs['sigma_y']

        if 'd_px' in kwargs.keys():
            params['post_subtract_px'] = kwargs['d_px']
            del kwargs['d_px']

        if 'd_py' in kwargs.keys():
            params['post_subtract_py'] = kwargs['d_py']
            del kwargs['d_py']

        return params

    def _allocate_xobject(self, num_gl_points, **kwargs):
        self.xoinitialize(
            U2=num_gl_points,
            ONE_MINUS_U2=num_gl_points,
            HALF_U2=num_gl_points,
            U_W=num_gl_points,
            gl_A=num_gl_points,
            gl_B=num_gl_points,
            gl_Wx=num_gl_points,
            gl_Wy=num_gl_points,
            **kwargs
            )


    def get_gl_tanh(self, N=128, c=20):
        t_nodes, t_weights = np.polynomial.legendre.leggauss(N)
        v_nodes = 0.5 * (t_nodes + 1.0)
        v_weights = 0.5 * t_weights
        if c == 0:
            return v_nodes, v_weights
        tanh_half_c = np.tanh(0.5 * c)
        u_nodes = 0.5 * (1.0 + np.tanh(c * (v_nodes - 0.5)) / tanh_half_c)
        phi_prime = (
            (0.5 * c / tanh_half_c) * (1.0 / np.cosh(c * (v_nodes - 0.5))) ** 2
        )
        u_weights = v_weights * phi_prime
        return u_nodes, u_weights

    def init_gl_tables(self, num_points, cc):
        if self.use_gl==1:
            u_nodes, u_weights = self.get_gl_tanh(num_points, c=cc)
            self.U2           = self._arr2ctx(u_nodes * u_nodes)
            self.ONE_MINUS_U2 = self._arr2ctx(1.0 - self.U2)
            self.HALF_U2      = self._arr2ctx(0.5 * self.U2)
            self.U_W          = self._arr2ctx(u_nodes * u_weights)
            self.refresh_gl_tables()
        else:
            self.U2          
            self.ONE_MINUS_U2[:] = 0
            self.HALF_U2     [:] = 0    
            self.U_W         [:] = 0         
            self.gl_A        [:] = 0
            self.gl_B        [:] = 0
            self.gl_Wx       [:] = 0
            self.gl_Wy       [:] = 0
            self.pref            = 0

    def refresh_gl_tables(self):
        sx2 = self.other_beam_Sigma_11
        sy2 = self.other_beam_Sigma_33
        inv_sx2 = 1.0 / sx2
        r = sy2 * inv_sx2
        inv_d = 1.0 / (self.ONE_MINUS_U2 + self.U2 * r)   # numpy copies of your GL tables
        A = self.HALF_U2 * inv_sx2
        self.gl_A  = self._arr2ctx(A)
        self.gl_B  = self._arr2ctx(A * inv_d)
        self.gl_Wx = self._arr2ctx(self.U_W * np.sqrt(inv_d))
        self.gl_Wy = self._arr2ctx(self.gl_Wx * inv_d)
        self.pref  = 17975103572.341595 * inv_sx2;


    def _track_collective(self, particles, _force_suspend=False):
        if self.config_for_update._working_on_bunch is None:
            # I am working on a new bunch

            if particles._num_active_particles == 0:
                return # All particles are lost

            # Check that the element is not occupied by a bunch
            assert self.config_for_update._working_on_bunch is None

            self.config_for_update._working_on_bunch = particles.name

            # Handle update frequency
            at_turn = particles._xobject.at_turn[0] # On CPU there is always an active particle in position 0
            if (self.config_for_update.update_every is not None
                    and at_turn % self.config_for_update.update_every == 0):
                self.config_for_update._do_update = True
            else:
                self.config_for_update._do_update = False

            # Can be used to test the resume without pipeline
            if _force_suspend:
                return xt.PipelineStatus(on_hold=True)

        assert self.config_for_update._working_on_bunch == particles.name

        ret = self._apply_bb_kicks(particles)

        return ret

        # Beam beam interaction in the boosted frame
        ret = self._apply_bb_kicks(particles)

        return ret

    def _apply_bb_kicks(self, particles):
        if self.config_for_update._do_update:
            if self.config_for_update.pipeline_manager.is_ready_to_send(self.config_for_update.element_name,
                                                 particles.name,
                                                 self.config_for_update.partner_particles_name,
                                                 particles.at_turn[0],
                                                 internal_tag=0):
                # Compute moments
                self.moments = self.compute_spacial_moments(particles)
                self.config_for_update.pipeline_manager.send_message(self.moments,
                                                 self.config_for_update.element_name,
                                                 particles.name,
                                                 self.config_for_update.partner_particles_name,
                                                 particles.at_turn[0],
                                                 internal_tag=0)

            if self.config_for_update.pipeline_manager.is_ready_to_receive(self.config_for_update.element_name,
                                    self.config_for_update.partner_particles_name,
                                    particles.name,
                                    internal_tag=0):
                self.config_for_update.pipeline_manager.receive_message(self.partner_moments,
                                    self.config_for_update.element_name,
                                    self.config_for_update.partner_particles_name,
                                    particles.name,
                                    internal_tag=0)
                self.update_from_received_moments()
            else:
                return xt.PipelineStatus(on_hold=True)

        self.track_WS(particles)

        self.config_for_update._working_on_bunch = None

        return None

    def compute_spacial_moments(self,particles):
        nplike_lib = self._buffer.context.nplike_lib
        moments = np.zeros((1+2+3), dtype=float)
        moments[0] = nplike_lib.sum(particles.weight)
        moments[1] = nplike_lib.sum(particles.x*particles.weight) / moments[0]
        moments[2] = nplike_lib.sum(particles.y*particles.weight) / moments[0]
        x_diff = particles.x-moments[1]
        y_diff = particles.y-moments[2]
        moments[3] = nplike_lib.sum(x_diff**2*particles.weight) / moments[0]
        moments[4] = nplike_lib.sum(x_diff*y_diff*particles.weight) / moments[0]
        moments[5] = nplike_lib.sum(y_diff**2*particles.weight) / moments[0]
        return moments

    def update_from_received_moments(self):
        # reference frame transformation as in https://github.com/lhcopt/lhcmask/blob/865eaf9d7b9b888c6486de00214c0c24ac93cfd3/pymask/beambeam.py#L310
        self.other_beam_num_particles = self.partner_moments[0]

        self.other_beam_shift_x = self.partner_moments[1] * (-1.0)
        self.other_beam_shift_y = self.partner_moments[2]

        self.other_beam_Sigma_11 = self.partner_moments[3]
        self.other_beam_Sigma_13 = self.partner_moments[4]
        self.other_beam_Sigma_33 = self.partner_moments[5]
        

    # Properties to mimic the old interfece (to be removed)
    @property
    def n_particles(self):
        return self.other_beam_num_particles

    @n_particles.setter
    def n_particles(self, value):
        self.other_beam_num_particles = value

    @property
    def q0(self):
        return self.other_beam_q0

    @q0.setter
    def q0(self, value):
        self.other_beam_q0 = value

    @property
    def beta0(self):
        return self.other_beam_beta0

    @beta0.setter
    def beta0(self, value):
        self.other_beam_beta0 = value

    @property
    def mean_x(self):
        return self.other_beam_shift_x

    @mean_x.setter
    def mean_x(self, value):
        self.other_beam_shift_x = value

    @property
    def mean_y(self):
        return self.other_beam_shift_y

    @mean_y.setter
    def mean_y(self, value):
        self.other_beam_shift_y = value

    @property
    def sigma_x(self):
        return np.sqrt(self.other_beam_Sigma_11)

    @sigma_x.setter
    def sigma_x(self, value):
        self.other_beam_Sigma_11 = value**2

    @property
    def sigma_y(self):
        return np.sqrt(self.other_beam_Sigma_33)

    @sigma_y.setter
    def sigma_y(self, value):
        self.other_beam_Sigma_33 = value**2

    @property
    def d_px(self):
        return self.post_subtract_px

    @d_px.setter
    def d_px(self, value):
        self.post_subtract_px = value

    @property
    def d_py(self):
        return self.post_subtract_py

    @d_py.setter
    def d_py(self, value):
        self.post_subtract_py = value


class ConfigForUpdateBeamBeamBiGaussian2D:

    def __init__(self,
        pipeline_manager=None,
        element_name=None,
        partner_particles_name=None,
        update_every=None):

        self.pipeline_manager = pipeline_manager
        self.element_name = element_name
        self.partner_particles_name = partner_particles_name
        self.update_every = update_every

        self._working_on_bunch = None



