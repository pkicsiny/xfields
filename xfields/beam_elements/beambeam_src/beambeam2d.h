// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2021.                   //
// ########################################### //

#ifndef XFIELDS_BEAMBEAM_H
#define XFIELDS_BEAMBEAM_H

#include "xtrack/headers/track.h"
#include "xfields/fieldmaps/bigaussian_src/bigaussian.h"


#if !defined(mysign)
    #define mysign(a) (((a) >= 0) - ((a) < 0))
#endif


GPUFUN
void BeamBeamBiGaussian2D_track_local_particle(
        BeamBeamBiGaussian2DData el, LocalParticle* part0){

    double const ref_shift_x = BeamBeamBiGaussian2DData_get_ref_shift_x(el);
    double const ref_shift_y = BeamBeamBiGaussian2DData_get_ref_shift_y(el);

    double const other_beam_shift_x = BeamBeamBiGaussian2DData_get_other_beam_shift_x(el);
    double const other_beam_shift_y = BeamBeamBiGaussian2DData_get_other_beam_shift_y(el);

    double const scale_strength = BeamBeamBiGaussian2DData_get_scale_strength(el);
    double const post_subtract_px = scale_strength*BeamBeamBiGaussian2DData_get_post_subtract_px(el);
    double const post_subtract_py = scale_strength*BeamBeamBiGaussian2DData_get_post_subtract_py(el);

    double const other_beam_q0 = scale_strength*BeamBeamBiGaussian2DData_get_other_beam_q0(el);
    double const other_beam_beta0 = BeamBeamBiGaussian2DData_get_other_beam_beta0(el);

    double const other_beam_num_particles = BeamBeamBiGaussian2DData_get_other_beam_num_particles(el);

    double const other_beam_Sigma_11 = BeamBeamBiGaussian2DData_get_other_beam_Sigma_11(el);
    double const other_beam_Sigma_13 = BeamBeamBiGaussian2DData_get_other_beam_Sigma_13(el);
    double const other_beam_Sigma_33 = BeamBeamBiGaussian2DData_get_other_beam_Sigma_33(el);

    double const min_sigma_diff = BeamBeamBiGaussian2DData_get_min_sigma_diff(el);

    int const use_gl = BeamBeamBiGaussian2DData_get_use_gl(el);

    //FILE *fptr3 = fopen("/Users/pkicsiny/work/projects/autodiff/notebooks/test_gl.txt", "a");
    //fprintf(fptr3, "GL=%d\n", use_gl);
    //fclose(fptr3);


    // compute GL sigma dependent stuff before per particle block
    GPUGLMEM double const* gl_A  = BeamBeamBiGaussian2DData_getp1_gl_A(el, 0);
    GPUGLMEM double const* gl_B  = BeamBeamBiGaussian2DData_getp1_gl_B(el, 0);
    GPUGLMEM double const* gl_Wx = BeamBeamBiGaussian2DData_getp1_gl_Wx(el, 0);
    GPUGLMEM double const* gl_Wy = BeamBeamBiGaussian2DData_getp1_gl_Wy(el, 0);
    double const pref   = BeamBeamBiGaussian2DData_get_pref(el);
    int const num_gl_points = BeamBeamBiGaussian2DData_get_num_gl_points(el);


    START_PER_PARTICLE_BLOCK(part0, part);
        double const x = LocalParticle_get_x(part);
        double const y = LocalParticle_get_y(part);
        double const part_q0 = LocalParticle_get_q0(part);
        double const part_mass0 = LocalParticle_get_mass0(part);
        double const part_chi = LocalParticle_get_chi(part);
        double const part_beta0 = LocalParticle_get_beta0(part);
        double const part_gamma0 = LocalParticle_get_gamma0(part);

        double const x_bar = x - ref_shift_x - other_beam_shift_x;
        double const y_bar = y - ref_shift_y - other_beam_shift_y;

        // Move to rotated frame to account for transverse coupling (if needed)
        double x_hat, y_hat, costheta, sintheta, Sig_11_hat, Sig_33_hat;
        if (fabs(other_beam_Sigma_13) > 1e-13) {
            double const R = other_beam_Sigma_11 - other_beam_Sigma_33;
            double const W = other_beam_Sigma_11 + other_beam_Sigma_33;
            double const T = R * R + 4 * other_beam_Sigma_13 * other_beam_Sigma_13;
            double const sqrtT = sqrt(T);
            double const signR = mysign(R);
            double const cos2theta = signR*R/sqrtT;
            costheta = sqrt(0.5*(1.+cos2theta));
            sintheta = signR*mysign(other_beam_Sigma_13)*sqrt(0.5*(1.-cos2theta));
            x_hat = x_bar*costheta +y_bar*sintheta;
            y_hat = -x_bar*sintheta +y_bar*costheta;
            Sig_11_hat = 0.5*(W+signR*sqrtT);
            Sig_33_hat = 0.5*(W-signR*sqrtT);
        }
        else{
            sintheta = 0;
            costheta = 1;
            x_hat = x_bar;
            y_hat = y_bar;
            Sig_11_hat = other_beam_Sigma_11;
            Sig_33_hat = other_beam_Sigma_33;
        }

        // Get transverse fields
        double Ex, Ey; // Ex = -dphi/dx, Ey = -dphi/dy

        if (use_gl==1){
            get_Ex_Ey_gauss_gl(x_hat, y_hat,
            sqrt(Sig_11_hat), gl_A, gl_B, gl_Wx, gl_Wy, pref, num_gl_points,
            &Ex, &Ey);

            //FILE *fptr3 = fopen("/Users/pkicsiny/work/projects/autodiff/notebooks/test_gl.txt", "a");
            //fprintf(fptr3, "GL=%d, Ex=%g", use_gl, Ex);
            //fclose(fptr3);
        }else if (use_gl==2){
            get_Ex_Ey_gauss_gl_naive(x_hat, y_hat,
            sqrt(Sig_11_hat), sqrt(Sig_33_hat),
            &Ex, &Ey);                          

            //FILE *fptr3 = fopen("/home/pkicsiny/pkicsiny/projects/autodiff/notebooks_pc101697/test_gl.txt", "a");
            //fprintf(fptr3, "BE=%d, Ex=%g\n", use_gl, Ex);
            //fclose(fptr3);
        }else{
            get_Ex_Ey_gauss(x_hat, y_hat,
            sqrt(Sig_11_hat), sqrt(Sig_33_hat),
            min_sigma_diff,
            &Ex, &Ey);
                                                                                                        
            //FILE *fptr3 = fopen("/Users/pkicsiny/work/projects/autodiff/notebooks/test_gl.txt", "a");
            //fprintf(fptr3, "BE=%d, Ex=%g", use_gl, Ex);
            //fclose(fptr3);
        }

        const double charge_mass_ratio = part_chi*QELEM*part_q0
                    /(part_mass0*QELEM/(C_LIGHT*C_LIGHT));
        const double factor = (charge_mass_ratio
                    * other_beam_num_particles * other_beam_q0 * QELEM
                    / (part_gamma0*part_beta0*C_LIGHT*C_LIGHT)
                    * (1+other_beam_beta0 * part_beta0)
                    / (other_beam_beta0 + part_beta0));

        double const dpx_hat = factor * Ex;
        double const dpy_hat = factor * Ey;

        double const dpx = dpx_hat*costheta - dpy_hat*sintheta;
        double const dpy = dpx_hat*sintheta + dpy_hat*costheta;


//        FILE *fptr3 = fopen("/Users/pkicsiny/work/projects/autodiff/notebooks/test_gl.txt", "a");
//        fprintf(fptr3, ", part_chi=%g, QELEM=%g, part_q0=%g, part_mass0=%g, C_LIGHT=%g, charge_mass_ratio=%g, other_beam_num_particles=%g, other_beam_q0=%g, part_gamma0=%g, part_beta0=%g, other_beam_beta0=%g, factor=%g, dpx=%g\n", part_chi, QELEM, part_q0, part_mass0, C_LIGHT, charge_mass_ratio, other_beam_num_particles, other_beam_q0, part_gamma0, part_beta0, other_beam_beta0, factor, dpx);
//        fclose(fptr3);

        LocalParticle_add_to_px(part, dpx - post_subtract_px);
        LocalParticle_add_to_py(part, dpy - post_subtract_py);
    END_PER_PARTICLE_BLOCK;
}

#endif
