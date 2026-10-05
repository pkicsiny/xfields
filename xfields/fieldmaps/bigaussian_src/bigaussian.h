// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2023.                   //
// ########################################### //

#ifndef XFIELDS_BIGAUSSIAN_H
#define XFIELDS_BIGAUSSIAN_H

#if defined(XO_CONTEXT_CPU) || defined(XO_CONTEXT_CL) || defined(XO_CONTEXT_CUDA)
    #include "xobjects/headers/common.h"
    #include "xfields/fieldmaps/bigaussian_src/faddeeva.h"
    #include "xfields/fieldmaps/bigaussian_src/compute_gx_gy.h"
    #include "xfields/headers/constants.h"
#else
    // For quick testing outside of the Xfields package
    #include "constants.h"
    #include "faddeeva.h"
    #include "complex_error_function.h"
    #include "compute_gx_gy.h"
#endif


GPUFUN
void get_charge_density(const double x,
                      const double y,
                      const double sigma_x,
                      const double sigma_y,
                      double* rho)
{

  // this is a PDF
  double factor = 1 / (2*PI*sigma_x*sigma_y);
  double exp_x = exp(-x*x/(2*sigma_x*sigma_x));
  double exp_y = exp(-y*y/(2*sigma_y*sigma_y));
  *rho = factor * exp_x * exp_y;  // [m^-2]
}

GPUFUN
void get_transv_field_gauss_round(
    double sigma, double Delta_x, double Delta_y,
    double x, double y,
    double* Ex,
    double* Ey)
{
  double r2, temp;

  r2 = (x-Delta_x)*(x-Delta_x)+(y-Delta_y)*(y-Delta_y);
  double const inv_sig2 = 1./(sigma*sigma);
  double const r2_sig2 = r2 * inv_sig2;
  if (r2_sig2<1e-10) temp = inv_sig2/(4.*PI*EPSILON_0); //linearised
  else temp = inv_sig2 * (1-exp(-0.5*r2_sig2))/(2.*PI*EPSILON_0*r2_sig2);

  (*Ex) = temp * (x-Delta_x);
  (*Ey) = temp * (y-Delta_y);
}

GPUFUN
void get_transv_field_gauss_ellip(
        double sigma_x,  double sigma_y,
        double Delta_x,  double Delta_y,
        const double x,
	const double y,
        double* Ex_out,
        double* Ey_out)
{
  double sigmax = sigma_x;
  double sigmay = sigma_y;

  // I always go to the first quadrant and then apply the signs a posteriori
  // numerically more stable (see http://inspirehep.net/record/316705/files/slac-pub-5582.pdf)

  double abx = fabs(x - Delta_x);
  double aby = fabs(y - Delta_y);

  double S, factBE, Ex, Ey;
  double etaBE_re, etaBE_im, zetaBE_re, zetaBE_im;
  double w_etaBE_re, w_etaBE_im, w_zetaBE_re, w_zetaBE_im;
  double expBE;


  if (sigmax>sigmay){
    S = sqrt(2.*(sigmax*sigmax-sigmay*sigmay));
    factBE = 1./(2.*EPSILON_0*SQRT_PI*S);

    etaBE_re = sigmay/sigmax*abx;
    etaBE_im = sigmax/sigmay*aby;

    zetaBE_re = abx;
    zetaBE_im = aby;

    //w_zetaBE_re, w_zetaBE_im = wfun(zetaBE_re/S, zetaBE_im/S)
    faddeeva_w(zetaBE_re/S, zetaBE_im/S , &(w_zetaBE_re), &(w_zetaBE_im));

    //w_etaBE_re, w_etaBE_im = wfun(etaBE_re/S, etaBE_im/S)
    faddeeva_w(etaBE_re/S, etaBE_im/S , &(w_etaBE_re), &(w_etaBE_im));

    expBE = exp(-abx*abx/(2*sigmax*sigmax)-aby*aby/(2*sigmay*sigmay));

    Ex = factBE*(w_zetaBE_im - w_etaBE_im*expBE);
    Ey = factBE*(w_zetaBE_re - w_etaBE_re*expBE);

  }
  else if (sigmax<sigmay){
    S = sqrt(2.*(sigmay*sigmay-sigmax*sigmax));
    factBE = 1./(2.*EPSILON_0*SQRT_PI*S);

    etaBE_re = sigmax/sigmay*aby;
    etaBE_im = sigmay/sigmax*abx;

    zetaBE_re = aby;
    zetaBE_im = abx;

    //w_zetaBE_re, w_zetaBE_im = wfun(zetaBE_re/S, zetaBE_im/S)
    faddeeva_w(zetaBE_re/S, zetaBE_im/S , &(w_zetaBE_re), &(w_zetaBE_im));

    //w_etaBE_re, w_etaBE_im = wfun(etaBE_re/S, etaBE_im/S)
    faddeeva_w(etaBE_re/S, etaBE_im/S , &(w_etaBE_re), &(w_etaBE_im));

    expBE = exp(-aby*aby/(2*sigmay*sigmay)-abx*abx/(2*sigmax*sigmax));

    Ey = factBE*(w_zetaBE_im - w_etaBE_im*expBE);
    Ex = factBE*(w_zetaBE_re - w_etaBE_re*expBE);

  }
  else{
    Ex = Ey = 0.;
  }

  if((x - Delta_x)<0) Ex=-Ex;
  if((y - Delta_y)<0) Ey=-Ey;

  (*Ex_out) = Ex;
  (*Ey_out) = Ey;
}

GPUFUN
void get_Ex_Ey_gauss(
             const double  x,
             const double  y,
             const double  sigma_x,
             const double  sigma_y,
             const double  min_sigma_diff,
             double* Ex_ptr,
             double* Ey_ptr){

        // round beam
	if (fabs(sigma_x-sigma_y)< min_sigma_diff){
	    double sigma = 0.5*(sigma_x+sigma_y);
	    	get_transv_field_gauss_round(sigma, 0., 0., x, y, Ex_ptr, Ey_ptr);
	}

        // elliptical beam
	else{
	    get_transv_field_gauss_ellip(
	            sigma_x, sigma_y, 0., 0., x, y, Ex_ptr, Ey_ptr);

	}
}


/* Gauss-Legendre nodes/weights on [-1, 1], n points (equivalent to np.polynomial.legendre.leggauss). */
static void gauss_legendre(int n, double *x, double *w)
{
    for (int i = 0; i < (n + 1) / 2; i++) {
        double z = cos(PI * (i + 0.75) / (n + 0.5));   /* initial guess */
        double pp = 0.0;
        for (int it = 0; it < 100; it++) {
            double p1 = 1.0, p2 = 0.0;
            for (int j = 1; j <= n; j++) {               /* recurrence for P_n(z) */
                double p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
            }
            pp = n * (z * p1 - p2) / (z * z - 1.0);      /* P_n'(z) */
            double dz = p1 / pp;
            z -= dz;
            if (fabs(dz) < 1e-15) break;
        }
        x[i]         = -z;                               /* ascending order, like NumPy */
        x[n - 1 - i] =  z;
        w[i] = w[n - 1 - i] = 2.0 / ((1.0 - z * z) * pp * pp);
    }
}

#define GL_N  128
#define GL_C  20.0
int GL_SWITCH=1;

/* Global tables, same names as your Python constants */
static double U2[GL_N];
static double ONE_MINUS_U2[GL_N];
static double HALF_U2[GL_N];
static double U_W[GL_N];

/* Call once at startup, before any threads are launched. */
void gl_tables_init(void)
{

    double t_nodes[GL_N], t_weights[GL_N];   /* 2 KB on the stack, fine */
    gauss_legendre(GL_N, t_nodes, t_weights);

    const double c = GL_C;
    const double tanh_half_c = (c != 0.0) ? tanh(0.5 * c) : 1.0;
    const double pref        = (c != 0.0) ? 0.5 * c / tanh_half_c : 1.0;

    for (int i = 0; i < GL_N; i++) {
        const double v  = 0.5 * (t_nodes[i] + 1.0);
        const double vw = 0.5 * t_weights[i];

        double u, uw;
        if (c == 0.0) {
            u = v;  uw = vw;
        } else {
            const double arg  = c * (v - 0.5);
            const double sech = 1.0 / cosh(arg);
            u  = 0.5 * (1.0 + tanh(arg) / tanh_half_c);
            uw = vw * pref * sech * sech;
        }

        const double u2 = u * u;
        U2[i]           = u2;
        ONE_MINUS_U2[i] = 1.0 - u2;
        HALF_U2[i]      = 0.5 * u2;
        U_W[i]          = u * uw;
    }
}

/*
GPUFUN
void gl_prepare(
    const double sigma_x, 
    const double sigma_y,
    double* A, double* B, double* Wx, double* Wy, double* pref,
){

    // at first call compute gl nodes and weights
    if (GL_SWITCH){
        gl_tables_init();
        GL_SWITCH = 0;
    }


    const double inv_sx2 = 1.0/(sigma_x*sigma_x);
    (*pref) = TWO_PI_EPS0_INV * inv_sx2;
    const double r = sigma_y*sigma_y*inv_sx2;

    for (int k = 0; k < GL_N; k++) {
        const double inv_d = 1.0/(ONE_MINUS_U2[k] + U2[k]*r);
        A[k]  = HALF_U2[k]*inv_sx2;
        B[k]  = A[k]*inv_d;
        Wx[k] = U_W[k]*sqrt(inv_d);
        Wy[k] = Wx[k]*inv_d;
    }
}

GPUFUN
void get_Ex_Ey_gauss_gl(
             const double  x,
             const double  y,
             const double  sigma_x,
             const double* A, const double* B, const double* Wx, const double* Wy, const double pref,
             double* Ex,
             double* Ey){

    const double x2 = x*x, y2 = y*y;
    double ix_sum = 0.0, iy_sum = 0.0;

    for (int k = 0; k < GL_N; k++) {
        const double e = exp(-(x2*A[k] + y2*B[k]));
        ix_sum += Wx[k]*e;
        iy_sum += Wy[k]*e;
    }

    (*Ex) = pref * x * ix_sum;
    (*Ey) = pref * y * iy_sum;
}
*/

GPUFUN
void get_Ex_Ey_gauss_gl_naive(
             const double  x,
             const double  y,
             const double  sigma_x,
             const double  sigma_y,
             double* Ex,
             double* Ey){

//    FILE *fptr3 = fopen("/home/pkicsiny/pkicsiny/projects/autodiff/notebooks_pc101697/test_gl.txt", "a");
//    fprintf(fptr3, "GL_SWITCH: %d\n", GL_SWITCH);
//    fclose(fptr3);



    // at first call compute gl nodes and weights
    if (GL_SWITCH){
        gl_tables_init();
        GL_SWITCH = 0;
    }

    const double inv_sx2 = 1.0 / (sigma_x * sigma_x);
    const double r       = sigma_y * sigma_y * inv_sx2;
    const double x2s = x * x * inv_sx2;
    const double y2s = y * y * inv_sx2;

    double ix_sum = 0.0, iy_sum = 0.0;
    for (int k = 0; k < GL_N; k++) {
        const double inv_d = 1.0 / (ONE_MINUS_U2[k] + U2[k] * r);
        const double e     = exp(-HALF_U2[k] * (x2s + y2s * inv_d));
        const double ix    = U_W[k] * sqrt(inv_d);
        ix_sum += ix * e;
        iy_sum += ix * inv_d * e;

//        FILE *fptr3 = fopen("/Users/pkicsiny/work/projects/autodiff/notebooks/test_gl.txt", "a");
//        fprintf(fptr3, "%d, %g\n", GL_SWITCH, U_W[k]);
//        fclose(fptr3);

    }

    const double pref = TWO_PI_EPS0_INV * inv_sx2;

    (*Ex) = pref * x * ix_sum;
    (*Ey) = pref * y * iy_sum;
}


#endif // XFIELDS_BIGAUSSIAN_H
