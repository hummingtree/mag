#pragma once

#include <iostream>
#include <fstream>
#include <omp.h>

#include <qlat/config.h>
#include <qlat/utils.h>
#include <qlat/mpi.h>
#include <qlat/field.h>
#include <qlat/field-io.h>
#include <qlat/field-comm.h>
#include <qlat/field-rng.h>

#include <timer.h>

#include "field-matrix.h"
#include "field-hmc.h"
#include <cmath>

using namespace cps;
using namespace qlat;
using namespace std;

QLAT_START_NAMESPACE

inline cps::Matrix get_U(Field<cps::Matrix>& fine_gField, const qlat::Coordinate& x, int mu){
	cps::Coordinate y = x; y[mu]++;
	return fine_gField.get_elems(x)[mu] * fine_gField.get_elems(y)[mu];
}

inline cps::Matrix get_Q(Field<cps::Matrix>& fine_gField, const qlat::Coordinate& x, int mu, double rho){
	// assuming properly communicated.
	cps::Matrix stp;
	get_staple_2x1(stp, fine_gField, x, mu);
	cps::Matrix dg; dg.Dagger(get_U(fine_gField, x, mu));
	return rho*dg*stp;
}

inline cps::Matrix compute_Gamma(const cps::Matrix& Q, const cps::Matrix& SigmaP, const cps::Matrix& U){
	
	double c0 = (Q * Q * Q).ReTr() / 3.;
	bool reflect = false;
	if(c0 < 0.){
		c0 = -c0;
		reflect = true;
	}
	double c1 = (Q * Q).ReTr() / 2.;
	double c0m = 2.* pow(c1/3., 1.5);
	double theta = acos(c0/c0m);
	double u = sqrt(c1/3.) * cos(theta/3.);
	double w = sqrt(c1) * sin(theta/3.);

	// qlat::Printf("u=%.12f\tw=%.12f\n", u, w);	

	double xi0;
	double xi1;
	if(w*w < 0.05*0.05){
		xi0 = 1.-w*w/6.*(1.-w*w/20.*(1.-w*w/42.));
		xi1 = -1./2.*(1.-w*w/12.*(1.-w*w/30.)) + 1./6.*(1.-w*w/20.*(1.-w*w/42.));
	}
	else{
		xi0 = sin(w)/w;
		xi1 = cos(w)/(w*w)-sin(w)/(w*w*w);
	}
	qlat::Complex h0 = (u*u-w*w)*expix(2.*u) + expix(-u)*(8.*u*u*cos(w)+i()*2.*u*(3.*u*u+w*w)*xi0);
	qlat::Complex h1 = 2.*u*expix(2.*u) - expix(-u)*(2.*u*cos(w)-i()*(3.*u*u-w*w)*xi0);
	qlat::Complex h2 = expix(2.*u)-expix(-u)*(cos(w)+i()*3.*u*xi0);
	qlat::Complex f0 = h0 / (9.*u*u-w*w);
	qlat::Complex f1 = h1 / (9.*u*u-w*w);
	qlat::Complex f2 = h2 / (9.*u*u-w*w);

	qlat::Complex r01 = 2.*(u+i()*(u*u-w*w))*expix(2.*u)+2.*expix(-u)*(4.*u*(2.-i()*u)*cos(w)+i()*xi0*(9.*u*u+w*w-i()*u*(3.*u*u+w*w)))
	qlat::Complex r01 = 2.*(u+i()*(u*u-w*w))*expix(2.*u)+2.*expix(-u)*(4.*u*(2.-i()*u)*cos(w)+i()*xi0*(9.*u*u+w*w-i()*u*(3.*u*u+w*w)))

	if(reflect){
		f0 = conj(f0);
		f1 = -1. * conj(f1);
		f2 = conj(f2);
	}

	// qlat::Printf("f0=%.12f\tf0=%.12f\n", f0.real(), f0.imag());	
	// qlat::Printf("f1=%.12f\tf1=%.12f\n", f1.real(), f1.imag());	
	// qlat::Printf("f2=%.12f\tf2=%.12f\n", f2.real(), f2.imag());	
	
	cps::Matrix one; one.UnitMatrix();

	return one * f0 + Q * f1 + Q * Q * f2;	

}



// notations follow hep-lat/0311018
inline void get_Lambda_field(Field<cps::Matrix>& Lambda_field,
								const Field<cps::Matrix>& coarse_gField, 
								const Field<cps::Matrix>& fine_gField,
								
){
	assert(is_matching_geo(Lambda_field.geo, coarse_gField.geo));
	


}



QLAT_END_NAMESPACE
