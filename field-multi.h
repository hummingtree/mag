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
// notations follow hep-lat/0311018
inline cps::Matrix get_U(Field<cps::Matrix>& fine_gField, const qlat::Coordinate& x, int mu){
	qlat::Coordinate y = x; y[mu]++;
	return fine_gField.get_elems(x)[mu] * fine_gField.get_elems(y)[mu];
}

inline cps::Matrix get_Q(Field<cps::Matrix>& fine_gField, const qlat::Coordinate& x, int mu, double rho_){
	// assuming properly communicated.
	cps::Matrix stp = get_staple_rect(fine_gField, x, mu);
	cps::Matrix dg; dg.Dagger(get_U(fine_gField, x, mu));
	cps::Matrix Omega = stp * dg; Omega.TrLessAntiHermMatrix();
	// qlat::Printf("%.12e\n", (Omega*Omega).ReTr());
	return Omega * qlat::Complex(0., -rho_); // minus sign b/c of the unconventional convention in (2) of hep-lat/0311018
}

inline cps::Matrix hermitian_traceless(const cps::Matrix& M){
	cps::Matrix one; one.UnitMatrix();
	cps::Matrix dg; dg.Dagger(M);
	return ((M+dg)-one*((M+dg).Tr()/3.))*0.5;
}

inline cps::Matrix expiQ(const cps::Matrix& Q){
	// 	hep-lat/0311018
	// Assuming Q is hermitian and traceless.
	// static const double c1m = 9. *rho*rho* (69. + 11. * sqrt(33.)) / 32.;

	cps::Matrix one; one.UnitMatrix();
	double c0 = (Q * Q * Q).ReTr() / 3.;
	bool reflect = false;
	if(c0 < 0.){
		c0 = -c0;
		reflect = true;
	}
	double c1 = (Q * Q).ReTr() / 2.;
	if(c1 == 0.) return one; 
	double c0m = 2.* pow(c1/3., 1.5);
	double theta = acos(c0/c0m);
	double u = sqrt(c1/3.) * cos(theta/3.);
	double w = sqrt(c1) * sin(theta/3.);

	double xi0;
	if(w*w < 0.05*0.05) xi0 = 1.-w*w/6.*(1.-w*w/20.*(1.-w*w/42.));
	else xi0 = sin(w)/w;
	qlat::Complex h0 = (u*u-w*w)*expix(2.*u) + expix(-u)*(8.*u*u*cos(w)+i()*2.*u*(3.*u*u+w*w)*xi0);
	qlat::Complex h1 = 2.*u*expix(2.*u) - expix(-u)*(2.*u*cos(w)-i()*(3.*u*u-w*w)*xi0);
	qlat::Complex h2 = expix(2.*u)-expix(-u)*(cos(w)+i()*3.*u*xi0);
	qlat::Complex f0 = h0 / (9.*u*u-w*w);
	qlat::Complex f1 = h1 / (9.*u*u-w*w);
	qlat::Complex f2 = h2 / (9.*u*u-w*w);

	if(reflect){
		f0 = conj(f0);
		f1 = -1. * conj(f1);
		f2 = conj(f2);
	}

	// qlat::Printf("f0=%.12f\tf0=%.12f\n", f0.real(), f0.imag());	
	// qlat::Printf("f1=%.12f\tf1=%.12f\n", f1.real(), f1.imag());	
	// qlat::Printf("f2=%.12f\tf2=%.12f\n", f2.real(), f2.imag());	
	

	return one * f0 + Q * f1 + Q * Q * f2;
}

inline cps::Matrix DexpiQ(const cps::Matrix& Q, const cps::Matrix& DQ){
	cps::Matrix one; one.UnitMatrix();
	double c0 = (Q * Q * Q).ReTr() / 3.;
	bool reflect = false;
	if(c0 < 0.){
		c0 = -c0;
		reflect = true;
	}
	double c1 = (Q * Q).ReTr() / 2.;
	cps::Matrix zero; zero.ZeroMatrix();
	if(c1 == 0) return zero;
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

	qlat::Complex r01 = 2.*( u + i()*(u*u-w*w) ) * expix(2.*u) + 
						2.*expix(-u)*( 
										4.*u*(2.-i()*u)*cos(w) + 
										i()*xi0*( 9.*u*u + w*w - i()*u*(3.*u*u+w*w) )
						);

	qlat::Complex r11 = 2.*( 1.+2.*i()*u ) * expix(2.*u) + 
						expix(-u)*( 
									-2.*(1.-i()*u)*cos(w) + 
									i()*xi0*(6.*u+i()*(w*w-3.*u*u)) 
						);

	qlat::Complex r21 = 2.*i()*expix(2.*u) + 
						i()*expix(-u)*( cos(w)-3.*xi0*(1.-i()*u) );

	qlat::Complex r02 = -2.*expix(2.*u) + 
						2.*i()*u*expix(-u)*( cos(w) + (1.+4.*i()*u)*xi0 + 3.*u*u*xi1 );
	qlat::Complex r12 = -i()*expix(-u) * ( 
											cos(w) 
											+ (1.+2.*i()*u)*xi0 
											- 3.*u*u*xi1 
						);
	qlat::Complex r22 = expix(-u) * (xi0 - 3.*i()*u*xi1);

	qlat::Complex b10 = (2.*u*r01 + (3.*u*u-w*w)*r02 - 2.*(15.*u*u+w*w)*f0) / ( 2.*(9.*u*u-w*w)*(9.*u*u-w*w) );
	qlat::Complex b11 = (2.*u*r11 + (3.*u*u-w*w)*r12 - 2.*(15.*u*u+w*w)*f1) / ( 2.*(9.*u*u-w*w)*(9.*u*u-w*w) );
	qlat::Complex b12 = (2.*u*r21 + (3.*u*u-w*w)*r22 - 2.*(15.*u*u+w*w)*f2) / ( 2.*(9.*u*u-w*w)*(9.*u*u-w*w) );

	qlat::Complex b20 = (r01 - 3.*u*r02 - 24.*u*f0) / ( 2.*(9.*u*u-w*w)*(9.*u*u-w*w) );
	qlat::Complex b21 = (r11 - 3.*u*r12 - 24.*u*f1) / ( 2.*(9.*u*u-w*w)*(9.*u*u-w*w) );
	qlat::Complex b22 = (r21 - 3.*u*r22 - 24.*u*f2) / ( 2.*(9.*u*u-w*w)*(9.*u*u-w*w) );

	if(reflect){
		f0 = conj(f0);
		f1 = -conj(f1);
		f2 = conj(f2);
	
		b10 = conj(b10);
		b11 = -conj(b11);
		b12 = conj(b12);
		b20 = -conj(b20);
		b21 = conj(b21);
		b22 = -conj(b22);
	}

	// qlat::Printf("f0=%.12f\tf0=%.12f\n", f0.real(), f0.imag());	
	// qlat::Printf("f1=%.12f\tf1=%.12f\n", f1.real(), f1.imag());	
	// qlat::Printf("f2=%.12f\tf2=%.12f\n", f2.real(), f2.imag());	
	
	cps::Matrix B1 = one * b10 + Q * b11 + Q * Q * b12;
	cps::Matrix B2 = one * b20 + Q * b21 + Q * Q * b22;

	return B1 * (Q*DQ).Tr() + B2 * (Q*Q*DQ).Tr() + DQ*f1 + DQ*Q*f2 + Q*DQ*f2;
}

inline cps::Matrix compute_Lambda(const cps::Matrix& Q, const cps::Matrix& SigmaP, const cps::Matrix& U){
	cps::Matrix one; one.UnitMatrix();
	double c0 = (Q * Q * Q).ReTr() / 3.;
	bool reflect = false;
	if(c0 < 0.){
		c0 = -c0;
		reflect = true;
	}
	double c1 = (Q * Q).ReTr() / 2.;
	cps::Matrix zero; zero.ZeroMatrix();
	if(c1 == 0) return zero;
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

	qlat::Complex r01 = 2.*( u + i()*(u*u-w*w) ) * expix(2.*u) + 
						2.*expix(-u)*( 
										4.*u*(2.-i()*u)*cos(w) + 
										i()*xi0*( 9.*u*u + w*w - i()*u*(3.*u*u+w*w) )
						);

	qlat::Complex r11 = 2.*( 1.+2.*i()*u ) * expix(2.*u) + 
						expix(-u)*( 
									-2.*(1.-i()*u)*cos(w) + 
									i()*xi0*(6.*u+i()*(w*w-3.*u*u)) 
						);

	qlat::Complex r21 = 2.*i()*expix(2.*u) + 
						i()*expix(-u)*( cos(w)-3.*xi0*(1.-i()*u) );

	qlat::Complex r02 = -2.*expix(2.*u) + 
						2.*i()*u*expix(-u)*( cos(w) + (1.+4.*i()*u)*xi0 + 3.*u*u*xi1 );
	qlat::Complex r12 = -i()*expix(-u) * ( 
											cos(w) 
											+ (1.+2.*i()*u)*xi0 
											- 3.*u*u*xi1 
						);
	qlat::Complex r22 = expix(-u) * (xi0 - 3.*i()*u*xi1);

	qlat::Complex b10 = (2.*u*r01 + (3.*u*u-w*w)*r02 - 2.*(15.*u*u+w*w)*f0) / ( 2.*(9.*u*u-w*w)*(9.*u*u-w*w) );
	qlat::Complex b11 = (2.*u*r11 + (3.*u*u-w*w)*r12 - 2.*(15.*u*u+w*w)*f1) / ( 2.*(9.*u*u-w*w)*(9.*u*u-w*w) );
	qlat::Complex b12 = (2.*u*r21 + (3.*u*u-w*w)*r22 - 2.*(15.*u*u+w*w)*f2) / ( 2.*(9.*u*u-w*w)*(9.*u*u-w*w) );

	qlat::Complex b20 = (r01 - 3.*u*r02 - 24.*u*f0) / ( 2.*(9.*u*u-w*w)*(9.*u*u-w*w) );
	qlat::Complex b21 = (r11 - 3.*u*r12 - 24.*u*f1) / ( 2.*(9.*u*u-w*w)*(9.*u*u-w*w) );
	qlat::Complex b22 = (r21 - 3.*u*r22 - 24.*u*f2) / ( 2.*(9.*u*u-w*w)*(9.*u*u-w*w) );

	if(reflect){
		f0 = conj(f0);
		f1 = -conj(f1);
		f2 = conj(f2);
	
		b10 = conj(b10);
		b11 = -conj(b11);
		b12 = conj(b12);
		b20 = -conj(b20);
		b21 = conj(b21);
		b22 = -conj(b22);
	}

	// qlat::Printf("f0=%.12f\tf0=%.12f\n", f0.real(), f0.imag());	
	// qlat::Printf("f1=%.12f\tf1=%.12f\n", f1.real(), f1.imag());	
	// qlat::Printf("f2=%.12f\tf2=%.12f\n", f2.real(), f2.imag());	
	
	cps::Matrix B1 = one * b10 + Q * b11 + Q * Q * b12;
	cps::Matrix B2 = one * b20 + Q * b21 + Q * Q * b22;

	cps::Matrix Gamma = Q * (SigmaP*B1*U).Tr() + Q * Q * (SigmaP*B2*U).Tr() + U*SigmaP*f1 + Q*U*SigmaP*f2 + U*SigmaP*Q*f2;
	return hermitian_traceless(Gamma);
}

inline double get_kinetic_energy(Field<cps::Matrix>& mField, double m = 1.){

	double localSum = 0.; // local sum of Tr(\pi*\pi^\dagger)
	int numThreads;
	vector<double> ppLocalSum;

#pragma omp parallel
{
	if(omp_get_thread_num() == 0){
		numThreads = omp_get_num_threads();
		ppLocalSum.resize(numThreads);
	}
	double pLocalSum = 0.;
#pragma omp barrier
#pragma omp for
	for(long index = 0; index < mField.geo.local_volume(); index++){
		qlat::Coordinate x = mField.geo.coordinate_from_index(index);
		const qlat::Vector<cps::Matrix> mx = mField.get_elems_const(x);
		for(int mu = 0; mu < DIMN; mu++){
			pLocalSum += (mx[mu] * mx[mu]).ReTr();
	}}
	ppLocalSum[omp_get_thread_num()] = pLocalSum;
}
	for(int i = 0; i < numThreads; i++){
		localSum += ppLocalSum[i];
	}

	double globalSum;
	MPI_Allreduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, get_comm());
	return globalSum / (2. * m);
}

inline double get_eta_energy(Field<cps::Matrix>& FgField, Field<cps::Matrix>& CgField){
	TIMER("get_eta_energy");
	
	double localSum = 0.;
	int numThreads;
	vector<double> ppLocalSum;

#pragma omp parallel
{
	if(omp_get_thread_num() == 0){
		numThreads = omp_get_num_threads();
		ppLocalSum.resize(numThreads);
	}
	double pLocalSum = 0.;
#pragma omp barrier
#pragma omp for
	for(long index = 0; index < CgField.geo.local_volume(); index++){
		qlat::Coordinate x = CgField.geo.coordinate_from_index(index);
		const qlat::Vector<cps::Matrix> mx = CgField.get_elems_const(x);
		for(int mu = 0; mu < DIMN; mu++){
			// The actaul work
			// first compute Q.
			cps::Matrix Q = get_Q(FgField, 2*x, mu, rho);
			cps::Matrix Gb = expiQ(Q)*get_U(FgField, 2*x, mu); 
			cps::Matrix Ucd; Ucd.Dagger(mx[mu]);
	//		cps::Matrix Ucd; Ucd.UnitMatrix();
			pLocalSum += (Ucd*Gb).ReTr()/eta_sqr;
	}}
	ppLocalSum[omp_get_thread_num()] = pLocalSum;
}
	for(int i = 0; i < numThreads; i++){
		localSum += ppLocalSum[i];
	}

	double globalSum;
	MPI_Allreduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, get_comm());
	return -globalSum;
}

inline double get_hamiltonian_multi(
	Field<cps::Matrix>& FgField, 
	Field<cps::Matrix>& FmField, 
	const Arg_chmc& Farg, 
	Chart<cps::Matrix>& Fchart, 
	Field<cps::Matrix>& CgField, 
	Field<cps::Matrix>& CmField, 
	Chart<cps::Matrix>& Cchart, 
	vector<double>& part
){
	TIMER("get_hamiltonian_multi()");
	
	// momentum part
	double kinetic_energy = get_kinetic_energy(FmField) + get_kinetic_energy(CmField);

	// original fine action
	fetch_expanded_chart(FgField, Fchart);
	// fetch_expanded_chart(CgField, Cchart);
	double potential_energy;
	if(Farg.gauge.type == qlat::WILSON){
		potential_energy = -total_plaq(FgField) * Farg.beta / 3.;
	}
	if(Farg.gauge.type == IWASAKI){
		double p1 = -total_plaq(FgField);
		double p2 = -total_rectangular(FgField);
		potential_energy = (p1*(1.-8.*Farg.gauge.c1) + p2*Farg.gauge.c1) * Farg.beta / 3.;
	}

	// eta part
	double eta_energy;
	eta_energy = get_eta_energy(FgField, CgField);

	// summing
	part.resize(3);
	part[0] = kinetic_energy;
	part[1] = potential_energy;
	part[2] = eta_energy;
	qlat::Printf("energy partition: %.12f\t%.12f\t%.12f\n", part[0], part[1], part[2]);
//	return kinetic_energy + potential_energy;
	return kinetic_energy + potential_energy + eta_energy;
}

inline vector<int> count_num_odd(const qlat::Coordinate& x){
	vector<int> rtn(0);
	for(int i = 0; i < DIMN; i++){
		if(x[i] % 2 == 1) rtn.push_back(i);
	}
	return rtn;
}

inline array<int, 2> stout_type(qlat::Coordinate& x, int mu){
	// return 0, 1, 2, 3, 4 for different stout types.
	vector<int> cnt = count_num_odd(x);
	if(cnt.size() == 0){
		return array<int, 2>{1, 0}; // type 1
	}else if(cnt.size() == 1){
		if(cnt[0] == mu) return array<int, 2>{3, 0}; // type 3
		else return array<int, 2>{2, cnt[0]}; // type 2
	}else if(cnt.size() == 2){
		if(cnt[0] == mu) return array<int, 2>{4, cnt[1]}; // type 4
		if(cnt[1] == mu) return array<int, 2>{4, cnt[0]};
		return array<int, 2>{0, 0};
	}else{
		return array<int, 2>{0, 0};
	}
}

inline void get_Fforce(
    Field<cps::Matrix>& FfField,
    Field<cps::Matrix>& FgField,
    Field<cps::Matrix>& CgField,
    const Arg_chmc& Farg,
	Chart<cps::Matrix>& Cchart
){
	// TODO!!!
	// HUGE amount of work need to be done.
	
	TIMER("get_Fforce()");
	assert(is_matching_geo(FfField.geo, FgField.geo));

	// first repeat the usual gauge force.
	if(Farg.gauge.type == qlat::WILSON){
#pragma omp parallel for
		for(long index = 0; index < FfField.geo.local_volume(); index++){
			qlat::Coordinate x; x = FfField.geo.coordinate_from_index(index);
			cps::Matrix mStaple1, mStaple2, mTemp;
			const qlat::Vector<cps::Matrix> gx = FgField.get_elems_const(x);
			qlat::Vector<cps::Matrix> fx = FfField.get_elems(x);
			for(int mu = 0; mu < FfField.geo.multiplicity; mu++){
				get_staple_dagger(mStaple1, FgField, x, mu);
				mTemp = gx[mu] * mStaple1;
				mTemp.TrLessAntiHermMatrix(); 
				fx[mu] = mTemp * qlat::Complex(0., Farg.beta / 3.);
		}}
	}
	if(Farg.gauge.type == IWASAKI){
#pragma omp parallel for
		for(long index = 0; index < FfField.geo.local_volume(); index++){
			qlat::Coordinate x; x = FfField.geo.coordinate_from_index(index);
			cps::Matrix mStaple1, mStaple2, mTemp;
			const qlat::Vector<cps::Matrix> gx = FgField.get_elems_const(x);
			qlat::Vector<cps::Matrix> fx = FfField.get_elems(x);
			for(int mu = 0; mu < FfField.geo.multiplicity; mu++){
				get_extended_staple_dagger(mStaple1, FgField, x, mu, Farg.gauge.c1);
				mTemp = gx[mu] * mStaple1;
				mTemp.TrLessAntiHermMatrix(); 
				fx[mu] = mTemp * qlat::Complex(0., Farg.beta / 3.);
		}}
	
	}

	// Now add the nasty stout part.
	
	// first evaulate the Lambda field
	Field<cps::Matrix> ClField; ClField.init(CgField.geo); 
	// assuming CgField.geo has expansion available for communication.	

#pragma omp parallel for
	for(long index = 0; index < ClField.geo.local_volume(); index++){
		qlat::Coordinate x; x = ClField.geo.coordinate_from_index(index);
		cps::Matrix mStaple1, mStaple2, mTemp;
		for(int mu = 0; mu < ClField.geo.multiplicity; mu++){
			cps::Matrix Q = get_Q(FgField, 2*x, mu, rho);
			cps::Matrix SigmaP = dagger(CgField.get_elems(x)[mu]);
			ClField.get_elems(x)[mu] = compute_Lambda(Q, SigmaP, get_U(FgField, 2*x, mu));
			// qlat::Printf("%.12f\n", ClField.get_elems(x)[mu]);
	}}
	fetch_expanded_chart(ClField, Cchart);

#pragma omp parallel for
	for(long index = 0; index < FfField.geo.local_volume(); index++){
		qlat::Coordinate x; 
		cps::Matrix mStaple1, mStaple2, mTemp; mTemp.ZeroMatrix();
		x = FfField.geo.coordinate_from_index(index);
		const qlat::Vector<cps::Matrix> gx = FgField.get_elems_const(x);
		qlat::Vector<cps::Matrix> fx = FfField.get_elems(x);
		for(int mu = 0; mu < FfField.geo.multiplicity; mu++){
			mTemp.ZeroMatrix();
			vector<int> directions(6);
			qlat::Coordinate y; // insertion pos
			qlat::Coordinate s = x; s[mu]++; // starting pos
			cps::Matrix ins; // insertion matrix
			array<int, 2> type_num = stout_type(x, mu);
//			qlat::Printf("Type %d-%d:(%d,%d,%d,%d)%d\n", type_num[0], type_num[1], x[0],x[1],x[2],x[3],mu);
			switch(type_num[0]){
				case 1:{
//					qlat::Printf("Type %d: (%d,%d,%d,%d)\n", 1, x[0],x[1],x[2],x[3]);
					y = x; y[mu]++;
					mTemp += FgField.get_elems(y)[mu] * dagger(CgField.get_elems(x/2)[mu]) * expiQ(get_Q(FgField, x, mu, rho));
//					mTemp += FgField.get_elems(y)[mu] * dagger(CgField.get_elems(x/2)[mu]);

					for(int nu = 0; nu < DIMN; nu++){
						if(nu == mu) continue;
						
						y = x;
						ins = ClField.get_elems(y/2)[mu];
//						assert(y[0]%2==0 && y[1]%2==0 && y[2]%2==0 && y[3]%2==0);
						directions[0] = mu;
						directions[1] = nu;
						directions[2] = mu+DIMN;
						directions[3] = mu+DIMN;
						directions[4] = nu+DIMN;
						directions[5] = -1;
						mTemp += get_path_ordered_product_insertion(FgField, s, directions, ins) * (+rho*i());
						
						y = x;
						ins = ClField.get_elems(y/2)[mu];
//						assert(y[0]%2==0 && y[1]%2==0 && y[2]%2==0 && y[3]%2==0);
						directions[0] = mu;
						directions[1] = nu+DIMN;
						directions[2] = mu+DIMN;
						directions[3] = mu+DIMN;
						directions[4] = nu;
						directions[5] = -1;
						mTemp += get_path_ordered_product_insertion(FgField, s, directions, ins) * (+rho*i());
						
						y = x;
						ins = ClField.get_elems(y/2)[nu];
//						assert(y[0]%2==0 && y[1]%2==0 && y[2]%2==0 && y[3]%2==0);
						directions[0] = nu;
						directions[1] = nu;
						directions[2] = mu+DIMN;
						directions[3] = nu+DIMN;
						directions[4] = nu+DIMN;
						directions[5] = -1;
						mTemp += get_path_ordered_product_insertion(FgField, s, directions, ins) * (-rho*i());
					
						y = x; y[nu] += -2;
						ins = ClField.get_elems(y/2)[nu];
//						assert(y[0]%2==0 && y[1]%2==0 && y[2]%2==0 && y[3]%2==0);
						directions[0] = nu+DIMN;
						directions[1] = nu+DIMN;
						directions[2] = mu+DIMN;
						directions[3] = -1;
						directions[4] = nu;
						directions[5] = nu;
						mTemp += get_path_ordered_product_insertion(FgField, s, directions, ins) * (+rho*i());
					}

					break;
				}
				case 2:{
					y = x; y[type_num[1]]--;
					ins = ClField.get_elems(y/2)[mu];
//						assert(y[0]%2==0 && y[1]%2==0 && y[2]%2==0 && y[3]%2==0);
					directions[0] = mu;
					directions[1] = type_num[1]+DIMN;
					directions[2] = mu+DIMN;
					directions[3] = mu+DIMN;
					directions[4] = -1;
					directions[5] = type_num[1];
					mTemp += get_path_ordered_product_insertion(FgField, s, directions, ins) * (-rho*i());
				
					y = x; y[type_num[1]]++;
					ins = ClField.get_elems(y/2)[mu];
//					assert(y[0]%2==0 && y[1]%2==0 && y[2]%2==0 && y[3]%2==0);
					directions[0] = mu;
					directions[1] = type_num[1];
					directions[2] = mu+DIMN;
					directions[3] = mu+DIMN;
					directions[4] = -1;
					directions[5] = type_num[1]+DIMN;
					mTemp += get_path_ordered_product_insertion(FgField, s, directions, ins) * (-rho*i());
					
					break;
				}
				case 3:{
//					qlat::Printf("Type %d: (%d,%d,%d,%d)\n", 3, x[0],x[1],x[2],x[3]);
					y = x; y[mu]--;
					mTemp += dagger(CgField.get_elems(y/2)[mu]) * expiQ(get_Q(FgField, y, mu, rho)) * FgField.get_elems(y)[mu];
//					mTemp += dagger(CgField.get_elems(y/2)[mu]) * FgField.get_elems(y)[mu];
					for(int nu = 0; nu < DIMN; nu++){
						if(nu == mu) continue;
						
						y = x; y[mu]--;
						ins = ClField.get_elems(y/2)[mu];
//						assert(y[0]%2==0 && y[1]%2==0 && y[2]%2==0 && y[3]%2==0);
						directions[0] = nu;
						directions[1] = mu+DIMN;
						directions[2] = mu+DIMN;
						directions[3] = nu+DIMN;
						directions[4] = -1;
						directions[5] = mu;
						mTemp += get_path_ordered_product_insertion(FgField, s, directions, ins) * (+rho*i());
					
						y = x; y[mu]--;
						ins = ClField.get_elems(y/2)[mu];
//						assert(y[0]%2==0 && y[1]%2==0 && y[2]%2==0 && y[3]%2==0);
						directions[0] = nu+DIMN;
						directions[1] = mu+DIMN;
						directions[2] = mu+DIMN;
						directions[3] = nu;
						directions[4] = -1;
						directions[5] = mu;
						mTemp += get_path_ordered_product_insertion(FgField, s, directions, ins) * (+rho*i());
					
						y = x; y[mu]++;
						ins = ClField.get_elems(y/2)[nu];
//						assert(y[0]%2==0 && y[1]%2==0 && y[2]%2==0 && y[3]%2==0);
						directions[0] = -1;
						directions[1] = nu;
						directions[2] = nu;
						directions[3] = mu+DIMN;
						directions[4] = nu+DIMN;
						directions[5] = nu+DIMN;
						mTemp += get_path_ordered_product_insertion(FgField, s, directions, ins) * (+rho*i());
					
						y = x; y[mu]++; y[nu] += -2;
//						assert(y[0]%2==0 && y[1]%2==0 && y[2]%2==0 && y[3]%2==0);
						ins = ClField.get_elems(y/2)[nu];
						directions[0] = nu+DIMN;
						directions[1] = nu+DIMN;
						directions[2] = -1;
						directions[3] = mu+DIMN;
						directions[4] = nu;
						directions[5] = nu;
						mTemp += get_path_ordered_product_insertion(FgField, s, directions, ins) * (-rho*i());
					}

					break;
				}
				case 4:{
					y = x; y[type_num[1]]--; y[mu]--; // y-\nu-\mu
					ins = ClField.get_elems(y/2)[mu];
					assert(y[0]%2==0 && y[1]%2==0 && y[2]%2==0 && y[3]%2==0);
					directions[0] = type_num[1]+DIMN;
					directions[1] = mu+DIMN;
					directions[2] = mu+DIMN;
					directions[3] = -1;
					directions[4] = type_num[1];
					directions[5] = mu;
					mTemp += get_path_ordered_product_insertion(FgField, s, directions, ins) * (-rho*i());
				
					y = x; y[type_num[1]]++; y[mu]--; // y+\nu+\mu
					ins = ClField.get_elems(y/2)[mu]; 
					assert(y[0]%2==0 && y[1]%2==0 && y[2]%2==0 && y[3]%2==0);
					directions[0] = type_num[1];
					directions[1] = mu+DIMN;
					directions[2] = mu+DIMN;
					directions[3] = -1;
					directions[4] = type_num[1]+DIMN;
					directions[5] = mu;
					mTemp += get_path_ordered_product_insertion(FgField, s, directions, ins) * (-rho*i());
	
					break;
				}
				case 0:{
					break; // do nothing
				}
				default: assert(false);
			}
			mTemp = gx[mu] * mTemp;
			mTemp.TrLessAntiHermMatrix(); 
			fx[mu] += mTemp * ( i()/eta_sqr );
	}}

}

inline void get_Cforce(
	Field<cps::Matrix>& CfField,
	Field<cps::Matrix>& FgField,
	Field<cps::Matrix>& CgField,
	const Arg_chmc& Farg
){
	TIMER("get_Cforce()");
	assert(is_matching_geo(CfField.geo, CgField.geo));

#pragma omp parallel for
	for(long index = 0; index < CfField.geo.local_volume(); index++){
		qlat::Coordinate x; x = CfField.geo.coordinate_from_index(index);
		cps::Matrix Q, Gbd, mTemp;
		const qlat::Vector<cps::Matrix> gx = CgField.get_elems_const(x);
		qlat::Vector<cps::Matrix> fx = CfField.get_elems(x);
		for(int mu = 0; mu < CfField.geo.multiplicity; mu++){
			// actual work!
			Q = get_Q(FgField, 2*x, mu, rho);
			Gbd.Dagger( expiQ(Q)*get_U(FgField, 2*x, mu) ); 		
			mTemp = gx[mu] * Gbd;
			mTemp.TrLessAntiHermMatrix(); 
			fx[mu] = mTemp * (i()/eta_sqr);
	}}
}

inline void force_gradient_integrator_multi(
	Field<cps::Matrix>& FgField, Field<cps::Matrix>& FmField, 
	Field<cps::Matrix>& FgFieldAuxil, Field<cps::Matrix>& FfField,
	const Arg_chmc& Farg, 
	Chart<cps::Matrix>& Fchart,
	Field<cps::Matrix>& CgField, Field<cps::Matrix>& CmField, 
	Field<cps::Matrix>& CgFieldAuxil, Field<cps::Matrix>& CfField,
	Chart<cps::Matrix>& Cchart
){
	// See mag.pdf for notations.

	sync_node();
	TIMER("force_gradient_integrator_multi()"); 

	assert(is_matching_geo(FgField.geo, FmField.geo));
	assert(is_matching_geo(FgField.geo, FfField.geo));
	assert(is_matching_geo(FgField.geo, FgFieldAuxil.geo));
	assert(is_matching_geo(CgField.geo, CmField.geo));
	assert(is_matching_geo(CgField.geo, CfField.geo));
	assert(is_matching_geo(CgField.geo, CgFieldAuxil.geo));
	const double alpha = (3.-sqrt(3.))*Farg.dt/6.;
	const double beta = Farg.dt/sqrt(3.);
	const double gamma = (2.-sqrt(3.))*Farg.dt*Farg.dt/12.;

	evolve_gauge_field(FgField, FmField, alpha, Farg);
	evolve_gauge_field(CgField, CmField, alpha, Farg);

	sync_node();
	for(int i = 0; i < Farg.trajectory_length; i++){
		fetch_expanded_chart(FgField, Fchart);
		fetch_expanded_chart(CgField, Cchart);
		// TODO!!!
		get_Fforce(FfField, FgField, CgField, Farg, Cchart);
		get_Cforce(CfField, FgField, CgField, Farg);
		FgFieldAuxil = FgField;
		CgFieldAuxil = CgField;
		evolve_gauge_field(FgFieldAuxil, FfField, gamma, Farg);
		evolve_gauge_field(CgFieldAuxil, CfField, gamma, Farg);
		fetch_expanded_chart(FgFieldAuxil, Fchart);
		fetch_expanded_chart(CgFieldAuxil, Cchart);
		// TODO!!!
		get_Fforce(FfField, FgFieldAuxil, CgFieldAuxil, Farg, Cchart);
		get_Cforce(CfField, FgFieldAuxil, CgFieldAuxil, Farg);
		evolve_momentum(FmField, FfField, 0.5 * Farg.dt, Farg);
		evolve_momentum(CmField, CfField, 0.5 * Farg.dt, Farg);

		evolve_gauge_field(FgField, FmField, beta, Farg);
		evolve_gauge_field(CgField, CmField, beta, Farg);
	
		fetch_expanded_chart(FgField, Fchart);
		fetch_expanded_chart(CgField, Cchart);
		// TODO!!!
		get_Fforce(FfField, FgField, CgField, Farg, Cchart);
		get_Cforce(CfField, FgField, CgField, Farg);
		FgFieldAuxil = FgField;
		CgFieldAuxil = CgField;
		evolve_gauge_field(FgFieldAuxil, FfField, gamma, Farg);
		evolve_gauge_field(CgFieldAuxil, CfField, gamma, Farg);
		fetch_expanded_chart(FgFieldAuxil, Fchart);
		fetch_expanded_chart(CgFieldAuxil, Cchart);
		// TODO!!!
		get_Fforce(FfField, FgFieldAuxil, CgFieldAuxil, Farg, Cchart);
		get_Cforce(CfField, FgFieldAuxil, CgFieldAuxil, Farg);
		evolve_momentum(FmField, FfField, 0.5 * Farg.dt, Farg);
		evolve_momentum(CmField, CfField, 0.5 * Farg.dt, Farg);

		if(i < Farg.trajectory_length - 1){
			evolve_gauge_field(FgField, FmField, 2. * alpha, Farg);
			evolve_gauge_field(CgField, CmField, 2. * alpha, Farg);
		} 
		else{
			evolve_gauge_field(FgField, FmField, alpha, Farg);
			evolve_gauge_field(CgField, CmField, alpha, Farg);
		}
	}

	qlat::Printf("reunitarize FgField: max deviation = %.8e\n", reunitarize(FgField));
	qlat::Printf("reunitarize CgField: max deviation = %.8e\n", reunitarize(CgField));
}

inline void run_hmc_multi(
	Field<cps::Matrix>& FgField_ext, const Arg_chmc &Farg,
	Field<cps::Matrix>& CgField_ext
){
	
	// perform a number of fine updates and them a number of coarse updates.
	
	TIMER("run_hmc_multi");
	
	FILE *p = NULL;
	if(Farg.summary_dir_stem.size() > 0){
		p = Fopen((Farg.summary_dir_stem + "/summary.dat").c_str(), "a");
	}

	if(!get_id_node()) assert(p != NULL);

	time_t now = time(NULL);
	Fprintf(p, "# %s", 							ctime(&now));
	Fprintf(p, "# %s\n", 						show(FgField_ext.geo).c_str());
	Fprintf(p, "# mag               = %i\n", 	Farg.mag);
	Fprintf(p, "# trajectory_length = %i\n", 	Farg.trajectory_length);
	Fprintf(p, "# num_trajectory    = %i\n", 	Farg.num_trajectory);
	Fprintf(p, "# beta              = %.6f\n", 	Farg.beta);
	Fprintf(p, "# dt                = %.5f\n", 	Farg.dt);
	Fprintf(p, "# c1                = %.5f\n", 	Farg.gauge.c1);
	Fprintf(p, "# GAUGE_TYPE        = %d\n", 	Farg.gauge.type);
	Fprintf(p, "# traj. number\texp(-DeltaH)\tavgPlaq\taccept/reject\n");
	Fflush(p);
	qlat::Printf("p opened.");

	assert(Farg.num_trajectory > 20);
	RngState globalRngState("By the witness of the martyrs.");

	qlat::Coordinate expansion(2, 2, 2, 2);

	// declare fine lattice variables
	Geometry Fgeo_expanded = FgField_ext.geo; Fgeo_expanded.resize(expansion, expansion);
	Geometry Fgeo_local = FgField_ext.geo;
	Field<cps::Matrix> FgField; 		FgField.init(Fgeo_expanded); FgField = FgField_ext;
	Field<cps::Matrix> FgField_auxil; 	FgField_auxil.init(Fgeo_expanded);
	Field<cps::Matrix> FmField; 		FmField.init(Fgeo_expanded);
	Field<cps::Matrix> FfField; 		FfField.init(Fgeo_expanded);

	Geometry Frng_geo; 
	Frng_geo.init(FmField.geo.geon, 1, FmField.geo.node_site);
	RngField Frng_field; 
	Frng_field.init(Frng_geo, RngState("Ich liebe dich."));

	//declare coarse lattice variables
	Geometry Cgeo_expanded = CgField_ext.geo; Cgeo_expanded.resize(expansion, expansion);

	Field<cps::Matrix> CgField; CgField.init(Cgeo_expanded); CgField = CgField_ext;
	Field<cps::Matrix> CgField_auxil; CgField_auxil.init(Cgeo_expanded);
	Field<cps::Matrix> CmField; CmField.init(Cgeo_expanded);
	Field<cps::Matrix> CfField; CfField.init(Cgeo_expanded);

	fetch_expanded(CgField);
	fetch_expanded(FgField);
    qlat::Printf("FINE   Plaquette = %.12f\n", avg_plaquette(FgField));	
    qlat::Printf("COARSE Plaquette = %.12f\n", avg_plaquette(CgField));	

	Geometry Crng_geo; 
	Crng_geo.init(CmField.geo.geon, 1, CmField.geo.node_site);
	RngField Crng_field; 
	Crng_field.init(Crng_geo, RngState("Tut mir leid."));

	// declare the communication patterns
	Chart<cps::Matrix> Fchart;
	produce_chart_envelope(Fchart, FgField_ext.geo, Farg.gauge);
	Chart<cps::Matrix> Cchart;
	produce_chart_envelope(Cchart, CgField_ext.geo, Farg.gauge);

	// start the hmc 
	double old_hamiltonian;
	double new_hamiltonian;
	double die_roll;
	double del_hamiltonian;
	double accept_probability;
	double average_plaquette;
	double Caverage_plaquette;
	
	vector<double> old_energy_partition;
	vector<double> new_energy_partition;
	bool does_accept;
	int num_accept = 0;
	int num_reject = 0;

	// update fine and coarse lattices in one single hmc.
	for(int i = 0; i < Farg.num_trajectory; i++){

		init_momentum(FmField, Frng_field);
		init_momentum(CmField, Crng_field);
		
		// TODO!!!
		old_hamiltonian = get_hamiltonian_multi(FgField, FmField, Farg, Fchart, CgField, CmField, Cchart, old_energy_partition);
		force_gradient_integrator_multi(FgField, FmField, FgField_auxil, FfField, Farg, Fchart,
										CgField, CmField, CgField_auxil, CfField, Cchart);
		new_hamiltonian = get_hamiltonian_multi(FgField, FmField, Farg, Fchart, CgField, CmField, Cchart, new_energy_partition);
	
		die_roll = u_rand_gen(globalRngState);
		del_hamiltonian = new_hamiltonian - old_hamiltonian;
		accept_probability = std::exp(old_hamiltonian - new_hamiltonian);
		
		does_accept = die_roll < accept_probability;
		// make sure that all the node make the same decision.
		MPI_Bcast((void *)&does_accept, 1, MPI_BYTE, 0, get_comm());
		
		if(i < Farg.num_forced_accept_step){
			qlat::Printf("End Trajectory %d:\t FORCE ACCEPT.\n", i + 1);
			FgField_ext = FgField;
			CgField_ext = CgField;
			does_accept = true;
		}else{
			if(does_accept){
				qlat::Printf("End Trajectory %d:\t ACCEPT.\n", i + 1);
				num_accept++;
				FgField_ext = FgField;
				CgField_ext = CgField;
			}else{
				qlat::Printf("End Trajectory %d:\t REJECT.\n", i + 1);
				num_reject++;
				FgField = FgField_ext;	
				CgField = CgField_ext;	
			}	
		}
		
		qlat::Printf("Old Hamiltonian =\t%+.12e\n", old_hamiltonian);
		qlat::Printf("New Hamiltonian =\t%+.12e\n", new_hamiltonian);
		qlat::Printf("Delta H         =\t%+.12e\n", del_hamiltonian); 
		qlat::Printf("exp(-Delta H)   =\t%12.6f\n", accept_probability);
		qlat::Printf("Die Roll        =\t%12.6f\n", die_roll); 	
	
		fetch_expanded_chart(FgField, Fchart);
		fetch_expanded(CgField);
		average_plaquette = avg_plaquette(FgField);
		Caverage_plaquette = avg_plaquette(CgField);
		qlat::Printf("Avg Plaquette   =\t%+.12e\n", average_plaquette); 
		qlat::Printf("Avg Plaquette C =\t%+.12e\n", Caverage_plaquette); 
		qlat::Printf("ACCEPT RATE     =\t%+.4f\n", (double)num_accept / (num_accept + num_reject));	

		if(Farg.summary_dir_stem.size() > 0){
			Fprintf(p, "%d\t%.6e\t%.6e\t%.12e\t%.12e\t%i\t%.12e\t%.12e\t%.12e\n", 
					i+1, abs(del_hamiltonian), accept_probability, average_plaquette,
					Caverage_plaquette,
					does_accept, 
					does_accept?new_energy_partition[0]:old_energy_partition[0],
					does_accept?new_energy_partition[1]:old_energy_partition[1],
					does_accept?new_energy_partition[2]:old_energy_partition[2]);
			Fflush(p);
		}

		if((i+1) % Farg.num_step_between_output == 0 && i+1 >= Farg.num_step_before_output){
			Arg_export arg_export;
			arg_export.beta = 			Farg.beta;
			arg_export.sequence_num = 	i+1;
			arg_export.ensemble_label = "multi";
			
			if(Farg.export_dir_stem.size() > 0){
				string address = Farg.export_dir_stem + "/ckpoint_lat." + show(i + 1);
				export_config_nersc(FgField_ext, address, arg_export, true);
				address = Farg.export_dir_stem + "/ckpoint_Clat." + show(i + 1);
				export_config_nersc(CgField_ext, address, arg_export, true);
			}
			
			sync_node();
		}

	}
	Timer::display();
}




QLAT_END_NAMESPACE
