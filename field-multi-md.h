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
#include "field-multi.h"
#include <cmath>

using namespace cps;
using namespace qlat;
using namespace std;

namespace md { // This a variant of the original functions.

static const double md_alpha = -1.25;

inline double get_xi_energy(
	Field<cps::Matrix>& FgField, 
	Field<cps::Matrix>& CgField,
	Field<double>& CxField // the xi field
	){
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
		const qlat::Vector<double> xx = CxField.get_elems_const(x);
		for(int mu = 0; mu < DIMN; mu++){
			// The actaul work
			// first compute Q.
			cps::Matrix Q = get_Q(FgField, 2*x, mu, rho);
			cps::Matrix Gb = expiQ(Q)*get_U(FgField, 2*x, mu); 
			cps::Matrix Ucd; Ucd.Dagger(mx[mu]);
	//		cps::Matrix Ucd; Ucd.UnitMatrix();
			pLocalSum += (Ucd*Gb).ReTr() * xx[mu];
	}}
	ppLocalSum[omp_get_thread_num()] = pLocalSum;
}
	for(int i = 0; i < numThreads; i++){
		localSum += ppLocalSum[i];
	}

	double globalSum;
	MPI_Allreduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, get_comm());
	return -globalSum * (1.+md_alpha); // note the minus sign
}

inline double get_hamiltonian_multi(
	Field<cps::Matrix>& FgField, 
	Field<cps::Matrix>& FmField, 
	const Arg_chmc& Farg, 
	Chart<cps::Matrix>& Fchart, 
	Field<cps::Matrix>& CgField, 
	Chart<cps::Matrix>& Cchart, 
	Field<double>& CxField, // the xi field
	vector<double>& part
){
	TIMER("get_hamiltonian_multi()");
	
	// momentum part
	double kinetic_energy = get_kinetic_energy(FmField);

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

	// xi part
	double xi_energy;
	xi_energy = get_xi_energy(FgField, CgField, CxField);

	// summing
	part.resize(3);
	part[0] = kinetic_energy;
	part[1] = potential_energy;
	part[2] = xi_energy;
	qlat::Printf("energy partition: %.12f\t%.12f\t%.12f\n", part[0], part[1], part[2]);
//	return kinetic_energy + potential_energy;
	return kinetic_energy + potential_energy + xi_energy;
}

inline void get_Fforce(
    Field<cps::Matrix>& FfField,
    Field<cps::Matrix>& FgField,
    Field<cps::Matrix>& CgField,
    const Arg_chmc& Farg,
	Chart<cps::Matrix>& Cchart,
	Field<double>& CxField
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
	
	fetch_expanded(CxField); // could be accerlarated by chart

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
					mTemp += FgField.get_elems(y)[mu] * dagger(CgField.get_elems(x/2)[mu]) * expiQ(get_Q(FgField, x, mu, rho)) * CxField.get_elems(x/2)[mu];
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
						mTemp += get_path_ordered_product_insertion(FgField, s, directions, ins) * (+rho*i()) * CxField.get_elems(y/2)[mu];
						
						y = x;
						ins = ClField.get_elems(y/2)[mu];
//						assert(y[0]%2==0 && y[1]%2==0 && y[2]%2==0 && y[3]%2==0);
						directions[0] = mu;
						directions[1] = nu+DIMN;
						directions[2] = mu+DIMN;
						directions[3] = mu+DIMN;
						directions[4] = nu;
						directions[5] = -1;
						mTemp += get_path_ordered_product_insertion(FgField, s, directions, ins) * (+rho*i()) * CxField.get_elems(y/2)[mu];
						
						y = x;
						ins = ClField.get_elems(y/2)[nu];
//						assert(y[0]%2==0 && y[1]%2==0 && y[2]%2==0 && y[3]%2==0);
						directions[0] = nu;
						directions[1] = nu;
						directions[2] = mu+DIMN;
						directions[3] = nu+DIMN;
						directions[4] = nu+DIMN;
						directions[5] = -1;
						mTemp += get_path_ordered_product_insertion(FgField, s, directions, ins) * (-rho*i()) * CxField.get_elems(y/2)[nu];
					
						y = x; y[nu] += -2;
						ins = ClField.get_elems(y/2)[nu];
//						assert(y[0]%2==0 && y[1]%2==0 && y[2]%2==0 && y[3]%2==0);
						directions[0] = nu+DIMN;
						directions[1] = nu+DIMN;
						directions[2] = mu+DIMN;
						directions[3] = -1;
						directions[4] = nu;
						directions[5] = nu;
						mTemp += get_path_ordered_product_insertion(FgField, s, directions, ins) * (+rho*i()) * CxField.get_elems(y/2)[nu];
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
					mTemp += get_path_ordered_product_insertion(FgField, s, directions, ins) * (-rho*i()) * CxField.get_elems(y/2)[mu];
				
					y = x; y[type_num[1]]++;
					ins = ClField.get_elems(y/2)[mu];
//					assert(y[0]%2==0 && y[1]%2==0 && y[2]%2==0 && y[3]%2==0);
					directions[0] = mu;
					directions[1] = type_num[1];
					directions[2] = mu+DIMN;
					directions[3] = mu+DIMN;
					directions[4] = -1;
					directions[5] = type_num[1]+DIMN;
					mTemp += get_path_ordered_product_insertion(FgField, s, directions, ins) * (-rho*i()) * CxField.get_elems(y/2)[mu];
					
					break;
				}
				case 3:{
//					qlat::Printf("Type %d: (%d,%d,%d,%d)\n", 3, x[0],x[1],x[2],x[3]);
					y = x; y[mu]--;
					mTemp += dagger(CgField.get_elems(y/2)[mu]) * expiQ(get_Q(FgField, y, mu, rho)) * FgField.get_elems(y)[mu] * CxField.get_elems(y/2)[mu];
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
						mTemp += get_path_ordered_product_insertion(FgField, s, directions, ins) * (+rho*i()) * CxField.get_elems(y/2)[mu];
					
						y = x; y[mu]--;
						ins = ClField.get_elems(y/2)[mu];
//						assert(y[0]%2==0 && y[1]%2==0 && y[2]%2==0 && y[3]%2==0);
						directions[0] = nu+DIMN;
						directions[1] = mu+DIMN;
						directions[2] = mu+DIMN;
						directions[3] = nu;
						directions[4] = -1;
						directions[5] = mu;
						mTemp += get_path_ordered_product_insertion(FgField, s, directions, ins) * (+rho*i()) * CxField.get_elems(y/2)[mu];
					
						y = x; y[mu]++;
						ins = ClField.get_elems(y/2)[nu];
//						assert(y[0]%2==0 && y[1]%2==0 && y[2]%2==0 && y[3]%2==0);
						directions[0] = -1;
						directions[1] = nu;
						directions[2] = nu;
						directions[3] = mu+DIMN;
						directions[4] = nu+DIMN;
						directions[5] = nu+DIMN;
						mTemp += get_path_ordered_product_insertion(FgField, s, directions, ins) * (+rho*i()) * CxField.get_elems(y/2)[nu];
					
						y = x; y[mu]++; y[nu] += -2;
//						assert(y[0]%2==0 && y[1]%2==0 && y[2]%2==0 && y[3]%2==0);
						ins = ClField.get_elems(y/2)[nu];
						directions[0] = nu+DIMN;
						directions[1] = nu+DIMN;
						directions[2] = -1;
						directions[3] = mu+DIMN;
						directions[4] = nu;
						directions[5] = nu;
						mTemp += get_path_ordered_product_insertion(FgField, s, directions, ins) * (-rho*i()) * CxField.get_elems(y/2)[nu];
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
					mTemp += get_path_ordered_product_insertion(FgField, s, directions, ins) * (-rho*i()) * CxField.get_elems(y/2)[mu];
				
					y = x; y[type_num[1]]++; y[mu]--; // y+\nu+\mu
					ins = ClField.get_elems(y/2)[mu]; 
					assert(y[0]%2==0 && y[1]%2==0 && y[2]%2==0 && y[3]%2==0);
					directions[0] = type_num[1];
					directions[1] = mu+DIMN;
					directions[2] = mu+DIMN;
					directions[3] = -1;
					directions[4] = type_num[1]+DIMN;
					directions[5] = mu;
					mTemp += get_path_ordered_product_insertion(FgField, s, directions, ins) * (-rho*i()) * CxField.get_elems(y/2)[mu];
	
					break;
				}
				case 0:{
					break; // do nothing
				}
				default: assert(false);
			}
			mTemp = gx[mu] * mTemp;
			mTemp.TrLessAntiHermMatrix(); 
			fx[mu] += mTemp * ( i() );
	}}

}

inline cps::Matrix get_DQ(
	Field<cps::Matrix>& FgField,
	Field<cps::Matrix>& FmField,
	const qlat::Coordinate& x, 
	int mu, double rho_
){
	// assuming properly communicated.
	
	vector<int> dir; dir.reserve(6);
	cps::Matrix series; series.ZeroMatrix();

	for(int nu = 0; nu < DIMN; nu++){
		if(mu == nu) continue;
		
		dir.clear();
 		dir.push_back(nu); dir.push_back(mu); dir.push_back(mu); 
		dir.push_back(nu + DIMN); dir.push_back(mu + DIMN); dir.push_back(mu + DIMN);
		series += get_path_ordered_product_leftD(FgField, FmField, x, dir);
		
		dir.clear();
 		dir.push_back(nu + DIMN); dir.push_back(mu); dir.push_back(mu); 
		dir.push_back(nu); dir.push_back(mu + DIMN); dir.push_back(mu + DIMN);
		series += get_path_ordered_product_leftD(FgField, FmField, x, dir);
	}
	series.TrLessAntiHermMatrix();
	return series*(i()*-rho_);
}

inline void get_Cforce(
	Field<cps::Matrix>& CfField,
	Field<cps::Matrix>& FgField,
	Field<cps::Matrix>& FmField
){
	TIMER("get_Cforce()");

	// assuming properly communicated.

#pragma omp parallel for
	for(long index = 0; index < CfField.geo.local_volume(); index++){
		qlat::Coordinate x; x = CfField.geo.coordinate_from_index(index);
		cps::Matrix Q, U, DQ, DU, Gbd, mTemp;
		qlat::Vector<cps::Matrix> fx = CfField.get_elems(x);
		for(int mu = 0; mu < CfField.geo.multiplicity; mu++){
			// actual work!
			vector<int> dir(2, mu);
			Q = get_Q(FgField, 2*x, mu, rho);
			U = get_U(FgField, 2*x, mu);
			DQ = get_DQ(FgField, FmField, 2*x, mu, rho);
			DU = get_path_ordered_product_leftD(FgField, FmField, 2*x, dir);
			mTemp = ( DexpiQ(Q, DQ)*U + expiQ(Q)*DU ) * dagger(expiQ(Q)*U);
			// cps::Matrix old = mTemp;
			// mTemp.TrLessAntiHermMatrix();
			// qlat::Printf("%.12e\n", ((old-mTemp)*(old-mTemp)).ReTr());
			fx[mu] = mTemp*(-1.*i()*md_alpha/(1.+md_alpha));
	}}
}
//
//inline void evolve_Cgauge_field(
//	Field<cps::Matrix>& CgField,
//	Field<cps::Matrix>& CmField,
//	Field<cps::Matrix>& FgField,
//	Field<cps::Matrix>& FmField,
//	double dt,
//	const Arg_chmck& arg
//){
//	get_Cforce(CmField, FgField, FmField);
//	evolve_gauge_field(CgField, CmField, dt, arg);
//}
//
//

inline void force_gradient_integrator_zeta_T( // zeta_T = zeta_f + zeta_c 
	Field<cps::Matrix>& FgField, 
	Field<cps::Matrix>& FmField, 
	Chart<cps::Matrix>& Fchart,
	Field<cps::Matrix>& CgField, 
	Field<cps::Matrix>& CmField, 
	double dt, const Arg_chmc& Farg 
){
	// See mag.pdf for notations.

	sync_node();
	TIMER("fg_integrator_zeta_T()"); 

	const double alpha = 1./6.;
	const double beta = 2./3.;

	fetch_expanded_chart(FmField, Fchart);
	
	evolve_gauge_field(FgField, FmField, alpha*dt, Farg);
	
	fetch_expanded_chart(FgField, Fchart);
	get_Cforce(CmField, FgField, FmField);
	evolve_gauge_field(CgField, CmField, 0.5*dt, Farg);

	evolve_gauge_field(FgField, FmField, beta*dt, Farg);
	
	fetch_expanded_chart(FgField, Fchart);
	get_Cforce(CmField, FgField, FmField);
	evolve_gauge_field(CgField, CmField, 0.5*dt, Farg);

	evolve_gauge_field(FgField, FmField, alpha*dt, Farg);

//	qlat::Printf("reunitarize FgField: max deviation = %.8e\n", reunitarize(FgField));
//	qlat::Printf("reunitarize CgField: max deviation = %.8e\n", reunitarize(CgField));
}

inline void force_gradient_integrator_zeta_H( // This is integrator is not supposed to be nested.
											 // Use a variant of the force gradient integrator
	Field<cps::Matrix>& FgField, Field<cps::Matrix>& FmField, 
	Field<cps::Matrix>& FgFieldAuxil, Field<cps::Matrix>& FfField,
	const Arg_chmc& Farg, 
	Chart<cps::Matrix>& Fchart,
	Field<cps::Matrix>& CgField, Field<cps::Matrix>& CmField, 
	Field<cps::Matrix>& CgFieldAuxil, Field<cps::Matrix>& CfField,
	Chart<cps::Matrix>& Cchart,
	Field<double>& CxField
){
	// See mag.pdf for notations.

	sync_node();
	TIMER("fg_integrator_zeta_H()"); 

	assert(is_matching_geo(FgField.geo, FmField.geo));
	assert(is_matching_geo(FgField.geo, FfField.geo));
	assert(is_matching_geo(FgField.geo, FgFieldAuxil.geo));
	assert(is_matching_geo(CgField.geo, CmField.geo));
	assert(is_matching_geo(CgField.geo, CfField.geo));
	assert(is_matching_geo(CgField.geo, CgFieldAuxil.geo));
	assert(is_matching_geo(CgField.geo, CxField.geo));
	
	const double alpha = (3.-sqrt(3.))*Farg.dt/6.;
	const double beta = Farg.dt/sqrt(3.);
	const double gamma = (2.-sqrt(3.))*Farg.dt*Farg.dt/12.;

//	fetch_expanded_chart(FgField, Fchart);
//	fetch_expanded_chart(FmField, Fchart);
//	get_Cforce(CmField, FgField, FmField);
//	evolve_gauge_field(FgField, FmField, alpha, Farg);
//	evolve_gauge_field(CgField, CmField, alpha, Farg);

	force_gradient_integrator_zeta_T(FgField, FmField, Fchart, CgField, CmField, alpha, Farg);

	sync_node();
	for(int i = 0; i < Farg.trajectory_length; i++){

		fetch_expanded_chart(FgField, Fchart);
		fetch_expanded_chart(CgField, Cchart);
		// TODO!!!
		get_Fforce(FfField, FgField, CgField, Farg, Cchart, CxField);
//		get_Cforce(CfField, FgField, CgField, Farg, CxField);
		FgFieldAuxil = FgField;
		CgFieldAuxil = CgField;
		
//		fetch_expanded_chart(FfField, Fchart);
//		get_Cforce(CfField, FgField, FfField);
//		evolve_gauge_field(FgFieldAuxil, FfField, gamma, Farg);
//		evolve_gauge_field(CgFieldAuxil, CfField, gamma, Farg);
		
		force_gradient_integrator_zeta_T(FgFieldAuxil, FfField, Fchart, CgFieldAuxil, CmField, gamma, Farg);
		
		fetch_expanded_chart(FgFieldAuxil, Fchart);
		fetch_expanded_chart(CgFieldAuxil, Cchart);
		// TODO!!!

		get_Fforce(FfField, FgFieldAuxil, CgFieldAuxil, Farg, Cchart, CxField);
		evolve_momentum(FmField, FfField, 0.5 * Farg.dt, Farg);
	
//		get_Fforce(FfField, FgField, CgFieldAuxil, Farg, Cchart, CxField);
//		evolve_momentum(FmField, FfField, 0.5 * Farg.dt, Farg);

// -----------		

//		fetch_expanded_chart(FgField, Fchart);
//		fetch_expanded_chart(FmField, Fchart);
//		get_Cforce(CmField, FgField, FmField);
//		evolve_gauge_field(FgField, FmField, beta, Farg);
//		evolve_gauge_field(CgField, CmField, beta, Farg);

		force_gradient_integrator_zeta_T(FgField, FmField, Fchart, CgField, CmField, beta, Farg);

// -----------		
		fetch_expanded_chart(FgField, Fchart);
		fetch_expanded_chart(CgField, Cchart);
		// TODO!!!
		get_Fforce(FfField, FgField, CgField, Farg, Cchart, CxField);
//		get_Cforce(CfField, FgField, CgField, Farg, CxField);
		FgFieldAuxil = FgField;
		CgFieldAuxil = CgField;
		
//		fetch_expanded_chart(FfField, Fchart);
//		get_Cforce(CfField, FgField, FfField);
//		evolve_gauge_field(FgFieldAuxil, FfField, gamma, Farg);
//		evolve_gauge_field(CgFieldAuxil, CfField, gamma, Farg);
		
		force_gradient_integrator_zeta_T(FgFieldAuxil, FfField, Fchart, CgFieldAuxil, CmField, gamma, Farg);
		
		fetch_expanded_chart(FgFieldAuxil, Fchart);
		fetch_expanded_chart(CgFieldAuxil, Cchart);
		// TODO!!!

		get_Fforce(FfField, FgFieldAuxil, CgFieldAuxil, Farg, Cchart, CxField);
		evolve_momentum(FmField, FfField, 0.5 * Farg.dt, Farg);
	
//		get_Fforce(FfField, FgField, CgFieldAuxil, Farg, Cchart, CxField);
//		evolve_momentum(FmField, FfField, 0.5 * Farg.dt, Farg);
	
		if(i < Farg.trajectory_length - 1){
//			fetch_expanded_chart(FgField, Fchart);
//			fetch_expanded_chart(FmField, Fchart);
//			get_Cforce(CmField, FgField, FmField);
//			evolve_gauge_field(FgField, FmField, 2.*alpha, Farg);
//			evolve_gauge_field(CgField, CmField, 2.*alpha, Farg);
			force_gradient_integrator_zeta_T(FgField, FmField, Fchart, CgField, CmField, 2.*alpha, Farg);
		} 
		else{
//			fetch_expanded_chart(FgField, Fchart);
//			fetch_expanded_chart(FmField, Fchart);
//			get_Cforce(CmField, FgField, FmField);
//			evolve_gauge_field(FgField, FmField, alpha, Farg);
//			evolve_gauge_field(CgField, CmField, alpha, Farg);
			force_gradient_integrator_zeta_T(FgField, FmField, Fchart, CgField, CmField, alpha, Farg);
		}
	}

	qlat::Printf("reunitarize FgField: max deviation = %.8e\n", reunitarize(FgField));
	qlat::Printf("reunitarize CgField: max deviation = %.8e\n", reunitarize(CgField));
}

inline void force_gradient_integrator(
	Field<cps::Matrix>& FgField, 
	Field<cps::Matrix>& FmField, 
	Field<cps::Matrix>& FgFieldAuxil, 
	Field<cps::Matrix>& FfField,
	const Arg_chmc& Farg, double dt, int steps,
	Chart<cps::Matrix>& Fchart,
	Field<cps::Matrix>& CgField, 
	Chart<cps::Matrix>& Cchart,
	Field<double>& CxField
){
	// See mag.pdf for notations.
	// ONLY update Fg and Fm.
	sync_node();
	TIMER("force_gradient_integrator()"); 

	assert(is_matching_geo(FgField.geo, FmField.geo));
	assert(is_matching_geo(FgField.geo, FfField.geo));
	assert(is_matching_geo(FgField.geo, FgFieldAuxil.geo));

	const double alpha = (3.-sqrt(3.))*Farg.dt/6.;
	const double beta = Farg.dt/sqrt(3.);
	const double gamma = (2.-sqrt(3.))*Farg.dt*Farg.dt/12.;

	fetch_expanded_chart(CgField, Cchart);
	
	evolve_gauge_field(FgField, FmField, alpha, Farg);
	sync_node();
	
	for(int i = 0; i < steps; i++){

		fetch_expanded_chart(FgField, Fchart);
		get_Fforce(FfField, FgField, CgField, Farg, Cchart, CxField);
		FgFieldAuxil = FgField;
		evolve_gauge_field(FgFieldAuxil, FfField, gamma, Farg);
		fetch_expanded_chart(FgFieldAuxil, Fchart);
		get_Fforce(FfField, FgFieldAuxil, CgField, Farg, Cchart, CxField);
		evolve_momentum(FmField, FfField, 0.5 * Farg.dt, Farg);
	
// -----------		

		evolve_gauge_field(FgField, FmField, beta, Farg);

// -----------		
		fetch_expanded_chart(FgField, Fchart);
		get_Fforce(FfField, FgField, CgField, Farg, Cchart, CxField);
		FgFieldAuxil = FgField;
		evolve_gauge_field(FgFieldAuxil, FfField, gamma, Farg);
		fetch_expanded_chart(FgFieldAuxil, Fchart);
		get_Fforce(FfField, FgFieldAuxil, CgField, Farg, Cchart, CxField);
		evolve_momentum(FmField, FfField, 0.5 * Farg.dt, Farg);
	
		if(i < steps-1){
			evolve_gauge_field(FgField, FmField, 2.*alpha, Farg);
		} 
		else{
			evolve_gauge_field(FgField, FmField, alpha, Farg);
		}
	}

	qlat::Printf("reunitarize FgField: max deviation = %.8e\n", reunitarize(FgField));
	qlat::Printf("reunitarize CgField: max deviation = %.8e\n", reunitarize(CgField));
}

inline void nested_integrator(
    Field<cps::Matrix>& FgField, 
	Field<cps::Matrix>& FmField,
    Field<cps::Matrix>& FgFieldAuxil, 
	Field<cps::Matrix>& FfField,
    const Arg_chmc& Farg, Chart<cps::Matrix>& Fchart,
    Field<cps::Matrix>& CgField, 
	Field<cps::Matrix>& CmField,
    Chart<cps::Matrix>& Cchart,
    Field<double>& CxField 
){
	fetch_expanded_chart(FgField, Fchart);
	fetch_expanded_chart(FmField, Fchart);
	get_Cforce(CmField, FgField, FmField);
	evolve_gauge_field(CgField, CmField, Farg.dt/2., Farg);
	for(int i = 0; i < Farg.trajectory_length; i++){
		force_gradient_integrator(FgField, FmField, FgFieldAuxil, FfField,
						    		Farg, Farg.dt, 1, Fchart,
								    CgField, Cchart, CxField);
		if(i < Farg.trajectory_length - 1){
			fetch_expanded_chart(FgField, Fchart);
			fetch_expanded_chart(FmField, Fchart);
			get_Cforce(CmField, FgField, FmField);
			evolve_gauge_field(CgField, CmField, Farg.dt, Farg);
		} 
		else{
			fetch_expanded_chart(FgField, Fchart);
			fetch_expanded_chart(FmField, Fchart);
			get_Cforce(CmField, FgField, FmField);
			evolve_gauge_field(CgField, CmField, Farg.dt/2., Farg);
		}
	}
}

inline void init_xi(
	Field<double>& CxField,
	Field<cps::Matrix>& FgField,	
	Field<cps::Matrix>& CgField,
	Chart<cps::Matrix>& Fchart,
	RngField& rng_field,
	Field<double>& CbField
	){

	fetch_expanded_chart(FgField, Fchart);

#pragma omp parallel for
	for(long index = 0; index < CxField.geo.local_volume(); index++){
		qlat::Coordinate x = CxField.geo.coordinate_from_index(index);
		qlat::Vector<double> xx = CxField.get_elems(x);
		for(int mu = 0; mu < DIMN; mu++){
			xx[mu] = XI0;
	}}

//#pragma omp parallel for
//	for(long index = 0; index < CxField.geo.local_volume(); index++){
//		qlat::Coordinate x = CxField.geo.coordinate_from_index(index);
//		qlat::Vector<cps::Matrix> mx = CgField.get_elems(x);
//		qlat::Vector<double> xx = CxField.get_elems(x);
//		qlat::Vector<double> bx = CbField.get_elems(x);
//		double B;
//		for(int mu = 0; mu < DIMN; mu++){
//			// The actaul work
//			// first compute Q.
//			cps::Matrix Q = get_Q(FgField, 2*x, mu, rho);
//			cps::Matrix Gb = expiQ(Q)*get_U(FgField, 2*x, mu); 
//			cps::Matrix Ucd; Ucd.Dagger(mx[mu]);
//	//		cps::Matrix Ucd; Ucd.UnitMatrix(); 
//			B = (Ucd*Gb).ReTr();
//			double p = 1./(1.+std::exp((8.-2.)*B - (13.942245-1.235890))); // F2 = 24.33674, F1 = 1.23589  
////			xx[mu] = g_rand_gen(rng_field.get_elem(x), XI0+ALPHA*ALPHA*B, ALPHA);
//			if(u_rand_gen(rng_field.get_elem(x), 1., 0) < p){
//			//	xx[mu] = g_rand_gen(rng_field.get_elem(x), 1.6+0.16*B, 0.4);
//				xx[mu] = 2.;
//				bx[mu] = B;
////				mx[mu] = dagger(mx[mu]);
//			}
//			else{
//			//	xx[mu] = g_rand_gen(rng_field.get_elem(x), 11.6+0.16*B, 0.4);
//				xx[mu] = 8.;
//				bx[mu] = B;
//			}
//	}}

}

inline void simplest_metropolis(cps::Matrix& var, const cps::Matrix& env, double coeff, RngState& rng){
	cps::Matrix new_var = var * random_su3_from_su2(0.8, rng); // small = 0.8
	// TODO!!! should change the interface.
	double diff = coeff*(new_var*dagger(env)-var*dagger(env)).ReTr();
	if(u_rand_gen(rng, 1., 0.) < std::exp(diff)){
		var = new_var;
	}
}

inline void heatbath(Field<cps::Matrix>& CgField, Field<cps::Matrix>& FgField, RngField& rng_field){
	// assumming properly communicated.
#pragma omp parallel for
	for(long index = 0; index < CgField.geo.local_volume(); index++){
		qlat::Coordinate x = CgField.geo.coordinate_from_index(index);
		qlat::Vector<cps::Matrix> gx = CgField.get_elems(x);
		cps::Matrix Q, U;
		for(int mu = 0; mu < DIMN; mu++){
			Q = get_Q(FgField, 2*x, mu, rho);
			U = get_U(FgField, 2*x, mu);
			simplest_metropolis(gx[mu], expiQ(Q)*U, (1.+md_alpha)*XI0, rng_field.get_elem(x));
	}}
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
	Field<double> CxField; CxField.init(Cgeo_expanded); // xi field
	Field<double> CbField; CbField.init(Cgeo_expanded); // xi field

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
//		should do Metropolis for CgField.
		
		fetch_expanded_chart(FgField, Fchart);
		heatbath(CgField, FgField, Crng_field);

		fetch_expanded(CgField);
		fetch_expanded(FgField);
    	qlat::Printf("FINE   Plaquette = %.12f\n", avg_plaquette(FgField));	
    	qlat::Printf("COARSE Plaquette = %.12f\n", avg_plaquette(CgField));	

//		init_momentum(CmField, Crng_field, M);
	
		init_xi(CxField, FgField, CgField, Fchart, Crng_field, CbField);

		// TODO!!!
		old_hamiltonian = get_hamiltonian_multi(FgField, FmField, Farg, Fchart, CgField, Cchart, CxField, old_energy_partition);
		force_gradient_integrator_zeta_H(FgField, FmField, FgField_auxil, FfField, Farg, Fchart,
										CgField, CmField, CgField_auxil, CfField, Cchart, CxField);
//		nested_integrator(FgField, FmField, FgField_auxil, FfField, Farg, Fchart,
//							CgField, CmField, Cchart, CxField);
		new_hamiltonian = get_hamiltonian_multi(FgField, FmField, Farg, Fchart, CgField, Cchart, CxField, new_energy_partition);
	
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
				
//				address = Farg.export_dir_stem + "/ckpoint_Xlat." + show(i + 1);
//				qlat::Field<double> X_output; X_output.init(CxField.geo);
//				qlat::sophisticated_make_to_order(X_output, CxField);
//				qlat::sophisticated_serial_write(X_output, address);
//				
//				address = Farg.export_dir_stem + "/ckpoint_Blat." + show(i + 1);
//				qlat::Field<double> B_output; B_output.init(CbField.geo);
//				qlat::sophisticated_make_to_order(B_output, CbField);
//				qlat::sophisticated_serial_write(B_output, address);

			}
			
			sync_node();
		}

	}
	Timer::display();
}

}
