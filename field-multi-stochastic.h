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

namespace stochastic { // This a variant of the original functions.

inline double get_kinetic_energy(Field<cps::Matrix>& mField){

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
	return globalSum / 2.;
}

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
	return -globalSum; // note the minus sign
}

inline double get_hamiltonian_multi(
	Field<cps::Matrix>& FgField, 
	Field<cps::Matrix>& FmField, 
	const Arg_chmc& Farg, 
	Chart<cps::Matrix>& Fchart, 
	Field<cps::Matrix>& CgField, 
	Field<cps::Matrix>& CmField, 
	Chart<cps::Matrix>& Cchart, 
	Field<double>& CxField, // the xi field
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
						mTemp += get_path_ordered_product_insertion(FgField, s, directions, ins) * (-rho*i()) * CxField.get_elems(y/2)[mu];
					
						y = x; y[nu] += -2;
						ins = ClField.get_elems(y/2)[nu];
//						assert(y[0]%2==0 && y[1]%2==0 && y[2]%2==0 && y[3]%2==0);
						directions[0] = nu+DIMN;
						directions[1] = nu+DIMN;
						directions[2] = mu+DIMN;
						directions[3] = -1;
						directions[4] = nu;
						directions[5] = nu;
						mTemp += get_path_ordered_product_insertion(FgField, s, directions, ins) * (+rho*i()) * CxField.get_elems(y/2)[mu];
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
						mTemp += get_path_ordered_product_insertion(FgField, s, directions, ins) * (+rho*i()) * CxField.get_elems(y/2)[mu];
					
						y = x; y[mu]++; y[nu] += -2;
//						assert(y[0]%2==0 && y[1]%2==0 && y[2]%2==0 && y[3]%2==0);
						ins = ClField.get_elems(y/2)[nu];
						directions[0] = nu+DIMN;
						directions[1] = nu+DIMN;
						directions[2] = -1;
						directions[3] = mu+DIMN;
						directions[4] = nu;
						directions[5] = nu;
						mTemp += get_path_ordered_product_insertion(FgField, s, directions, ins) * (-rho*i()) * CxField.get_elems(y/2)[mu];
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

inline void get_Cforce(
	Field<cps::Matrix>& CfField,
	Field<cps::Matrix>& FgField,
	Field<cps::Matrix>& CgField,
	const Arg_chmc& Farg,
	Field<double>& CxField
){
	TIMER("get_Cforce()");
	assert(is_matching_geo(CfField.geo, CgField.geo));

#pragma omp parallel for
	for(long index = 0; index < CfField.geo.local_volume(); index++){
		qlat::Coordinate x; x = CfField.geo.coordinate_from_index(index);
		cps::Matrix Q, Gbd, mTemp;
		const qlat::Vector<cps::Matrix> gx = CgField.get_elems_const(x);
		const qlat::Vector<double> xx = CxField.get_elems_const(x);
		qlat::Vector<cps::Matrix> fx = CfField.get_elems(x);
		for(int mu = 0; mu < CfField.geo.multiplicity; mu++){
			// actual work!
			Q = get_Q(FgField, 2*x, mu, rho);
			Gbd.Dagger( expiQ(Q)*get_U(FgField, 2*x, mu) ); 		
			mTemp = gx[mu] * Gbd;
			mTemp.TrLessAntiHermMatrix(); 
			fx[mu] = mTemp * (i() * xx[mu]);
	}}
}

inline void force_gradient_integrator_multi(
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
	TIMER("force_gradient_integrator_multi()"); 

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

	evolve_gauge_field(FgField, FmField, alpha, Farg);
	evolve_gauge_field(CgField, CmField, alpha, Farg);

	sync_node();
	for(int i = 0; i < Farg.trajectory_length; i++){
		fetch_expanded_chart(FgField, Fchart);
		fetch_expanded_chart(CgField, Cchart);
		// TODO!!!
		get_Fforce(FfField, FgField, CgField, Farg, Cchart, CxField);
		get_Cforce(CfField, FgField, CgField, Farg, CxField);
		FgFieldAuxil = FgField;
		CgFieldAuxil = CgField;
		evolve_gauge_field(FgFieldAuxil, FfField, gamma, Farg);
		evolve_gauge_field(CgFieldAuxil, CfField, gamma, Farg);
		fetch_expanded_chart(FgFieldAuxil, Fchart);
		fetch_expanded_chart(CgFieldAuxil, Cchart);
		// TODO!!!
		get_Fforce(FfField, FgFieldAuxil, CgFieldAuxil, Farg, Cchart, CxField);
		get_Cforce(CfField, FgFieldAuxil, CgFieldAuxil, Farg, CxField);
		evolve_momentum(FmField, FfField, 0.5 * Farg.dt, Farg);
		evolve_momentum(CmField, CfField, 0.5 * Farg.dt, Farg);

		evolve_gauge_field(FgField, FmField, beta, Farg);
		evolve_gauge_field(CgField, CmField, beta, Farg);
	
		fetch_expanded_chart(FgField, Fchart);
		fetch_expanded_chart(CgField, Cchart);
		// TODO!!!
		get_Fforce(FfField, FgField, CgField, Farg, Cchart, CxField);
		get_Cforce(CfField, FgField, CgField, Farg, CxField);
		FgFieldAuxil = FgField;
		CgFieldAuxil = CgField;
		evolve_gauge_field(FgFieldAuxil, FfField, gamma, Farg);
		evolve_gauge_field(CgFieldAuxil, CfField, gamma, Farg);
		fetch_expanded_chart(FgFieldAuxil, Fchart);
		fetch_expanded_chart(CgFieldAuxil, Cchart);
		// TODO!!!
		get_Fforce(FfField, FgFieldAuxil, CgFieldAuxil, Farg, Cchart, CxField);
		get_Cforce(CfField, FgFieldAuxil, CgFieldAuxil, Farg, CxField);
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

inline void init_xi(
	Field<double>& CxField,
	Field<cps::Matrix>& FgField,	
	Field<cps::Matrix>& CgField,	
	RngField& rng_field
	){
#pragma omp parallel for
	for(long index = 0; index < CxField.geo.local_volume(); index++){
		qlat::Coordinate x = CxField.geo.coordinate_from_index(index);
		const qlat::Vector<cps::Matrix> mx = CgField.get_elems_const(x);
		qlat::Vector<double> xx = CxField.get_elems(x);
		double B;
		for(int mu = 0; mu < DIMN; mu++){
			// The actaul work
			// first compute Q.
			cps::Matrix Q = get_Q(FgField, 2*x, mu, rho);
			cps::Matrix Gb = expiQ(Q)*get_U(FgField, 2*x, mu); 
			cps::Matrix Ucd; Ucd.Dagger(mx[mu]);
	//		cps::Matrix Ucd; Ucd.UnitMatrix(); 
			B = (Ucd*Gb).ReTr();
			xx[mu] = g_rand_gen(rng_field.get_elems(x), XI0+ALPHA*ALPHA*B, ALPHA);
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
		old_hamiltonian = get_hamiltonian_multi(FgField, FmField, Farg, Fchart, CgField, CmField, Cchart, CxField, old_energy_partition);
		force_gradient_integrator_multi(FgField, FmField, FgField_auxil, FfField, Farg, Fchart,
										CgField, CmField, CgField_auxil, CfField, Cchart, CxField);
		new_hamiltonian = get_hamiltonian_multi(FgField, FmField, Farg, Fchart, CgField, CmField, Cchart, CxField, new_energy_partition);
	
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

}
