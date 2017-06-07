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
#include <cmath>

using namespace cps;
using namespace qlat;
using namespace std;

#define SU3_NUM_OF_GENERATORS 8

QLAT_START_NAMESPACE

static const double inv_sqrt2 = 1. / sqrt(2.);

inline vector<cps::Matrix> init_generators(){
	cps::Matrix T1, T2, T3, T4, T5, T6, T7, T8;
	// the eight Hermitian generators of SU3	
	T1.ZeroMatrix();
	T1(0, 1) = qlat::Complex(1., 0.);  T1(1, 0) = qlat::Complex(1., 0.);
	T1 *= inv_sqrt2;

	T2.ZeroMatrix();
	T2(0, 1) = qlat::Complex(0., -1.); T2(1, 0) = qlat::Complex(0., 1.);
	T2 *= inv_sqrt2;

	T3.ZeroMatrix();
	T3(0, 0) = qlat::Complex(1., 0.);  T3(1, 1) = qlat::Complex(-1., 0.);
	T3 *= inv_sqrt2;

	T4.ZeroMatrix();
	T4(0, 2) = qlat::Complex(1., 0.);  T4(2, 0) = qlat::Complex(1., 0.);
	T4 *= inv_sqrt2;

	T5.ZeroMatrix();
	T5(0, 2) = qlat::Complex(0., -1.); T5(2, 0) = qlat::Complex(0., 1.);
	T5 *= inv_sqrt2;

	T6.ZeroMatrix();
	T6(1, 2) = qlat::Complex(1., 0.);  T6(2, 1) = qlat::Complex(1., 0.);
	T6 *= inv_sqrt2;

	T7.ZeroMatrix();
	T7(1, 2) = qlat::Complex(0., -1.); T7(2, 1) = qlat::Complex(0., 1.);
	T7 *= inv_sqrt2;

	T8.ZeroMatrix();
	T8(0 ,0) = qlat::Complex(1., 0.);  T8(1, 1) = qlat::Complex(1., 0.); 
	T8(2, 2) = qlat::Complex(-2., 0.);
	T8 *= 1. / sqrt(6.);
	
	vector<cps::Matrix> ret {T1, T2, T3, T4, T5, T6, T7, T8};
	
	return ret;
}

static const vector<cps::Matrix> su3_generators = init_generators();

inline void exp(cps::Matrix &expM, const cps::Matrix &M){
        cps::Matrix mTemp2 = M, mTemp3;
	for(int i = 9; i > 1; i--){
		mTemp3.OneMinusfTimesM(-1. / i, mTemp2);
		mTemp2.DotMEqual(M, mTemp3);
	}
	expM.OneMinusfTimesM(-1., mTemp2);
}

inline void algebra_to_group(cps::Matrix &expiM, const cps::Matrix &M){
	// expiM = exp(i * M)
	cps::Matrix mTemp = M; mTemp *= qlat::Complex(0., 1.);
	exp(expiM, mTemp);
}

inline void get_rectangular_dagger(cps::Matrix &rec, const Field<cps::Matrix> &field,
														const Coordinate &x, int mu){
	// Assuming properly communicated
 	vector<int> dir; dir.reserve(5);
 	cps::Matrix rec_, m; rec_.ZeroMatrix();
 	for(int nu = 0; nu < DIMN; nu++){
 		if(mu == nu) continue;
 		
		dir.clear();
 		dir.push_back(nu); 
		dir.push_back(mu);
		dir.push_back(mu);
		dir.push_back(nu + DIMN);
		dir.push_back(mu + DIMN);
 		get_path_ordered_product(m, field, x, dir);
 		rec_ += m;
 		
		dir.clear();
 		dir.push_back(nu + DIMN); 
		dir.push_back(mu);
		dir.push_back(mu);
		dir.push_back(nu);
		dir.push_back(mu + DIMN);
 		get_path_ordered_product(m, field, x, dir);
 		rec_ += m;

		dir.clear();
 		dir.push_back(nu); 
		dir.push_back(nu);
		dir.push_back(mu);
		dir.push_back(nu + DIMN);
		dir.push_back(nu + DIMN);
 		get_path_ordered_product(m, field, x, dir);
 		rec_ += m;

		dir.clear();
 		dir.push_back(nu + DIMN); 
		dir.push_back(nu + DIMN);
		dir.push_back(mu);
		dir.push_back(nu);
		dir.push_back(nu);
 		get_path_ordered_product(m, field, x, dir);
 		rec_ += m;	

		dir.clear();
 		dir.push_back(mu + DIMN); 
		dir.push_back(nu);
		dir.push_back(mu);
		dir.push_back(mu);
		dir.push_back(nu + DIMN);
 		get_path_ordered_product(m, field, x, dir);
 		rec_ += m;
	
		dir.clear();
 		dir.push_back(mu + DIMN); 
		dir.push_back(nu + DIMN);
		dir.push_back(mu);
		dir.push_back(mu);
		dir.push_back(nu);
 		get_path_ordered_product(m, field, x, dir);
 		rec_ += m;	
	}
	
	rec.Dagger(rec_);
}

inline void get_staple_dagger(cps::Matrix &staple, const Field<cps::Matrix> &field, 
														const Coordinate &x, int mu){
//	Coordinate y = x;
//	const qlat::Vector<cps::Matrix> gx = field.get_elems_const(x);
//	vector<qlat::Vector<cps::Matrix> > gxex(DIMN * 2);
//	for(int alpha = 0; alpha < DIMN; alpha++){
//		y[alpha]++;
//		gxex[alpha] = field.get_elems_const(y);
//		y[alpha] -= 2;
//		gxex[alpha + DIMN] = field.get_elems_const(y);
//		y[alpha]++;
//	}
//	cps::Matrix m, dagger, acc; acc.ZeroMatrix();
//	for(int nu = 0; nu < DIMN; nu++){
//		if(mu == nu) continue;
//		dagger.Dagger(gxex[mu][nu]);
//		acc += (gx[nu] * gxex[nu][mu]) * dagger;
//		dagger.Dagger(gxex[nu + DIMN][nu]);
//		Coordinate z = x; z[nu]--; z[mu]++;
//		acc += dagger * (gxex[nu + DIMN][mu] * field.get_elems_const(z)[nu]);
//	}
//	staple.Dagger(acc);

 	vector<int> dir; dir.reserve(3);
 	cps::Matrix staple_; staple_.ZeroMatrix();
 	cps::Matrix m;
 	for(int nu = 0; nu < DIMN; nu++){
 		if(mu == nu) continue;
 		dir.clear();
 		dir.push_back(nu); dir.push_back(mu); dir.push_back(nu + DIMN);
 		get_path_ordered_product(m, field, x, dir);
 		staple_ += m;
 		dir.clear();
 		dir.push_back(nu + DIMN); dir.push_back(mu); dir.push_back(nu);
 		get_path_ordered_product(m, field, x, dir);
 		staple_ += m;
 	}
 	staple.Dagger(staple_);

}

inline void get_staple_2x1(cps::Matrix &staple, const Field<cps::Matrix> &field, 
														const Coordinate &x, int mu){
 	vector<int> dir; dir.reserve(4);
 	cps::Matrix staple_; staple_.ZeroMatrix();
 	cps::Matrix m;
 	for(int nu = 0; nu < DIMN; nu++){
 		if(mu == nu) continue;
 		dir.clear();
 		dir.push_back(nu); dir.push_back(mu); dir.push_back(mu); dir.push_back(nu + DIMN);
 		get_path_ordered_product(m, field, x, dir);
 		staple_ += m;
 		dir.clear();
 		dir.push_back(nu + DIMN); dir.push_back(mu); dir.push_back(mu); dir.push_back(nu);
 		get_path_ordered_product(m, field, x, dir);
 		staple_ += m;
 	}
 	staple = staple_;
}

inline void get_extended_staple_dagger(cps::Matrix &stp, const Field<cps::Matrix> &f,
									const Coordinate &x, int mu, double c1){
	double c0 = (1 - 8. * c1);
	cps::Matrix m0, m1;
	get_staple_dagger(m0, f, x, mu);
	get_rectangular_dagger(m1, f, x, mu);
	stp = m0 * c0 + m1 * c1; 
}

inline void rn_filling_SHA256_gaussian(std::vector<double> &xs)
{
	using namespace qlat;
	static bool initialized = false;
	static Geometry geo;
	static qlat::RngField rf;
	if (false == initialized){
		geo.init(get_size_node(), 1);
		rf.init(geo, RngState("Ich liebe dich."));
		initialized = true;
	}
	assert(xs.size() % geo.local_volume()== 0);
	const int chunk = xs.size() / geo.local_volume();
#pragma omp parallel for
	for (long index = 0; index < geo.local_volume(); index++){
		Coordinate xl = geo.coordinate_from_index(index);
		RngState& rs = rf.get_elem(xl);
		for (int i = chunk * index; i < chunk * (index + 1); ++i){
			xs[i] = g_rand_gen(rs);
		}
	}
}

class Arg_chmc{
public:
	int mag;
	int trajectory_length;
	int num_trajectory;
	int num_step_between_output;
	int num_forced_accept_step;
	int num_step_before_output;
	double beta;
	double dt;
	string export_dir_stem; // config output
	string summary_dir_stem;
	Gauge gauge;
};

inline int is_constrained(const Coordinate &x, int mu, int mag)
{
	// return 0: not constrained;
	// return 1: constrained but neither the first nor the last one 
	// on the segment
	// return 10: constrained and the first one on the segment
	// return 100: constrained and the last one on the segment

	// debug start

	// return 0;

	// debug end

	if(!mag) return 0; // this is not a constrained evolution: always return 0;

	bool is_constrained_ = true;
	for(int i = 0; i < 4; i++){
		if(i == mu) continue;
		is_constrained_ = is_constrained_ && (x[i] % mag == 0);
	}
	if(is_constrained_){
		if(x[mu] % mag == mag - 1) return 100;
		if(x[mu] % mag == 0) return 10;
		return 1;
	}else{
		return 0;
	}
}

inline void get_force(Field<cps::Matrix> &fField, const Field<cps::Matrix> &gField,
			const Arg_chmc &arg){
	TIMER("get_force()");
	assert(is_matching_geo(fField.geo, gField.geo));

	if(arg.gauge.type == qlat::WILSON){
#pragma omp parallel for
		for(long index = 0; index < fField.geo.local_volume(); index++){
			Coordinate x; 
			cps::Matrix mStaple1, mStaple2, mTemp;
			x = fField.geo.coordinate_from_index(index);
			const qlat::Vector<cps::Matrix> gx = gField.get_elems_const(x);
			qlat::Vector<cps::Matrix> fx = fField.get_elems(x);
			for(int mu = 0; mu < fField.geo.multiplicity; mu++){
				switch(is_constrained(x, mu, arg.mag)){
				case 0: {
					get_staple_dagger(mStaple1, gField, x, mu);
					mTemp = gx[mu] * mStaple1;
					break;
				}
				case 1:
				case 10: {
					Coordinate y(x); y[mu]++;
					get_staple_dagger(mStaple1, gField, x, mu);
					get_staple_dagger(mStaple2, gField, y, mu);
					mTemp = gField.get_elems_const(y)[mu] * mStaple2 - mStaple1 * gx[mu];
					break;
				}
				case 100: mTemp.ZeroMatrix(); break;
			
			// 	test case start
			// 	case 100: {
			// 		get_staple_dagger(mStaple1, gField, x, mu);
			// 		mTemp = mStaple1 * gField.get_elems_const(x)[mu] * -1.;
			// 		break;
			// 	} 
		 	// 	test case end
		
			 	default: assert(false);
				}
		
				mTemp.TrLessAntiHermMatrix(); 
				fx[mu] = mTemp * qlat::Complex(0., arg.beta / 3.);
		}}
	}
	if(arg.gauge.type == IWASAKI){
#pragma omp parallel for
		for(long index = 0; index < fField.geo.local_volume(); index++){
			Coordinate x; 
			cps::Matrix mStaple1, mStaple2, mTemp;
			x = fField.geo.coordinate_from_index(index);
			const qlat::Vector<cps::Matrix> gx = gField.get_elems_const(x);
			qlat::Vector<cps::Matrix> fx = fField.get_elems(x);
			for(int mu = 0; mu < fField.geo.multiplicity; mu++){
				switch(is_constrained(x, mu, arg.mag)){
				case 0: {
					get_extended_staple_dagger(mStaple1, gField, x, mu, arg.gauge.c1);
					mTemp = gx[mu] * mStaple1;
					break;
				}
				case 1:
				case 10: {
					Coordinate y(x); y[mu]++;
					get_extended_staple_dagger(mStaple1, gField, x, mu, arg.gauge.c1);
					get_extended_staple_dagger(mStaple2, gField, y, mu, arg.gauge.c1);
					mTemp = gField.get_elems_const(y)[mu] * mStaple2 - mStaple1 * gx[mu];
					break;
				}
				case 100: mTemp.ZeroMatrix(); break;
			
			// 	test case start
			// 	case 100: {
			// 		get_staple_dagger(mStaple1, gField, x, mu);
			// 		mTemp = mStaple1 * gField.get_elems_const(x)[mu] * -1.;
			// 		break;
			// 	} 
		 	// 	test case end
		
			 	default: assert(false);
				}
		
				mTemp.TrLessAntiHermMatrix(); 
				fx[mu] = mTemp * qlat::Complex(0., arg.beta / 3.);
		}}
	
	}
}

inline void evolve_momentum(Field<cps::Matrix> &mField, 
				const Field<cps::Matrix> &fField, double dt, 
				const Arg_chmc &arg){
	TIMER("evolve_momentum()");
	assert(is_matching_geo(mField.geo, fField.geo));
#pragma omp parallel for
	for(long index = 0; index < mField.geo.local_volume(); index++){
		Coordinate x = mField.geo.coordinate_from_index(index);
		const qlat::Vector<cps::Matrix> fx = fField.get_elems_const(x);
			  qlat::Vector<cps::Matrix> mx = mField.get_elems(x);
		for(int mu = 0; mu < mField.geo.multiplicity; mu++){
			mx[mu] += fx[mu] * dt;
	}}
}

inline void evolve_gauge_field(Field<cps::Matrix> &gField, 
				const Field<cps::Matrix> &mField, double dt, 
				const Arg_chmc &arg){
	TIMER("evolve_gauge_field()");
	assert(is_matching_geo(mField.geo, gField.geo));
#pragma omp parallel for
	for(long index = 0; index < gField.geo.local_volume(); index++){
		Coordinate x = gField.geo.coordinate_from_index(index);
		cps::Matrix mL, mR;
		const qlat::Vector<cps::Matrix> mx = mField.get_elems_const(x);
			  qlat::Vector<cps::Matrix> gx = gField.get_elems(x);
		for(int mu = 0; mu < gField.geo.multiplicity; mu++){
		// only works for cps::Matrix
			Coordinate y(x); y[mu]--;
			switch(is_constrained(x, mu, arg.mag)){
			case 0: {
				algebra_to_group(mL, mx[mu] * dt);
				gx[mu] = mL * gx[mu];
				break;
			}
			// case 100: // test case
			case 1: {
				algebra_to_group(mL, mField.get_elems_const(y)[mu] * dt);
				algebra_to_group(mR, mx[mu] * -dt);
				gx[mu] = mL * gx[mu] * mR;
				break;
			}
			case 10: {
				algebra_to_group(mR, mx[mu] * -dt);
				gx[mu] = gx[mu] * mR;
			break;
			}
			case 100: {
				algebra_to_group(mL, mField.get_elems_const(y)[mu] * dt);
				gx[mu] = mL * gx[mu];
				break;
			}
			default: assert(false);
			}
	}}
}

inline void force_gradient_integrator(Field<cps::Matrix> &gField, Field<cps::Matrix> &mField, 
					const Arg_chmc &arg, Chart<cps::Matrix> &chart){
	// now this CANNOT be used in a multigrid algorithm
	TIMER("force_gradient_integrator()"); 

	assert(is_matching_geo(gField.geo, mField.geo));
	const double alpha = (3. - sqrt(3.)) * arg.dt / 6.;
	const double beta = arg.dt / sqrt(3.);
	const double gamma = (2. - sqrt(3.)) * arg.dt * arg.dt / 12.;
	
	static Field<cps::Matrix> gFieldAuxil; gFieldAuxil.init(gField.geo);
	static Field<cps::Matrix> fField; fField.init(mField.geo);

	evolve_gauge_field(gField, mField, alpha, arg);
	
	for(int i = 0; i < arg.trajectory_length; i++){
		fetch_expanded_chart(gField, chart);
		get_force(fField, gField, arg);
		gFieldAuxil = gField;
		evolve_gauge_field(gFieldAuxil, fField, gamma, arg);
		fetch_expanded_chart(gFieldAuxil, chart);
		get_force(fField, gFieldAuxil, arg);
		evolve_momentum(mField, fField, 0.5 * arg.dt, arg);

		evolve_gauge_field(gField, mField, beta, arg);
	
		fetch_expanded_chart(gField, chart);
		get_force(fField, gField, arg);
		gFieldAuxil = gField;
		evolve_gauge_field(gFieldAuxil, fField, gamma, arg);
		fetch_expanded_chart(gFieldAuxil, chart);
		get_force(fField, gFieldAuxil, arg);
		evolve_momentum(mField, fField, 0.5 * arg.dt, arg);

		if(i < arg.trajectory_length - 1) 
			evolve_gauge_field(gField, mField, 2. * alpha, arg);
		else evolve_gauge_field(gField, mField, alpha, arg);
	}
	qlat::Printf("reunitarize: max deviation = %.8e\n", reunitarize(gField));
}

inline void force_gradient_integrator(Field<cps::Matrix> &gField, Field<cps::Matrix> &mField, 
										Field<cps::Matrix> &gFieldAuxil, Field<cps::Matrix> &fField,
										const Arg_chmc &arg, Chart<cps::Matrix> &chart){
	// now this CANNOT be used in a multigrid algorithm
	sync_node();
	TIMER("force_gradient_integrator()"); 

	assert(is_matching_geo(gField.geo, mField.geo));
	assert(is_matching_geo(gField.geo, fField.geo));
	assert(is_matching_geo(gField.geo, gFieldAuxil.geo));
	const double alpha = (3. - sqrt(3.)) * arg.dt / 6.;
	const double beta = arg.dt / sqrt(3.);
	const double gamma = (2. - sqrt(3.)) * arg.dt * arg.dt / 12.;

	evolve_gauge_field(gField, mField, alpha, arg);


	sync_node();
	for(int i = 0; i < arg.trajectory_length; i++){
		fetch_expanded_chart(gField, chart);
		get_force(fField, gField, arg);
		gFieldAuxil = gField;
		evolve_gauge_field(gFieldAuxil, fField, gamma, arg);
		fetch_expanded_chart(gFieldAuxil, chart);
		get_force(fField, gFieldAuxil, arg);
		evolve_momentum(mField, fField, 0.5 * arg.dt, arg);

		evolve_gauge_field(gField, mField, beta, arg);
	
		fetch_expanded_chart(gField, chart);
		get_force(fField, gField, arg);
		gFieldAuxil = gField;
		evolve_gauge_field(gFieldAuxil, fField, gamma, arg);
		fetch_expanded_chart(gFieldAuxil, chart);
		get_force(fField, gFieldAuxil, arg);
		evolve_momentum(mField, fField, 0.5 * arg.dt, arg);

		if(i < arg.trajectory_length - 1) 
			evolve_gauge_field(gField, mField, 2. * alpha, arg);
		else evolve_gauge_field(gField, mField, alpha, arg);
	}
	qlat::Printf("reunitarize: max deviation = %.8e\n", reunitarize(gField));
}

inline void leap_frog_integrator(Field<cps::Matrix> &gField, Field<cps::Matrix> &mField, 
				const Arg_chmc &arg, Chart<cps::Matrix> chart){
	// not for multigrid

	TIMER("leap_frog_integrator()");
	assert(is_matching_geo(gField.geo, mField.geo));
	Geometry geo_; 
	geo_.init(gField.geo.geon, gField.geo.multiplicity, gField.geo.node_site);
	static Field<cps::Matrix> fField; fField.init(geo_);
	evolve_gauge_field(gField, mField, arg.dt / 2., arg);
	for(int i = 0; i < arg.trajectory_length; i++){
		fetch_expanded_chart(gField, chart);
		get_force(fField, gField, arg);
		evolve_momentum(mField, fField, arg.dt, arg);
		if(i < arg.trajectory_length - 1) 
			evolve_gauge_field(gField, mField, arg.dt, arg);
		else evolve_gauge_field(gField, mField, arg.dt / 2., arg);
	}
}

inline double get_hamiltonian(Field<cps::Matrix> &gField, const Field<cps::Matrix> &mField,
				const Arg_chmc &arg, Chart<cps::Matrix> &chart, vector<double> &part){
	
	TIMER("get_hamiltonian()");
	double localSum = 0.; // local sum of tr(\pi*\pi^\dagger)
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
		Coordinate x = mField.geo.coordinate_from_index(index);
		const qlat::Vector<cps::Matrix> mx = mField.get_elems_const(x);
		for(int mu = 0; mu < DIMN; mu++){
			switch(is_constrained(x, mu, arg.mag)){
				case 100: break;
				// case 100: // test case
				case 0:
				case 1:
				case 10:{
					pLocalSum += (mx[mu] * mx[mu]).ReTr();
					break;
				}
				default: assert(false);
			}
	}}
	ppLocalSum[omp_get_thread_num()] = pLocalSum;
}
	for(int i = 0; i < numThreads; i++){
		localSum += ppLocalSum[i];
	}

	double globalSum;
	MPI_Allreduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, get_comm());
	double kineticEnergy = globalSum / 2.;
	fetch_expanded_chart(gField, chart);
	double potential_energy = 0.;
	if(arg.gauge.type == qlat::WILSON){
		potential_energy = -total_plaq(gField) * arg.beta / 3.;
	}
	if(arg.gauge.type == IWASAKI){
		double p1 = -total_plaq(gField);
		double p2 = -total_rectangular(gField);
		potential_energy = (p1 * (1. - 8. * arg.gauge.c1) + p2 * arg.gauge.c1) 
																* arg.beta / 3.;
	}
	part.resize(2);
	part[0] = kineticEnergy; part[1] = potential_energy;
	return kineticEnergy + potential_energy;
}

inline void init_momentum(Field<cps::Matrix> &mField){
	TIMER("init_momentum()");

	using namespace qlat;
	static bool initialized = false;
	static Geometry rng_geo;
	static RngField rng_field;
	if(initialized == false){
		rng_geo.init(mField.geo.geon, 1, mField.geo.node_site);
		rng_field.init(rng_geo, RngState("Ich liebe dich."));
		initialized = true;
	}

#pragma omp parallel for
	for(long index = 0; index < mField.geo.local_volume(); index++){
		Coordinate x = mField.geo.coordinate_from_index(index);
		qlat::Vector<cps::Matrix> mx = mField.get_elems(x);
		cps::Matrix mTemp;
		for(int mu = 0; mu < mField.geo.multiplicity; mu++){
			mTemp.ZeroMatrix();
			for(int a = 0; a < SU3_NUM_OF_GENERATORS; a++){
				mTemp += su3_generators[a] * g_rand_gen(rng_field.get_elem(x));
			}
			mx[mu] = mTemp;
	}}
}

inline void init_momentum(Field<cps::Matrix> &mField, RngField &rng_field){
	TIMER("init_momentum()");
	using namespace qlat;

#pragma omp parallel for
	for(long index = 0; index < mField.geo.local_volume(); index++){
		Coordinate x = mField.geo.coordinate_from_index(index);
		qlat::Vector<cps::Matrix> mx = mField.get_elems(x);
		cps::Matrix mTemp;
		for(int mu = 0; mu < mField.geo.multiplicity; mu++){
			mTemp.ZeroMatrix();
			for(int a = 0; a < SU3_NUM_OF_GENERATORS; a++){
				mTemp += su3_generators[a] * g_rand_gen(rng_field.get_elem(x));
			}
			mx[mu] = mTemp;
	}}
}

inline double derivative(const Field<cps::Matrix> &gField, const Coordinate &x, int mu, int a){
	cps::Matrix &U = gField.get_elems_const(x)[mu];
	cps::Matrix V_dagger; get_staple_dagger(V_dagger, gField, x, mu);
	cps::Matrix temp = su3_generators[a] * U * V_dagger * qlat::Complex(0., 1.);
	return temp.ReTr();
}

inline double derivative_Iwasaki(const Field<cps::Matrix> &gField, 
									const Coordinate &x, int mu, int a, const Arg_chmc &arg){
	double c1 = arg.gauge.c1;
	double c0 = 1. - 8. * c1;
	cps::Matrix &u = gField.get_elems_const(x)[mu];
	cps::Matrix stp_dagger; get_staple_dagger(stp_dagger, gField, x, mu);
	cps::Matrix rtg_dagger; get_rectangular_dagger(rtg_dagger, gField, x, mu);
	cps::Matrix tot_dagger = stp_dagger * c0 + rtg_dagger * c1;
	cps::Matrix temp = su3_generators[a] * u * tot_dagger * qlat::Complex(0., 1.);
	return temp.ReTr();
}

inline double derivative_pair(const Field<cps::Matrix> &gField, const Coordinate &x, int mu, int b){
	Coordinate y = x; y[mu]++;
	cps::Matrix &U1 = gField.get_elems_const(x)[mu];								
	cps::Matrix &U2 = gField.get_elems_const(y)[mu];
	cps::Matrix V1_dagger; get_staple_dagger(V1_dagger, gField, x, mu);
	cps::Matrix V2_dagger; get_staple_dagger(V2_dagger, gField, y, mu);
	cps::Matrix temp1 = su3_generators[b] * V1_dagger * U1 * qlat::Complex(0., -1.);
	cps::Matrix temp2 = su3_generators[b] * U2 * V2_dagger * qlat::Complex(0., 1.);
	return (temp1 + temp2).ReTr();
}

inline void derivative_field(Field<double> &dField, Field<cps::Matrix> &gField, 
								const Arg_chmc &arg, bool does_pair = false){
	for(int i = 0; i < DIMN; i++){
		assert(gField.geo.node_site[i] % dField.geo.node_site[i] == 0); 
	}
	fetch_expanded(gField);
#pragma omp parallel for
	for(long index = 0; index < dField.geo.local_volume(); index++){
		Coordinate x = dField.geo.coordinate_from_index(index);
		int spin_color_index;
		qlat::Vector<double> dx = dField.get_elems(x);
		for(int mu = 0; mu < DIMN; mu++){
		for(int a = 0; a < SU3_NUM_OF_GENERATORS; a++){
			spin_color_index = mu * SU3_NUM_OF_GENERATORS + a;
			if(does_pair)
				dx[spin_color_index] = derivative_Iwasaki(gField, arg.mag * x, mu, a, arg)
									+ derivative_pair(gField, arg.mag * x, mu, (a+3)%8);
			else
				dx[spin_color_index] = derivative_Iwasaki(gField, arg.mag * x, mu, a, arg);
		}}
	}
}

inline double derivative_sum(Field<cps::Matrix> &gField, const Arg_chmc &arg){
	fetch_expanded(gField);
	double local_sum = 0.;
	long count;
	for(int x = 0; x < gField.geo.node_site[0]; x += arg.mag){
	for(int y = 0; y < gField.geo.node_site[1]; y += arg.mag){
	for(int z = 0; z < gField.geo.node_site[2]; z += arg.mag){
	for(int t = 0; t < gField.geo.node_site[3]; t += arg.mag){
	for(int mu = 0; mu < DIMN; mu++){
	for(int a = 0; a < 8; a++){
		Coordinate coor(x, y, z, t);
		local_sum += derivative(gField, coor, mu, a);
		count++;
	}}}}}}

	double global_sum;
	MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, get_comm());

	Printf("derivative_sum(): mag = %d\tcount = %d", arg.mag, count); 
	return global_sum;
}

inline void run_chmc(Field<cps::Matrix> &gFieldExt, const Arg_chmc &arg, FILE *pFile){
	TIMER("run_chmc()");
	if(!get_id_node()) assert(pFile != NULL);
	assert(arg.num_trajectory > 20);

	RngState globalRngState("By the witness of the martyrs.");

	Coordinate expansion(2, 2, 2, 2);
	Geometry geoExpand1 = gFieldExt.geo; geoExpand1.resize(expansion, expansion);
	Geometry geoLocal = gFieldExt.geo;
	Field<cps::Matrix> gField; gField.init(geoExpand1); gField = gFieldExt;
	Field<cps::Matrix> mField; mField.init(geoLocal);

	Chart<cps::Matrix> chart;
	produce_chart_envelope(chart, gFieldExt.geo, arg.gauge);

	Coordinate total_size_coarse;
	for(int i = 0; i < DIMN; i++){
		total_size_coarse[i] = geoLocal.total_site()[i] / arg.mag;
	}
	Geometry geo_coarse; 
	geo_coarse.init(total_size_coarse, DIMN * SU3_NUM_OF_GENERATORS);
	Field<double> dField; dField.init(geo_coarse);

//	long alpha_size = product(gFieldExt.geo.node_site) * DIMN * SU3_NUM_OF_GENERATORS;
//	vector<vector<double> > dev_list; dev_list.resize(alpha_size);

	double oldH, newH;
	double dieRoll;
	double deltaH, percentDeltaH;
	double acceptProbability;
	double avgPlaq;
	vector<double> energy_partition_old, energy_partition_new;
	bool doesAccept;
	int numAccept = 0, numReject = 0;

	for(int i = 0; i < arg.num_trajectory; i++){
		
		qlat::Printf("---------- START OF TRAJECTORY %5d --------\n", i+1);
		
		init_momentum(mField);
		
		oldH = get_hamiltonian(gField, mField, arg, chart, energy_partition_old);
		// leapFrogIntegrator(gField, mField, arg);
		force_gradient_integrator(gField, mField, arg, chart);
		
		newH = get_hamiltonian(gField, mField, arg, chart, energy_partition_new);
	
		dieRoll = u_rand_gen(globalRngState);
		deltaH = newH - oldH;
		percentDeltaH = deltaH / oldH;
		acceptProbability = std::exp(oldH - newH);
		doesAccept = (dieRoll < acceptProbability);
		MPI_Bcast((void *)&doesAccept, 1, MPI_BYTE, 0, get_comm());
		// make sure that all the node make the same decision.
		
		if(i < arg.num_forced_accept_step){
//			report << "End trajectory " << i + 1
//				<< ":\tFORCE ACCEPT." << endl;
			gFieldExt = gField;
			doesAccept = true;
		}else{
			if(doesAccept){
//				report << "End trajectory " << i + 1
//					<< ":\tACCEPT." << endl;
				numAccept++;
				gFieldExt = gField;
			}else{
//				report << "End trajectory " << i + 1
//					<< ":\tREJECT." << endl;
				numReject++;
				gField = gFieldExt;	
			}	
		}
	
		qlat::Printf("---------- END OF TRAJECTORY %5d ----------\n", i+1);
		qlat::Printf("Old Hamiltonian =\t%+.12e\n", oldH);
		qlat::Printf("New Hamiltonian =\t%+.12e\n", newH);
		qlat::Printf("Delta H         =\t%+.12e\n", deltaH); 
		qlat::Printf("Delta H Ratio   =\t%+.12e\n", percentDeltaH); 
		qlat::Printf("exp(-Delta H)   =\t%12.6f\n", acceptProbability);
		qlat::Printf("Die Roll        =\t%12.6f\n", dieRoll); 	
	
		fetch_expanded_chart(gField, chart);
		avgPlaq = avg_plaquette(gField);
		qlat::Printf("Avg Plaquette   =\t%+.12e\n", avgPlaq); 
		qlat::Printf("ACCEPT RATE     =\t%+.4f\n", 
						(double)numAccept / (numAccept + numReject));	
//		derivative_list(dev_list, gField, arg);	
//		double dv_sum = derivative_sum(gField, arg);
//		report << "FINE DERIVATIVE SUM =\t" << dv_sum << endl;

		Fprintf(pFile, "%i\t%.6e\t%.6e\t%.12e\t%i\t%.12e\n", 
				i + 1, abs(deltaH), acceptProbability, avgPlaq, doesAccept, 
				doesAccept?energy_partition_new[1]:energy_partition_old[1]);
		Fflush(pFile);

		if((i + 1) % arg.num_step_between_output == 0 
											&& i + 1 >= arg.num_step_before_output){
			Arg_export arg_export;
			arg_export.beta = arg.beta;
			arg_export.sequence_num = i + 1;
			arg_export.ensemble_label = "constrained_hmc";
			if(arg.export_dir_stem.size() > 0){
				string address = arg.export_dir_stem + "ckpoint." + show(i + 1);
				export_config_nersc(gFieldExt, address, arg_export, true);
			}
			
			sync_node();
	
			if(arg.summary_dir_stem.size() > 0){
			// TODO: Test if doing pair would change the result.
			//	derivative_field(dField, gField, arg, true);
				derivative_field(dField, gField, arg, false);
				Field<double> dField_output; dField_output.init(geo_coarse);
				sophisticated_make_to_order(dField_output, dField);
				sophisticated_serial_write(dField_output, arg.summary_dir_stem + "./dev_dump." + show(i + 1));
			}
		}
	}

//	double ATC;
//	vector<double> dev_val; dev_val.resize(alpha);
//	vector<double> dev_err; dev_err.resize(alpha);
//	for(long j = 0; j < alpha; j++){
//		ATC = autoCorrelation(dev_list[j]);
//		dev_err[j] = jackknife(dev_list[j].data(), dev_list[j].size(), 
//									int(ceil(ATC)), dev_val[j]);
//	}

	Fprintf(pFile, "Accept Rate = %.4f\n", (double)numAccept / (numAccept + numReject));
	Fflush(pFile);

	Timer::display();
}

inline void update_field(Field<cps::Matrix> &gField_ext, Field<cps::Matrix> &gField, Field<cps::Matrix> &mField,
						Field<cps::Matrix> &gFieldAuxil, Field<cps::Matrix> &fField, RngField &rng_field, 
						const Arg_chmc &arg, Chart<cps::Matrix> &chart, FILE *p_summary, int count, 
						RngState &globalRngState){

	static double old_hamiltonian;
	static double new_hamiltonian;
	static double die_roll;
	static double del_hamiltonian;
	static double accept_probability;
	static double average_plaquette;

	static vector<double> old_energy_partition;
	static vector<double> new_energy_partition;
	static bool does_accept;
	static int num_accept = 0;
	static int num_reject = 0;

	init_momentum(mField, rng_field);

	old_hamiltonian = get_hamiltonian(gField, mField, arg, chart, old_energy_partition);
	force_gradient_integrator(gField, mField, gFieldAuxil, fField, arg, chart);
	new_hamiltonian = get_hamiltonian(gField, mField, arg, chart, new_energy_partition);

	die_roll = u_rand_gen(globalRngState);
	del_hamiltonian = new_hamiltonian - old_hamiltonian;
	accept_probability = std::exp(old_hamiltonian - new_hamiltonian);
	
	does_accept = die_roll < accept_probability;
	// make sure that all the node make the same decision.
	MPI_Bcast((void *)&does_accept, 1, MPI_BYTE, 0, get_comm());
	
	if(count < arg.num_forced_accept_step){
		qlat::Printf("End Trajectory %d:\t FORCE ACCEPT.\n", count + 1);
		gField_ext = gField;
		does_accept = true;
	}else{
		if(does_accept){
			qlat::Printf("End Trajectory %d:\t ACCEPT.\n", count + 1);
			num_accept++;
			gField_ext = gField;
		}else{
			qlat::Printf("End Trajectory %d:\t REJECT.\n", count + 1);
			num_reject++;
			gField = gField_ext;	
		}	
	}
	
	qlat::Printf("Old Hamiltonian =\t%+.12e\n", old_hamiltonian);
	qlat::Printf("New Hamiltonian =\t%+.12e\n", new_hamiltonian);
	qlat::Printf("Del Hamiltonian =\t%+.12e\n", del_hamiltonian); 
	qlat::Printf("exp(-Delta H)   =\t%12.6f\n", accept_probability);
	qlat::Printf("Die Roll        =\t%12.6f\n", die_roll); 	

	fetch_expanded_chart(gField, chart);
	average_plaquette = avg_plaquette(gField);
	qlat::Printf("Avg Plaquette   =\t%+.12e\n", average_plaquette); 
	qlat::Printf("ACCEPT RATE     =\t%+.4f\n", (double)num_accept / (num_accept + num_reject));	

	if(arg.summary_dir_stem.size() > 0){
		Fprintf(p_summary, "%i\t%.6e\t%.6e\t%.12e\t%i\t%.12e\n", 
				count + 1, abs(del_hamiltonian), accept_probability, average_plaquette, does_accept, 
				does_accept?new_energy_partition[1]:old_energy_partition[1]);
		Fflush(p_summary);
	}

	if((count + 1) % arg.num_step_between_output == 0 && count + 1 >= arg.num_step_before_output){
		Arg_export arg_export;
		arg_export.beta = 			arg.beta;
		arg_export.sequence_num = 	count + 1;
		arg_export.ensemble_label = "double_multigrid";
		
		if(arg.export_dir_stem.size() > 0){
			string address = arg.export_dir_stem + "ckpoint." + show(count + 1);
			export_config_nersc(gField_ext, address, arg_export, true);
		}
		
		sync_node();
	}

}

inline void update_constrain(Field<cps::Matrix> &gField, const Field<cps::Matrix> &gField_coarse, 
								const Arg_chmc &arg){
	assert(gField.geo.total_site() == arg.mag * gField_coarse.geo.total_site());
	
	sync_node();
#pragma omp parallel for
    for(long index = 0; index < gField_coarse.geo.local_volume(); index++){
		Coordinate xC = gField_coarse.geo.coordinate_from_index(index);
		Coordinate x = arg.mag * xC;
		qlat::Vector<cps::Matrix> pC = gField_coarse.get_elems_const(xC);
		qlat::Vector<cps::Matrix> p = gField.get_elems(x);
		for(int mu = 0; mu < gField_coarse.geo.multiplicity; mu++){
			Coordinate xP = x; xP[mu]++;
			vector<int> cotta(arg.mag - 1, mu);
			cps::Matrix m;
			get_path_ordered_product(m, gField, xP, cotta);
			cps::Matrix d; d.Dagger(m);
			p[mu] = pC[mu] * d;
    }}
    sync_node();
    qlat::Printf("Coarse lattice initialized.\n");
}

inline void double_multigrid(Field<cps::Matrix> &gField_ext, const Arg_chmc &arg, 
														const Arg_chmc &arg_coarse){
	
	// perform a number of fine updates and them a number of coarse updates.
	
	TIMER("double_multigrid");
	
	FILE *p = NULL;
	if(arg.summary_dir_stem.size() > 0){
		p = Fopen((arg.summary_dir_stem + "/summary.dat").c_str(), "a");
	}

	if(!get_id_node()) assert(p != NULL);

	time_t now = time(NULL);
	Fprintf(p, "# %s", ctime(&now));
	Fprintf(p, "# %s\n", show(gField_ext.geo).c_str());
	Fprintf(p, "# mag               = %i\n", arg.mag);
	Fprintf(p, "# trajectory_length = %i\n", arg.trajectory_length);
	Fprintf(p, "# num_trajectory    = %i\n", arg.num_trajectory);
	Fprintf(p, "# beta              = %.6f\n", arg.beta);
	Fprintf(p, "# dt                = %.5f\n", arg.dt);
	Fprintf(p, "# c1                = %.5f\n", arg.gauge.c1);
	Fprintf(p, "# GAUGE_TYPE        = %d\n", arg.gauge.type);
	Fprintf(p, "# traj. number\texp(-DeltaH)\tavgPlaq\taccept/reject\n");
	Fflush(p);
	qlat::Printf("p opened.");

	assert(arg.num_trajectory > 20);
	RngState globalRngState("By the witness of the martyrs.");

	Coordinate expansion(2, 2, 2, 2);

	// declare fine lattice variables
	Geometry geo_expanded = gField_ext.geo; geo_expanded.resize(expansion, expansion);
	Geometry geo_local = gField_ext.geo;
	Field<cps::Matrix> gField; gField.init(geo_expanded); gField = gField_ext;
	Field<cps::Matrix> gField_auxil; gField_auxil.init(geo_expanded);
	Field<cps::Matrix> mField; mField.init(geo_expanded);
	Field<cps::Matrix> fField; fField.init(geo_expanded);

	Geometry rng_geo; 
	rng_geo.init(mField.geo.geon, 1, mField.geo.node_site);
	RngField rng_field; 
	rng_field.init(rng_geo, RngState("Ich liebe dich."));

	//declare coarse lattice variables
	Coordinate total_size_coarse;
	for(int i = 0; i < DIMN; i++){
		total_size_coarse[i] = geo_local.total_site()[i] / arg.mag;
	}

	Geometry geo_local_coarse; geo_local_coarse.init(total_size_coarse, DIMN);
	Geometry geo_expanded_coarse = geo_local_coarse; geo_expanded_coarse.resize(expansion, expansion);
	Field<cps::Matrix> gField_coarse; gField_coarse.init(geo_expanded_coarse);
	Field<cps::Matrix> gField_ext_coarse; gField_ext_coarse.init(geo_expanded_coarse);
	Field<cps::Matrix> gField_auxil_coarse; gField_auxil_coarse.init(geo_expanded_coarse);
	Field<cps::Matrix> mField_coarse; mField_coarse.init(geo_expanded_coarse);
	Field<cps::Matrix> fField_coarse; fField_coarse.init(geo_expanded_coarse);
	
	// initialize the coarse field.
	sync_node();
#pragma omp parallel for
    for(long index = 0; index < geo_local_coarse.local_volume(); index++){
		for(int mu = 0; mu < geo_local_coarse.multiplicity; mu++){
			Coordinate xCoarse = geo_local_coarse.coordinate_from_index(index);
			Coordinate x = arg.mag * xCoarse;
			vector<int> cotta(arg.mag, mu);
			get_path_ordered_product(gField_coarse.get_elems(xCoarse)[mu], gField, x, cotta);
    }}
    sync_node();
    qlat::Printf("Coarse lattice initialized.\n");

	gField_ext_coarse = gField_coarse;

	fetch_expanded(gField_coarse);
	fetch_expanded(gField);
    qlat::Printf("CONSTRAINED Plaquette = %.12f\n", check_constrained_plaquette(gField, arg.mag));
    qlat::Printf("COARSE      Plaquette = %.12f\n", avg_plaquette(gField_coarse));	

	Geometry rng_geo_coarse; 
	rng_geo_coarse.init(mField_coarse.geo.geon, 1, mField_coarse.geo.node_site);
	RngField rng_field_coarse; 
	rng_field_coarse.init(rng_geo_coarse, RngState("Tut mir leid."));

	// declare the communication patterns
	Chart<cps::Matrix> chart;
	produce_chart_envelope(chart, gField_ext.geo, arg.gauge);
	Chart<cps::Matrix> chart_coarse;
	produce_chart_envelope(chart_coarse, gField_coarse.geo, arg_coarse.gauge);

	// start the HMC 
//	double old_hamiltonian;
//	double new_hamiltonian;
//	double die_roll;
//	double del_hamiltonian;
//	double accept_probability;
//	double average_plaquette;
//	
//	vector<double> old_energy_partition;
//	vector<double> new_energy_partition;
//	bool does_accept;
//	int num_accept = 0;
//	int num_reject = 0;
//
//	double old_hamiltonian_coarse;
//	double new_hamiltonian_coarse;
//	double die_roll_coarse;
//	double del_hamiltonian_coarse;
//	double accept_probability_coarse;
//	double average_plaquette_coarse;
//	
//	vector<double> old_energy_partition_coarse;
//	vector<double> new_energy_partition_coarse;
//	bool does_accept_coarse;
//	int num_accept_coarse = 0;
//	int num_reject_coarse = 0;
//
//	// first update the fine lattice
//	for(int i = 0; i < arg.num_trajectory; i++){
//		
//		// start fine lattice update
//
//		init_momentum(mField);
//		
//		old_hamiltonian = get_hamiltonian(gField, mField, arg, chart, old_energy_partition);
//		force_gradient_integrator(gField, mField, arg, chart);
//		new_hamiltonian = get_hamiltonian(gField, mField, arg, chart, new_energy_partition);
//	
//		die_roll = u_rand_gen(globalRngState);
//		del_hamiltonian = new_hamiltonian - old_hamiltonian;
//		accept_probability = exp(old_hamiltonian - new_hamiltonian);
//		
//		does_accept = die_roll < accept_probability;
//		// make sure that all the node make the same decision.
//		MPI_Bcast((void *)&does_accept, 1, MPI_BYTE, 0, get_comm());
//		
//		if(i < arg.num_forced_accept_step){
//			qlat::Printf("End Trajectory %d:\t FORCE ACCEPT.\n", i + 1);
//			gField_ext = gField;
//			does_accept = true;
//		}else{
//			if(does_accept){
//				qlat::Printf("End Trajectory %d:\t ACCEPT.\n", i + 1);
//				num_accept++;
//				gField_ext = gField;
//			}else{
//				qlat::Printf("End Trajectory %d:\t REJECT.\n", i + 1);
//				num_reject++;
//				gField = gField_ext;	
//			}	
//		}
//		
//		qlat::Printf("Old Hamiltonian =\t%+.12e\n", old_hamiltonian);
//		qlat::Printf("New Hamiltonian =\t%+.12e\n", new_hamiltonian);
//		qlat::Printf("Delta H         =\t%+.12e\n", del_hamiltonian); 
//		qlat::Printf("exp(-Delta H)   =\t%12.6f\n", accept_probability);
//		qlat::Printf("Die Roll        =\t%12.6f\n", die_roll); 	
//	
//		fetch_expanded_chart(gField, chart);
//		average_plaquette = avg_plaquette(gField);
//		qlat::Printf("Avg Plaquette   =\t%+.12e\n", avgPlaq); 
//		qlat::Printf("ACCEPT RATE     =\t%+.4f\n", (double)num_accept / (num_accept + num_reject));	
//
//		if(arg.summary_dir_stem.size() > 0){
//			Fprintf(p, "%i\t%.6e\t%.6e\t%.12e\t%i\t%.12e\n", 
//					i + 1, abs(deltaH), accept_probability, average_plaquette, does_accept, 
//					does_accept?new_energy_partition_new[1]:old_energy_partition_old[1]);
//			Fflush(p);
//		}
//
//		if((i + 1) % arg.num_step_between_output == 0 && i + 1 >= arg.num_step_before_output){
//			Arg_export arg_export;
//			arg_export.beta = 			arg.beta;
//			arg_export.sequence_num = 	i + 1;
//			arg_export.ensemble_label = "double_multigrid";
//			
//			if(arg.export_dir_stem.size() > 0){
//				string address = arg.export_dir_stem + "ckpoint." + show(i + 1);
//				export_config_nersc(gFieldExt, address, arg_export, true);
//			}
//			
//			sync_node();
//		}
	for(int i = 0; i < arg.num_trajectory; i++){
		
		// start fine lattice update

		for(int r = 0; r < 10; r++){
			qlat::Printf("Trajectory %d - %d:\n", i + 1, r + 1);
			update_field(gField_ext, gField, mField, gField_auxil, fField, rng_field, 
						arg, chart, p, i, globalRngState);
		}
		qlat::Printf("Trajectory %d - COARSE:\n", i + 1);
		update_field(gField_ext_coarse, gField_coarse, mField_coarse, gField_auxil_coarse, 
						fField_coarse, rng_field_coarse, arg_coarse, chart_coarse, 
						p, i, globalRngState);

		update_constrain(gField, gField_coarse, arg);
		gField_ext = gField;

		fetch_expanded_chart(gField_coarse, chart_coarse);
		fetch_expanded_chart(gField, chart);
   		qlat::Printf("CONSTRAINED Plaquette = %.12f\n", check_constrained_plaquette(gField, arg.mag));
		qlat::Printf("COARSE      Plaquette = %.12f\n", avg_plaquette(gField_coarse));
		// end fine lattice update
		// start coarse lattice update
		// end coarse lattice update

	}

	Timer::display();
}

QLAT_END_NAMESPACE
