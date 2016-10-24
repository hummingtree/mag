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

#include "field-matrix.h"

using namespace cps;
using namespace qlat;
using namespace std;

#define SU3_NUM_OF_GENERATORS 8

static const double inv_sqrt2 = 1. / sqrt(2.);

inline vector<Matrix> init_generators(){
	Matrix T1, T2, T3, T4, T5, T6, T7, T8;
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
	
	vector<Matrix> ret {T1, T2, T3, T4, T5, T6, T7, T8};
	
	return ret;
}

static const vector<Matrix> su3_generators = init_generators();

inline void exp(Matrix &expM, const Matrix &M){
        Matrix mTemp2 = M, mTemp3;
	for(int i = 9; i > 1; i--){
		mTemp3.OneMinusfTimesM(-1. / i, mTemp2);
		mTemp2.DotMEqual(M, mTemp3);
	}
	expM.OneMinusfTimesM(-1., mTemp2);
}

inline void algebra_to_group(Matrix &expiM, const Matrix &M){
	// expiM = exp(i * M)
	Matrix mTemp = M; mTemp *= qlat::Complex(0., 1.);
	exp(expiM, mTemp);
}

inline void get_staple_dagger(Matrix &staple, const Field<Matrix> &field, 
					const Coordinate &x, const int mu){
	vector<int> dir;
	Matrix staple_; staple_.ZeroMatrix();
	Matrix m;
	for(int nu = 0; nu < DIM; nu++){
		if(mu == nu) continue;
		dir.clear();
		dir.push_back(nu); dir.push_back(mu); dir.push_back(nu + DIM);
		get_path_ordered_product(m, field, x, dir);
		staple_ += m;
		dir.clear();
		dir.push_back(nu + DIM); dir.push_back(mu); dir.push_back(nu);
		get_path_ordered_product(m, field, x, dir);
		staple_ += m;
	}
	staple.Dagger(staple_);
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
		Coordinate xl; geo.coordinate_from_index(xl, index);
		RngState& rs = rf.get_elem(xl);
		for (int i = chunk * index; i < chunk * (index + 1); ++i){
			xs[i] = gRandGen(rs);
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

inline void get_force(Field<Matrix> &fField, const Field<Matrix> &gField,
			const Arg_chmc &arg){
	TIMER("getFoece()");
	assert(is_matching_geo(fField.geo, gField.geo));
#pragma omp parallel for
	for(long index = 0; index < fField.geo.local_volume(); index++){
		Coordinate x; 
		Matrix mStaple1, mStaple2, mTemp;
		fField.geo.coordinate_from_index(x, index);
		for(int mu = 0; mu < fField.geo.multiplicity; mu++){
			switch(is_constrained(x, mu, arg.mag)){
			case 0: {
				get_staple_dagger(mStaple1, gField, x, mu);
				mTemp = gField.get_elems_const(x)[mu] * mStaple1;
				break;
			}
			case 1:
			case 10: {
				Coordinate y(x); y[mu]++;
				get_staple_dagger(mStaple1, gField, x, mu);
				get_staple_dagger(mStaple2, gField, y, mu);
				mTemp = gField.get_elems_const(y)[mu] * mStaple2 \
					- mStaple1 * gField.get_elems_const(x)[mu];
				break;
			}
			case 100: mTemp.ZeroMatrix(); break;
		
			// test case start
		// 	case 100: {
		// 		get_staple_dagger(mStaple1, gField, x, mu);
		// 		mTemp = mStaple1 * gField.get_elems_const(x)[mu] * -1.;
		// 		break;
		// 	} 
		// 	// test case end
	
		 	default: assert(false);
			}
	
			mTemp.TrLessAntiHermMatrix(); 
			fField.get_elems(x)[mu] = \
				mTemp * qlat::Complex(0., arg.beta / 3.);
	}}
}

inline void evolve_momentum(Field<Matrix> &mField, 
				const Field<Matrix> &fField, double dt, 
				const Arg_chmc &arg){
	TIMER("algCHmcWilson::evolve_momentum()");
	assert(is_matching_geo(mField.geo, fField.geo));
#pragma omp parallel for
	for(long index = 0; index < mField.geo.local_volume(); index++){
		Coordinate x; 
		mField.geo.coordinate_from_index(x, index);
		for(int mu = 0; mu < mField.geo.multiplicity; mu++){
			mField.get_elems(x)[mu] += fField.get_elems_const(x)[mu] * dt;
	}}
}

inline void evolve_gauge_field(Field<Matrix> &gField, 
				const Field<Matrix> &mField, double dt, 
				const Arg_chmc &arg){
	TIMER("algCHmcWilson::evolve_gauge_field()");
	assert(is_matching_geo(mField.geo, gField.geo));
#pragma omp parallel for
	for(long index = 0; index < gField.geo.local_volume(); index++){
		Coordinate x; 
		gField.geo.coordinate_from_index(x, index);
		Matrix mL, mR;
		for(int mu = 0; mu < gField.geo.multiplicity; mu++){
		// only works for Matrix
			Matrix &U = gField.get_elems(x)[mu];
			Coordinate y(x); y[mu]--;
			switch(is_constrained(x, mu, arg.mag)){
			case 0: {
				algebra_to_group(mL, mField.get_elems_const(x)[mu] * dt);
				U = mL * U;
				break;
			}
			// case 100: // test case
			case 1: {
				algebra_to_group(mL, mField.get_elems_const(y)[mu] * dt);
				algebra_to_group(mR, mField.get_elems_const(x)[mu] * -dt);
				U = mL * U * mR;
				break;
			}
			case 10: {
				algebra_to_group(mR, mField.get_elems_const(x)[mu] * -dt);
				U = U * mR;
			break;
			}
			case 100: {
				algebra_to_group(mL, mField.get_elems_const(y)[mu] * dt);
				U = mL * U;
				break;
			}
			default: assert(false);
			}
	}}
}

inline void force_gradient_integrator(Field<Matrix> &gField, Field<Matrix> &mField, 
					const Arg_chmc &arg, Chart<Matrix> &chart){
    	TIMER("force_gradient_integrator()"); 

	assert(is_matching_geo(gField.geo, mField.geo));
	const double alpha = (3. - sqrt(3.)) * arg.dt / 6.;
	const double beta = arg.dt / sqrt(3.);
	const double gamma = (2. - sqrt(3.)) * arg.dt * arg.dt / 12.;
	
	static Field<Matrix> gFieldAuxil; gFieldAuxil.init(gField.geo);
	static Field<Matrix> fField; fField.init(mField.geo);

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
	report << "reunitarize: max deviation = " << reunitarize(gField) << endl;
}

inline void leap_frog_integrator(Field<Matrix> &gField, Field<Matrix> &mField, 
				const Arg_chmc &arg){
	TIMER("leap_frog_integrator()");
	assert(is_matching_geo(gField.geo, mField.geo));
	Geometry geo_; 
	geo_.init(gField.geo.geon, gField.geo.multiplicity, gField.geo.node_site);
	static Field<Matrix> fField; fField.init(geo_);
	evolve_gauge_field(gField, mField, arg.dt / 2., arg);
	for(int i = 0; i < arg.trajectory_length; i++){
		fetch_expanded(gField);
		get_force(fField, gField, arg);
		evolve_momentum(mField, fField, arg.dt, arg);
		if(i < arg.trajectory_length - 1) 
			evolve_gauge_field(gField, mField, arg.dt, arg);
		else evolve_gauge_field(gField, mField, arg.dt / 2., arg);
	}
}

inline double get_hamiltonian(Field<Matrix> &gField, const Field<Matrix> &mField,
				const Arg_chmc &arg){
	
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
		for(int mu = 0; mu < DIM; mu++){
			Coordinate x; mField.geo.coordinate_from_index(x, index);
			switch(is_constrained(x, mu, arg.mag)){
				case 100: break;
				// case 100: // test case
				case 0:
				case 1:
				case 10:{
					Matrix mTemp = mField.get_elems_const(x)[mu];
					pLocalSum += (mTemp * mTemp).ReTr();
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
	fetch_expanded(gField);
	double potentialEnergy = -total_plaq(gField) * arg.beta / 3.;
	return kineticEnergy + potentialEnergy;
}

inline void init_momentum(Field<Matrix> &mField){
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
		Coordinate x; mField.geo.coordinate_from_index(x, index);
		Matrix mTemp;
		for(int mu = 0; mu < mField.geo.multiplicity; mu++){
			mTemp.ZeroMatrix();
			for(int a = 0; a < SU3_NUM_OF_GENERATORS; a++){
				mTemp += su3_generators[a] * gRandGen(rng_field.get_elem(x));
			}
			mField.get_elems(x)[mu] = mTemp;
	}}
}

inline double derivative(const Field<Matrix> &gField, 
							const Coordinate &x, int mu, int a){
	Matrix &U = gField.get_elems_const(x)[mu];
	Matrix V_dagger; get_staple_dagger(V_dagger, gField, x, mu);
	Matrix temp = su3_generators[a] * U * V_dagger * qlat::Complex(0., 1.);
	return temp.ReTr();
}

inline void derivative_field(Field<double> &dField, Field<Matrix> &gField, 
								const Arg_chmc &arg){
	for(int i = 0; i < DIM; i++){
		assert(gField.geo.node_site[i] % dField.geo.node_site[i] == 0); 
	}
	fetch_expanded(gField);
#pragma omp parallel for
	for(long index = 0; index < dField.geo.local_volume(); index++){
		Coordinate x; dField.geo.coordinate_from_index(x, index);
		int spin_color_index;
		for(int mu = 0; mu < DIM; mu++){
		for(int a = 0; a < SU3_NUM_OF_GENERATORS; a++){
			spin_color_index = mu * SU3_NUM_OF_GENERATORS + a;
			dField.get_elems(x)[spin_color_index] = 
										derivative(gField, arg.mag * x, mu, a);
		}}
	}
}

inline double derivative_sum(Field<Matrix> &gField, const Arg_chmc &arg){
	fetch_expanded(gField);
	double local_sum = 0.;
	long count;
	for(int x = 0; x < gField.geo.node_site[0]; x += arg.mag){
	for(int y = 0; y < gField.geo.node_site[1]; y += arg.mag){
	for(int z = 0; z < gField.geo.node_site[2]; z += arg.mag){
	for(int t = 0; t < gField.geo.node_site[3]; t += arg.mag){
	for(int mu = 0; mu < DIM; mu++){
	for(int a = 0; a < 8; a++){
		Coordinate coor(x, y, z, t);
		local_sum += derivative(gField, coor, mu, a);
		count++;
	}}}}}}

	double global_sum;
	MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, get_comm());

	report << "derivative_sum(): mag = " << arg.mag << "\tcount = " << count << endl; 
	return global_sum;
}

inline void run_chmc(Field<Matrix> &gFieldExt, const Arg_chmc &arg, FILE *pFile){
	TIMER("algCHmcWilson::runHMC()");
	assert(pFile != NULL);
	assert(arg.num_trajectory > 20);

	RngState globalRngState("By the witness of the martyrs.");

	Geometry geoExpand1; geoExpand1.copyButExpand(gFieldExt.geo, 1);
	Geometry geoLocal; geoLocal.copyOnlyLocal(gFieldExt.geo);
	Field<Matrix> gField; gField.init(geoExpand1); gField = gFieldExt;
	Field<Matrix> mField; mField.init(geoLocal);

	Chart<Matrix> chart;
	produce_chart_envelope(chart, gFieldExt.geo, arg.gauge);

	Coordinate total_size_coarse;
	for(int i = 0; i < DIM; i++){
		total_size_coarse[i] = geoLocal.total_site(i) / arg.mag;
	}
	Geometry geo_coarse; 
	geo_coarse.init(total_size_coarse, DIM * SU3_NUM_OF_GENERATORS);
	Field<double> dField; dField.init(geo_coarse);

//	long alpha_size = product(gFieldExt.geo.node_site) * DIM * SU3_NUM_OF_GENERATORS;
//	vector<vector<double> > dev_list; dev_list.resize(alpha_size);

	double oldH, newH;
	double dieRoll;
	double deltaH, percentDeltaH;
	double acceptProbability;
	double avgPlaq;
	bool doesAccept;
	int numAccept = 0, numReject = 0;

	for(int i = 0; i < arg.num_trajectory; i++){
		init_momentum(mField);
		
		oldH = get_hamiltonian(gField, mField, arg);
		// leapFrogIntegrator(gField, mField, arg);
		force_gradient_integrator(gField, mField, arg, chart);
		
		newH = get_hamiltonian(gField, mField, arg);
	
		dieRoll = uRandGen(globalRngState);
		deltaH = newH - oldH;
		percentDeltaH = deltaH / oldH;
		acceptProbability = exp(oldH - newH);
		doesAccept = (dieRoll < acceptProbability);
		MPI_Bcast((void *)&doesAccept, 1, MPI_BYTE, 0, get_comm());
		// make sure that all the node make the same decision.
		
		if(i < arg.num_forced_accept_step){
			report << "End trajectory " << i + 1
				<< ":\tFORCE ACCEPT." << endl;
			gFieldExt = gField;
			doesAccept = true;
		}else{
			if(doesAccept){
				report << "End trajectory " << i + 1
					<< ":\tACCEPT." << endl;
				numAccept++;
				gFieldExt = gField;
			}else{
				report << "End trajectory " << i + 1
					<< ":\tREJECT." << endl;
				numReject++;
				gField = gFieldExt;	
			}	
		}
		
		report << "old Hamiltonian =\t" << oldH << endl;
		report << "new Hamiltonian =\t" << newH << endl;
		report << "exp(-Delta H) =  \t" << acceptProbability << endl;
		report << "Die Roll =       \t" << dieRoll << endl; 
		report << "Delta H =        \t" << deltaH << endl; 
		report << "Delta H Ratio =  \t" << percentDeltaH << endl; 
		
		fetch_expanded_chart(gField, chart);
		avgPlaq = avg_plaquette(gField);
		report << "avgPlaq =        \t" << avgPlaq << endl;

//		derivative_list(dev_list, gField, arg);	
		double dv_sum = derivative_sum(gField, arg);
		report << "FINE DERIVATIVE SUM =\t" << dv_sum << endl;

		if(get_id_node() == 0){
			fprintf(pFile, "%i\t%.6e\t%.6e\t%.12e\t%+.12e\t%i\n", i + 1, 
				abs(deltaH), acceptProbability, avgPlaq, dv_sum, doesAccept);
			fflush(pFile);
		}

		if((i + 1) % arg.num_step_between_output == 0 && i + 1 >= arg.num_step_before_output){
			Arg_export arg_export;
			arg_export.beta = arg.beta;
			arg_export.sequence_num = i + 1;
			arg_export.ensemble_label = "constrained_hmc";
			if(arg.export_dir_stem.size() > 0){
				string address = arg.export_dir_stem + "ckpoint." + show(i + 1);
				export_config_nersc(gFieldExt, address, arg_export, true);
			}
			if(get_id_node() == 0) printf("ACCEPT RATE = %.3f\n",
										(double)numAccept / (numAccept + numReject));		
			if(arg.summary_dir_stem.size() > 0){
				derivative_field(dField, gField, arg);
				Field<double> dField_output; dField_output.init(geo_coarse);
				sophisticated_make_to_order(dField_output, dField);
				sophisticated_serial_write(dField_output, 
									arg.summary_dir_stem + "./dev_dump." + show(i + 1));
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

	if(get_id_node() == 0){
		fprintf(pFile, "Accept Rate = %.3f\n", 
				(double)numAccept / (numAccept + numReject));
		fflush(pFile);
	}

	Timer::display();
}

