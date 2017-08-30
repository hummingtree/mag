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

#include "field-mass.h"

using namespace cps;
using namespace qlat;
using namespace std;

namespace mass {

inline void get_force(Field<cps::Matrix> &fField, const Field<cps::Matrix> &gField,
			const Arg_chmc &arg){
	TIMER("get_force()");
	assert(is_matching_geo(fField.geo, gField.geo));

	if(arg.gauge.type == qlat::WILSON){
#pragma omp parallel for
		for(long index = 0; index < fField.geo.local_volume(); index++){
			qlat::Coordinate x; 
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
					qlat::Coordinate y(x); y[mu]++;
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
			qlat::Coordinate x; 
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
					qlat::Coordinate y(x); y[mu]++;
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

inline void evolve_gauge_field(Field<cps::Matrix> &gField, 
				const Field<cps::Matrix> &mField, double dt, 
				const Arg_chmc &arg){
	TIMER("evolve_gauge_field()");
	assert(is_matching_geo(mField.geo, gField.geo));
#pragma omp parallel for
	for(long index = 0; index < gField.geo.local_volume(); index++){
		qlat::Coordinate x = gField.geo.coordinate_from_index(index);
		cps::Matrix mL, mR;
		const qlat::Vector<cps::Matrix> mx = mField.get_elems_const(x);
			  qlat::Vector<cps::Matrix> gx = gField.get_elems(x);
		for(int mu = 0; mu < gField.geo.multiplicity; mu++){
		// only works for cps::Matrix
			qlat::Coordinate y(x); y[mu]--;
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
		qlat::Coordinate x = mField.geo.coordinate_from_index(index);
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

inline void init_momentum(Field<cps::Matrix> &mField, double m = 1.){
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

	double sig = std::sqrt(m);

#pragma omp parallel for
	for(long index = 0; index < mField.geo.local_volume(); index++){
		qlat::Coordinate x = mField.geo.coordinate_from_index(index);
		qlat::Vector<cps::Matrix> mx = mField.get_elems(x);
		cps::Matrix mTemp;
		for(int mu = 0; mu < mField.geo.multiplicity; mu++){
			mTemp.ZeroMatrix();
			for(int a = 0; a < SU3_NUM_OF_GENERATORS; a++){
				mTemp += su3_generators[a] * g_rand_gen(rng_field.get_elem(x), 0., sig);
			}
			mx[mu] = mTemp;
	}}
}

inline void run_chmc(Field<cps::Matrix> &gFieldExt, const Arg_chmc &arg, FILE *pFile){
	TIMER("run_chmc()");
	if(!get_id_node()) assert(pFile != NULL);
	assert(arg.num_trajectory > 20);

	RngState globalRngState("By the witness of the martyrs.");

	qlat::Coordinate expansion(2, 2, 2, 2);
	Geometry geoExpand1 = gFieldExt.geo; geoExpand1.resize(expansion, expansion);
	Geometry geoLocal = gFieldExt.geo;
	Field<cps::Matrix> gField; gField.init(geoExpand1); gField = gFieldExt;
	Field<cps::Matrix> mField; mField.init(geoLocal);

	Chart<cps::Matrix> chart;
	produce_chart_envelope(chart, gFieldExt.geo, arg.gauge);

	qlat::Coordinate total_size_coarse;
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

}
