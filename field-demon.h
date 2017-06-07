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
#include "field-staple.h"

using namespace cps;
using namespace qlat;
using namespace std;

inline void demon_microcanonical(const Field<cps::Matrix> &gField_ext, const Arg_chmc &arg, double E_max, const std::string& info){
	// Arxiv:hep-lat/9406019, 27 Jun 1994
	TIMER("demon_microcanonical()");

// can ONLY be run on a single node
#ifndef USE_SINGLE_NODE
	assert(false);
#else

	const int num_hits = 2;

	RngState global_rng_state("By the witness of the martyrs");

	FILE *p = NULL;
	if(arg.summary_dir_stem.size() > 0){
		p = Fopen((arg.summary_dir_stem + "/demon_equil" + info + ".dat").c_str(), "a");
	}
	
	// TODO: now only for plaquette and rectangular;
	vector<double> E(4, 0.);
	vector<double> S(4);
	vector<double> S0(4);
	
	Coordinate expansion(2, 2, 2, 2);
	Geometry geo = gField_ext.geo; geo.resize(expansion, expansion);
	Field<cps::Matrix> gField1; gField1.init(geo); gField1 = gField_ext;
	Field<cps::Matrix> gField2; gField2.init(geo); gField2 = gField_ext;

	fetch_expanded(gField1);
	S0[0] = -total_plaq(gField1);
	S0[1] = -total_rectangular(gField1);

	long accept = 0;
	long reject = 0;

	cps::Matrix m, c;
	cps::Matrix P, R, C, T;
	vector<double> delta_S(4);

	for(int i = 0; i < arg.num_trajectory; i++){
		for(long index = 0; index < geo.local_volume(); index++){
			Coordinate x = coordinate_from_index(index, geo.node_site);
			qlat::Vector<cps::Matrix> v = gField1.get_elems(x);
			for(int mu = 0; mu < geo.multiplicity; mu++){	
				get_staple_dagger(P, gField1, x, mu);	
				get_rectangular_dagger(R, gField1, x, mu);
				C = chair_staple_dagger(gField1, x, mu);
				T = twist_staple_dagger(gField1, x, mu);
				for(int n = 0; n < num_hits; n++){
					m = random_su3_from_su2(0.9, global_rng_state);
					delta_S[0] = -(m * v[mu] * P).ReTr() + (v[mu] * P).ReTr();
					delta_S[1] = -(m * v[mu] * R).ReTr() + (v[mu] * R).ReTr();
					delta_S[2] = -(m * v[mu] * C).ReTr() + (v[mu] * C).ReTr();
					delta_S[3] = -(m * v[mu] * T).ReTr() + (v[mu] * T).ReTr();
					// printf("%.8f;\t %.8f;\n", delta_S[0], delta_S[1]);
					if(E[0] - delta_S[0] > -E_max && E[0] - delta_S[0] < E_max &&
							E[1] - delta_S[1] > -E_max && E[1] - delta_S[1] < E_max &&
							E[2] - delta_S[2] > -E_max && E[2] - delta_S[2] < E_max &&
							E[3] - delta_S[3] > -E_max && E[3] - delta_S[3] < E_max){
						v[mu] = m * v[mu];
						E[0] -= delta_S[0];
						E[1] -= delta_S[1];
						E[2] -= delta_S[2];
						E[3] -= delta_S[3];
						accept++;
					}else{
						reject++;
					}
				}
				

		}}

		if(i >= arg.num_step_before_output && i % arg.num_step_between_output == 0 && !get_id_node()){
			fwrite((char*)E.data(), 1, sizeof(double) * E.size(), p);
			fflush(p);
		}

//		fetch_expanded(gField1);
//		S[0] = -total_plaq(gField1);
//		S[1] = -total_rectangular(gField1);
//		
//		qlat::Printf("S0[0] = %.8f\n", S0[0]);
//		qlat::Printf("S0[1] = %.8f\n", S0[1]);
//		
//		qlat::Printf("S[0] = %.8f\n", S[0]);
//		qlat::Printf("S[1] = %.8f\n", S[1]);
	
		qlat::Printf("E[0] = %.8f.\n", E[0]);
		qlat::Printf("E[1] = %.8f.\n", E[1]);
		qlat::Printf("E[2] = %.8f.\n", E[2]);
		qlat::Printf("E[3] = %.8f.\n", E[3]);
		qlat::Printf("accept = %d\n", accept);
		qlat::Printf("reject = %d\n", reject);
		qlat::Printf("unitarize   = %.8e\n", reunitarize(gField1));
		qlat::Printf("Accept Rate = %.3f\n", (double)accept / (accept + reject));
		qlat::Printf("---------------------END OF %3d-------------------\n", i+1);
	}
	
	if(!get_id_node()) fclose(p);

#endif

//	assert(!arg.mag);
//	
//	RngState globalRngState("By the witness of the martyrs");
//
//	FILE *p = NULL;
//	if(arg.summary_dir_stem.size() > 0){
//		p = Fopen((arg.summary_dir_stem + "/demon.dat").c_str(), "a");
//	}
//
//	Coordinate expansion(2, 2, 2, 2);
//	Geometry geo = gField_ext.geo; geo.resize(expansion, expansion);
//	Field<cps::Matrix> gField1; gField1.init(geo); gField1 = gField_ext;
//	Field<cps::Matrix> gField2; gField2.init(geo); gField2 = gField_ext;
//	Field<cps::Matrix> mField; mField.init(geo);
//
//	Chart<cps::Matrix> chart;
//	produce_chart_envelope(chart, geo, arg.gauge);
//
//	// TODO: now only for plaquette and rectangular;
//	vector<double> E(2); E[0] = 0.; E[1] = 0.;
//	vector<double> S(2);
//	vector<double> S0(2);
//	
//	fetch_expanded_chart(gField1, chart);
//	S0[0] = -total_plaq(gField1);
//	S0[1] = -total_rectangular(gField1);
//
//	int accept = 0;
//	int reject = 0;
//
//	for(int i = 0; i < arg.num_trajectory; i++){
//		init_momentum(mField);
//		
//		force_gradient_integrator(gField1, mField, arg, chart);
//		
//		fetch_expanded_chart(gField1, chart);
//		S[0] = -total_plaq(gField1);
//		S[1] = -total_rectangular(gField1);
//		
//		qlat::Printf("S0[0] = %.12f\n", S0[0]);
//		qlat::Printf("S0[1] = %.12f\n", S0[1]);
//		
//		qlat::Printf("S[0] = %.12f\n", S[0]);
//		qlat::Printf("S[1] = %.12f\n", S[1]);
//		
//		if(S0[0] - S[0] < E_max && S0[0] - S[0] > -E_max 
//			&& S0[1] - S[1] < E_max && S0[1] - S[1] > -E_max){
//			E[0] = S0[0] - S[0];
//			E[1] = S0[1] - S[1];
//			gField2 = gField1;
//			qlat::Printf("DEMON Trajectory %3d: ACCEPT.\n", i+1);
//			accept++;
//		}else{
//			gField1 = gField2;
//			qlat::Printf("DEMON Trajectory %3d: REJECT.\n", i+1);
//			reject++;
//		}
//		
//		qlat::Printf("E[0] = %.12f.\n", E[0]);
//		qlat::Printf("E[1] = %.12f.\n", E[1]);
//		qlat::Printf("Accept Rate = %.3f\n", (double)accept / (accept + reject));
//		qlat::Printf("----------------------------------------\n");
//	
//		if(i >= arg.num_step_before_output && !get_id_node()){
//			fwrite(E.data(), sizeof(double), E.size(), p);
//			fflush(p);
//		}
//	}
//	
//	if(!get_id_node()) fclose(p);

}


