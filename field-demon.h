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

using namespace cps;
using namespace qlat;
using namespace std;

#define E_MAX 600.

inline void demon_microcanonical(const Field<Matrix> &gField_ext, const Arg_chmc &arg){
	// Arxiv:hep-lat/9406019, 27 Jun 1994
	TIMER("demon_microcanonical()");

	assert(!arg.mag);
	
	RngState globalRngState("By the witness of the martyrs");

	FILE *p = NULL;
	if(arg.summary_dir_stem.size() > 0){
		p = Fopen((arg.summary_dir_stem + "/demon.dat").c_str(), "a");
	}

	Coordinate expansion(2, 2, 2, 2);
	Geometry geo = gField_ext.geo; geo.resize(expansion, expansion);
	Field<Matrix> gField1; gField1.init(geo); gField1 = gField_ext;
	Field<Matrix> gField2; gField2.init(geo); gField2 = gField_ext;
	Field<Matrix> mField; mField.init(geo);

	Chart<Matrix> chart;
	produce_chart_envelope(chart, geo, arg.gauge);

	// TODO: now only for plaquette and rectangular;
	vector<double> E(2); E[0] = 0.; E[1] = 1.;
	vector<double> S(2);
	vector<double> S0(2);
	
	fetch_expanded_chart(gField1, chart);
	S0[0] = -total_plaq(gField1);
	S0[1] = -total_rectangular(gField1);
	
	for(int i = 0; i < arg.num_trajectory; i++){
		init_momentum(mField);
		force_gradient_integrator(gField1, mField, arg, chart);
		
		fetch_expanded_chart(gField1, chart);
		S[0] = -total_plaq(gField1);
		S[1] = -total_rectangular(gField1);
		
		if(S0[0] - S[0] < E_MAX && S0[0] - S[0] > -E_MAX 
			&& S0[1] - S[1] < E_MAX && S0[1] - S[1] > -E_MAX){
			E[0] = S0[0] - S[0];
			E[1] = S0[1] - S[1];
			gField2 = gField1;
			qlat::Printf("DEMON ACCEPT.\n");
		}else{
			gField1 = gField2;
			qlat::Printf("DEMON REJECT.\n");
		}
		
		qlat::Printf("E[0] = %.12f.\n", E[0]);
		qlat::Printf("E[1] = %.12f.\n", E[1]);

		Fprintf(p, "%.12f\t%.12f\n", E[0], E[1]);
	}
	
	if(!get_id_node()) fclose(p);

}


