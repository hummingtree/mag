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
#include "field-wloop.h"

using namespace cps;
using namespace qlat;
using namespace std;

inline double wilson_loop(Field<Matrix> &f, Coordinate &x, 
									array<int, DIM - 1> &R, int T){
	
	assert(T > 0);
	for(int i = 0; i < DIM - 1; i++) assert(R[i] >= 0);

	vector<int> dir; dir.clear();

	for(int i = 0; i < DIM - 1; i++){
	for(int j = 0; j < R[i]; j++){
		dir.push_back(i);
	}}
	
	for(int i = 0; i < T; i++){
		dir.push_back(DIM - 1);
	}
	
	for(int i = DIM - 2; i >= 0; i--){
	for(int j = 0; j < R[i]; j++){
		dir.push_back(i + DIM);
	}}
	
	for(int i = 0; i < T; i++){
		dir.push_back(DIM - 1 + DIM);
	}

	Matrix m;
	get_path_ordered_product(m, f, x, dir);

	return m.ReTr();
}

inline double avg_wilson_loop(Field<Matrix> &f, array<int, DIM - 1> &R, int T){
	TIMER("avg_wilson_loop()");
	double local_sum = 0.;
	int num_threads;
	vector<double> pp_local_sum; // container to store sum from different threads

#pragma omp parallel
	{
		if(omp_get_thread_num() == 0){
			num_threads = omp_get_num_threads();
			pp_local_sum.resize(numThreads);
		}
		double p_local_sum = 0.;
#pragma omp barrier
#pragma omp for
		for(long index = 0; index < mField.geo.local_volume(); index++){
			Coordinate x; f.geo.coordinate_from_index(x, index);
			p_local_sum += wilson_loop(f, x, R, T);
		}
		pp_local_sum[omp_get_thread_num()] = p_local_sum;
	}
	
	for(int i = 0; i < numThreads; i++){
		local_sum += pp_local_sum[i];
	}

	double global_sum;
	MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, get_comm());
	return global_sum / (get_num_node() * f.geo.local_volume());	
}


