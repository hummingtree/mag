#pragma once

#include <iostream>
#include <fstream>
#include <omp.h>
#include <algorithm>
#include <array> 

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

inline double wilson_loop(const Field<Matrix> &f, const Coordinate &x, 
									const array<int, DIM - 1> &r, int T){
	
	assert(T > 0);
	for(int i = 0; i < DIM - 1; i++) assert(r[i] >= 0);

	vector<int> dir; dir.clear();

	for(int i = 0; i < DIM - 1; i++){
	for(int j = 0; j < r[i]; j++){
		dir.push_back(i);
	}}
	
	for(int i = 0; i < T; i++){
		dir.push_back(DIM - 1);
	}
	
	for(int i = DIM - 2; i >= 0; i--){
	for(int j = 0; j < r[i]; j++){
		dir.push_back(i + DIM);
	}}
	
	for(int i = 0; i < T; i++){
		dir.push_back(DIM - 1 + DIM);
	}

	Matrix m;
	get_path_ordered_product(m, f, x, dir);

	return m.ReTr();
}

inline double avg_wilson_loop(const Field<Matrix> &f, 
										const array<int, DIM - 1> &r, int T){
	TIMER("avg_wilson_loop()");
	double local_sum = 0.;
	int num_threads;
	vector<double> pp_local_sum; // container to store sum from different threads

	// produce permutations of the spatial vector r
	array<int, DIM - 1> r_local = r; sort(r_local.begin(), r_local.end());
	vector<array<int, DIM - 1> > perm; perm.clear();

	do{
		perm.push_back(r_local);
	}while(next_permutation(r_local.begin(), r_local.end()));

#pragma omp parallel
	{
		if(omp_get_thread_num() == 0){
			num_threads = omp_get_num_threads();
			pp_local_sum.resize(num_threads);
		}
		double p_local_sum = 0.;
#pragma omp barrier
#pragma omp for
		for(long index = 0; index < f.geo.local_volume(); index++){
			Coordinate x = f.geo.coordinate_from_index(index);
			vector<array<int, DIM - 1> >::iterator it;
			for(it = perm.begin(); it != perm.end(); it++){
				p_local_sum += wilson_loop(f, x, *it, T);
			}
		}
		pp_local_sum[omp_get_thread_num()] = p_local_sum;
	}
	
	for(int i = 0; i < num_threads; i++){
		local_sum += pp_local_sum[i];
	}

	double global_sum;
	MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, get_comm());
	return global_sum / (perm.size() * get_num_node() * f.geo.local_volume());	
}

inline void get_staple_spatial(Matrix &staple, const Field<Matrix> &field, 
					const Coordinate &x, const int i){
	vector<int> dir;
	Matrix staple_; staple_.ZeroMatrix();
	Matrix m;
	for(int j = 0; j < DIM - 1; j++){
		if(i == j) continue;
		dir.clear();
		dir.push_back(j); dir.push_back(i); dir.push_back(j + DIM);
		get_path_ordered_product(m, field, x, dir);
		staple_ += m;
		dir.clear();
		dir.push_back(j + DIM); dir.push_back(i); dir.push_back(j);
		get_path_ordered_product(m, field, x, dir);
		staple_ += m;
	}
	staple = staple_;
}

inline void ape_smear(Field<Matrix> &f, double coeff, int num){
	// U_i -> (1 - coeff) * U_i + (coeff / 4) * \SUM(U_i_staples)
	
	Geometry geo1 = f.geo;
	Coordinate expansion(1, 1, 1, 1);
	geo1.resize(expansion, expansion);

	Field<Matrix> copy; copy.init(geo1, DIM);
	Field<Matrix> incr; incr.init(geo1, DIM);
	copy = f;
	
	Chart<Matrix> chart;
	Gauge gauge;
	produce_chart_envelope(chart, geo1, gauge);

	for(int count = 0; count < num; count++){
		fetch_expanded_chart(copy, chart);
#pragma omp parallel for
		for(long index = 0; index < copy.geo.local_volume(); index++){
			Coordinate x = copy.geo.coordinate_from_index(index);
			qlat::Vector<Matrix> ix = incr.get_elems(x);
			Matrix m;
			for(int i = 0; i < DIM - 1; i++){
				get_staple_spatial(m, copy, x, i);
				ix[i] = m;
		}}
	
#pragma omp parallel for
		for(long index = 0; index < copy.geo.local_volume(); index++){
			Coordinate x = copy.geo.coordinate_from_index(index);
			qlat::Vector<Matrix> cx = copy.get_elems(x);
			qlat::Vector<Matrix> ix = incr.get_elems(x);
			for(int i = 0; i < DIM - 1; i++){
				cx[i] = ix[i] * (coeff / 6.) + cx[i] * (1. - coeff);
// for Sommer scale purposes we don't need to unitarize the matrix.
				su3_proj(cx[i], 10e-8);
//				cx[i].Unitarize();
		}}
	}
	
	f = copy;	
}
