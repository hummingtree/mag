#pragma once

#include <iostream>
#include <fstream>
#include <omp.h>
#include <algorithm>
#include <array> 
#include <map>

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

inline map<int, vector<int> > init_chair_index(){
	map<int, vector<vector<int> > > ret;
	vector<int> dir(4);
	for(int mu = 0; mu < DIM; mu++){
		vector<vector<int> > assign(0);
		for(int nu = 0; nu < DIM; nu++){
			if(mu == nu) continue;
			for(int lambda = 0; lambda < DIM; lambda++){
				
				if(lambda == nu) continue;
				if(lambda == mu) continue;
				
				dir.resize(4);
				dir[0] = nu; dir[1] = lambda; dir[2] = nu + DIM; dir[3] = lambda + DIM;
				for(int i = 1; i < 4; i++){
					dir.insert(i, mu);
					assign.push_back(dir);
				}
				
				dir.resize(4);
				dir[0] = nu + DIM; dir[1] = lambda; dir[2] = nu; dir[3] = lambda + DIM;
				for(int i = 1; i < 4; i++){
					dir.insert(i, mu);
					assign.push_back(dir);
				}
				
				dir.resize(4);
				dir[0] = nu + DIM; dir[1] = lambdai + DIM; dir[2] = nu; dir[3] = lambda;
				for(int i = 1; i < 4; i++){
					dir.insert(i, mu);
					assign.push_back(dir);
				}
				
				dir.resize(4);
				dir[0] = nu; dir[1] = lambda + DIM; dir[2] = nu + DIM; dir[3] = lambda;
				for(int i = 1; i < 4; i++){
					dir.insert(i, mu);
					assign.push_back(dir);
				}
			}
		}
 
		ret[mu] = assign;
	}

	return ret;

}

const static map<int, vector<vector<int> > > chair_index = init_chair_index();

inline Matrix chair(const Field<Matrix> &f, const Coordinate &x){
	return Matrix;
} 

inline Matrix chair_staple_dagger(const Field<Matrix> &f, const Coordinate &x, int mu){
	return Matrix;
}




