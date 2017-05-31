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

inline map<int, vector<vector<int> > > init_chair_index(){
	map<int, vector<vector<int> > > ret;
	vector<int> dir(4), dir_copy;
	for(int mu = 0; mu < DIMN; mu++){
		vector<vector<int> > assign(0);
		for(int nu = 0; nu < DIMN; nu++){
			if(mu == nu) continue;
			for(int lambda = 0; lambda < DIMN; lambda++){
				
				if(lambda == nu) continue;
				if(lambda == mu) continue;

				dir[0] = nu; dir[1] = lambda; dir[2] = lambda + DIMN; dir[3] = nu + DIMN;
				for(int i = 1; i < 4; i++){
					dir_copy = dir;
					vector<int>::iterator it = dir_copy.begin(); 
					dir_copy.insert(it + i, mu);
					assign.push_back(dir_copy);
				}
				
				dir[0] = nu; dir[1] = lambda + DIMN; dir[2] = lambda + DIMN; dir[3] = nu;
				for(int i = 1; i < 4; i++){
					dir_copy = dir;
					vector<int>::iterator it = dir_copy.begin(); 
					dir_copy.insert(it + i, mu);
					assign.push_back(dir_copy);
				}
				
				dir[0] = nu + DIMN; dir[1] = lambda; dir[2] = lambda; dir[3] = nu + DIMN;
				for(int i = 1; i < 4; i++){
					dir_copy = dir;
					vector<int>::iterator it = dir_copy.begin(); 
					dir_copy.insert(it + i, mu);
					assign.push_back(dir_copy);
				}
				
				dir[0] = nu + DIMN; dir[1] = lambda + DIMN; dir[2] = lambda; dir[3] = nu;
				for(int i = 1; i < 4; i++){
					dir_copy = dir;
					vector<int>::iterator it = dir_copy.begin(); 
					dir_copy.insert(it + i, mu);
					assign.push_back(dir_copy);
				}
			}
		}
 
		ret[mu] = assign;
	}

	return ret;

}

const static map<int, vector<vector<int> > > chair_index = init_chair_index();

inline map<int, vector<vector<int> > > init_twist_index(){
	map<int, vector<vector<int> > > ret;
	vector<int> dir(4), dir_copy;
	for(int mu = 0; mu < DIMN; mu++){
		vector<vector<int> > assign(0);
		for(int nu = 0; nu < DIMN; nu++){
			if(mu == nu) continue;
			for(int lambda = 0; lambda < DIMN; lambda++){
				
				if(lambda == nu) continue;
				if(lambda == mu) continue;

				dir[0] = nu; dir[1] = lambda; dir[2] = nu + DIMN; dir[3] = lambda + DIMN;
				for(int i = 1; i < 4; i++){
					dir_copy = dir;
					vector<int>::iterator it = dir_copy.begin(); 
					dir_copy.insert(it + i, mu);
					assign.push_back(dir_copy);
				}
				
				dir[0] = nu; dir[1] = lambda + DIMN; dir[2] = nu + DIMN; dir[3] = lambda;
				for(int i = 1; i < 4; i++){
					dir_copy = dir;
					vector<int>::iterator it = dir_copy.begin(); 
					dir_copy.insert(it + i, mu);
					assign.push_back(dir_copy);
				}
				
				dir[0] = nu + DIMN; dir[1] = lambda; dir[2] = nu; dir[3] = lambda + DIMN;
				for(int i = 1; i < 4; i++){
					dir_copy = dir;
					vector<int>::iterator it = dir_copy.begin(); 
					dir_copy.insert(it + i, mu);
					assign.push_back(dir_copy);
				}
				
				dir[0] = nu + DIMN; dir[1] = lambda + DIMN; dir[2] = nu; dir[3] = lambda;
				for(int i = 1; i < 4; i++){
					dir_copy = dir;
					vector<int>::iterator it = dir_copy.begin(); 
					dir_copy.insert(it + i, mu);
					assign.push_back(dir_copy);
				}
			}
		}
 
		ret[mu] = assign;
	}

	return ret;

}

const static map<int, vector<vector<int> > > twist_index = init_twist_index();

inline Matrix chair(const Field<Matrix> &f, const Coordinate &x){
	return Matrix();
} 

inline Matrix chair_staple_dagger(const Field<Matrix> &f, const Coordinate &x, int mu){
	vector<vector<int> >::const_iterator it = chair_index.at(mu).cbegin();
	Matrix acc; acc.ZeroMatrix();
	Matrix temp;
	for(; it != chair_index.at(mu).cend(); it++){
		get_path_ordered_product(temp, f, x, *it);
		acc = acc + temp;
	}
	return acc;
}

inline Matrix twist(const Field<Matrix> &f, const Coordinate &x){
	return Matrix();
} 

inline Matrix twist_staple_dagger(const Field<Matrix> &f, const Coordinate &x, int mu){
	vector<vector<int> >::const_iterator it = twist_index.at(mu).cbegin();
	Matrix acc; acc.ZeroMatrix();
	Matrix temp;
	for(; it != twist_index.at(mu).cend(); it++){
		get_path_ordered_product(temp, f, x, *it);
		acc = acc + temp;
	}
	return acc;
}

