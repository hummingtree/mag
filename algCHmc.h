#pragma once

#include <iostream>

#include <qmp.h>
#include <config.h>
#include <util/lattice.h>
#include <util/gjp.h>
#include <util/verbose.h>
#include <util/error.h>
#include <util/random.h>
#include <alg/alg_pbp.h>
#include <alg/do_arg.h>
#include <alg/common_arg.h>
#include <alg/pbp_arg.h>

#include <alg/alg_meas.h>
#include <util/ReadLatticePar.h>
#include <util/qioarg.h>

#include <qlat/config.h>
#include <qlat/utils.h>
#include <qlat/mpi.h>
#include <qlat/field.h>
#include <qlat/field-io.h>
#include <qlat/field-comm.h>

using namespace cps;
using namespace qlat;
using namespace std;

double avg_plaquette(qlat::Field<cps::Matrix> &gauge_field_qlat){
	std::vector<Coordinate> dir_vec(4);
	dir_vec[0] = Coordinate(1, 0, 0, 0);
	dir_vec[1] = Coordinate(0, 1, 0, 0);
	dir_vec[2] = Coordinate(0, 0, 1, 0);
	dir_vec[3] = Coordinate(0, 0, 0, 1);

	qlat::Geometry geo_ = gauge_field_qlat.geo;

	double node_sum = 0.;
	
	for(long index = 0; index < geo_.localVolume(); index++){
		 Coordinate x_qlat; geo_.coordinateFromIndex(x_qlat, index);
		 for(int mu = 0; mu < DIM; mu++){
		 for(int nu = 0; nu < mu; nu++){	
		 	cps::Matrix mul; mul.UnitMatrix();
			mul = mul * gauge_field_qlat.getElems(x_qlat)[mu];
			x_qlat = x_qlat + dir_vec[mu];
			mul = mul * gauge_field_qlat.getElems(x_qlat)[nu];
			x_qlat = x_qlat + dir_vec[nu] - dir_vec[mu];
			cps::Matrix dag1; 
			dag1.Dagger(gauge_field_qlat.getElems(x_qlat)[mu]);
			mul = mul * dag1;
			x_qlat = x_qlat - dir_vec[nu];
			cps::Matrix dag2;
			dag2.Dagger(gauge_field_qlat.getElems(x_qlat)[nu]);
			mul = mul * dag2;

			node_sum += mul.ReTr();
		 }}
	}
	double global_sum;
	MPI_Allreduce(&node_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, getComm());

	return global_sum / (18. * getNumNode() * geo_.localVolume());
}

double avg_real_trace(qlat::Field<cps::Matrix> &gauge_field_qlat){
	qlat::Geometry geo_ = gauge_field_qlat.geo;
	double tr_node_sum = 0.;
	for(long index = 0; index < geo_.localVolume(); index++){
		 Coordinate x_qlat; geo_.coordinateFromIndex(x_qlat, index);
		 for(int mu = 0; mu < DIM; mu++){
		 	tr_node_sum += (gauge_field_qlat.getElems(x_qlat)[mu]).ReTr();
		 }
	}
	double tr_global_sum = 0.;
	MPI_Allreduce(&tr_node_sum, &tr_global_sum, 1, MPI_DOUBLE, MPI_SUM, getComm());

	return tr_global_sum / (12. * getNumNode() * geo_.localVolume());
}

double check_constrained_plaquette(qlat::Field<cps::Matrix> &gauge_field_qlat,
					int mag){
	std::vector<Coordinate> dir_vec(4);
	dir_vec[0] = Coordinate(1, 0, 0, 0);
	dir_vec[1] = Coordinate(0, 1, 0, 0);
	dir_vec[2] = Coordinate(0, 0, 1, 0);
	dir_vec[3] = Coordinate(0, 0, 0, 1);

	qlat::Geometry geo_ = gauge_field_qlat.geo;
	
	long count = 0;
	double node_sum = 0.;
	for(int x0 = 0; x0 < geo_.nodeSite[0]; x0 += mag){
	for(int x1 = 0; x1 < geo_.nodeSite[1]; x1 += mag){
	for(int x2 = 0; x2 < geo_.nodeSite[2]; x2 += mag){
	for(int x3 = 0; x3 < geo_.nodeSite[3]; x3 += mag){
		Coordinate x(x0, x1, x2, x3);
		for(int mu = 0; mu < DIM; mu++){
		for(int nu = 0; nu < mu; nu++){
			cps::Matrix mul; mul.UnitMatrix();
			for(int i = 0; i < mag; i++){
				mul = mul * gauge_field_qlat.getElems(x)[mu];
				x = x + dir_vec[mu];
			}
			for(int i = 0; i < mag; i++){
				mul = mul * gauge_field_qlat.getElems(x)[nu];
				x = x + dir_vec[nu];
			}
			cps::Matrix dag;
			for(int i = 0; i < mag; i++){
				x = x - dir_vec[mu];
				dag.Dagger(gauge_field_qlat.getElems(x)[mu]);
				mul = mul * dag;
			}
			for(int i = 0; i < mag; i++){
				x = x - dir_vec[nu];
				dag.Dagger(gauge_field_qlat.getElems(x)[nu]);
				mul = mul * dag;
			}
			count++;
			node_sum += mul.ReTr();
		}}
	}}}}

	double global_sum;
	MPI_Allreduce(&node_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, getComm());

	return global_sum / (3. * count * getNumNode());
}

inline void getPathOrderedProd(Matrix &prod, Field<Matrix> &field, 
				const Coordinate &x, vector<int> &dir){
	Matrix mul; mul.UnitMatrix();
	Matrix dag;
	Coordinate y(x);
	for(vector::iterator it = dir.begin(); it != dir.end(); it++){
		assert(*it < DIM * 2 && *it > -1);
		if(*it < DIM && *it > -1){
			mul = mul * field.getElemsConst(y)[*it];
			y[*it]++;
		}else{
			y[*it - DIM]--;
			dag.Dagger(field.getElemsConst(y)[*it - DIM]);
			mul = mul * dag;
		}
	}
	prod = mul;
}

inline void getStaple(Matrix &staple, Field<Matrix> &field, 
				const Coordinate &x, const int mu){
	vector<int> dir; dir.clear();
	Matrix staple_; staple.ZeroMatrix();
	for(int nu = 0; nu < DIM; nu++){
		if(mu == nu) continue;
		dir.push_back(nu); dir.push_back(mu); dir.push_back(nu + DIM);
		staple_ = staple_ + getPathOrderedProd(field, x, dir);
		dir.clear();
		dir.push_back(nu + DIM); dir.push_back(mu); dir.push_back(nu);
		staple_ = staple_ + getPathOrderedProd(filed, x, dir);
	}
	staple = staple_;
}

class argCHmcWilson{
public:
	int mag;
	double beta;
	Field<Matrix> *gField;
};

class algCHmcWilson{
private:
	
	argCHmcWilson *arg;
	Field<Matrix> fField;

	inline bool isConstrained(const Coordinate &x, int mu, int mag)
	{
		bool isConstrained_ = true;
		for(int i = 0; i < 4; i++){
			if(i == mu) continue;
			isConstrained_ = isConstrained_ && (x[i] % mag == 0);
		}
		return isConstrained_;
	}
	
	inline void getForce(Matrix &force, const Coordinate &x, const int mu){
		Matrix mStaple1, mStaple2, mTemp;
		Matrix dagger1, dagger2;
		Matrix mTemp1, mTemp2;
		if(isConstrained(x, mu, arg->mag)){
			if(x[mu] % arg->mag == arg->mag - 1){}
			else{
				Coordinate y(x); y[mu]++;
				getStaple(mStaple1, *(arg->gField), x, mu);
				getStaple(mStaple2, *(arg->gField), y, mu);
				dagger1.Dagger(mStaple1); 
				dagger2.Dagger(mStaple2);
				mTemp1 = arg->gField.getElemsConst(x)[mu];
				mTemp2 = arg->gField.getElemsConst(y)[mu];
				mTemp = dagger1 * mTemp1 - mTemp2 * dagger2;
			}
		}else{
			getStaple(mStaple1, *(arg->gField), x, mu);
			dagger1.Dagger(mStaple1); 
			mTemp1 = arg->gField.getElemsConst(x)[mu];
			mTemp = -mTemp1 * dagger1;
		}
		mTemp *= arg->beta / 3.; 
		force.TrLessAntiHermMatrix(mTemp);
	}

public:


};







