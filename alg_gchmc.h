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

#include "cps_util.h"

using namespace cps;
using namespace qlat;
using namespace std;

#define SU3_NUM_OF_GENERATORS 8


inline void getPathOrderedProd(Matrix &prod, const Field<Matrix> &field, 
					const Coordinate &x, const vector<int> &dir);
		// forward declearation

static const double invSqrt2 = 1. / sqrt(2.);
static const double xi = 0.1931833;

inline vector<Matrix> initGenerator(){
	Matrix T1, T2, T3, T4, T5, T6, T7, T8;
	// the eight Hermitian generators of SU3	
	T1.ZeroMatrix();
	T1(0, 1) = qlat::Complex(1., 0.);  T1(1, 0) = qlat::Complex(1., 0.);
	T1 *= invSqrt2;

	T2.ZeroMatrix();
	T2(0, 1) = qlat::Complex(0., -1.); T2(1, 0) = qlat::Complex(0., 1.);
	T2 *= invSqrt2;

	T3.ZeroMatrix();
	T3(0, 0) = qlat::Complex(1., 0.);  T3(1, 1) = qlat::Complex(-1., 0.);
	T3 *= invSqrt2;

	T4.ZeroMatrix();
	T4(0, 2) = qlat::Complex(1., 0.);  T4(2, 0) = qlat::Complex(1., 0.);
	T4 *= invSqrt2;

	T5.ZeroMatrix();
	T5(0, 2) = qlat::Complex(0., -1.); T5(2, 0) = qlat::Complex(0., 1.);
	T5 *= invSqrt2;

	T6.ZeroMatrix();
	T6(1, 2) = qlat::Complex(1., 0.);  T6(2, 1) = qlat::Complex(1., 0.);
	T6 *= invSqrt2;

	T7.ZeroMatrix();
	T7(1, 2) = qlat::Complex(0., -1.); T7(2, 1) = qlat::Complex(0., 1.);
	T7 *= invSqrt2;

	T8.ZeroMatrix();
	T8(0 ,0) = qlat::Complex(1., 0.);  T8(1, 1) = qlat::Complex(1., 0.); 
	T8(2, 2) = qlat::Complex(-2., 0.);
	T8 *= 1. / sqrt(6.);
	
	vector<Matrix> ret {T1, T2, T3, T4, T5, T6, T7, T8};
	
	return ret;
}

static const vector<Matrix> su3Generator = initGenerator();

inline double avg_plaquette(const qlat::Field<cps::Matrix> &gauge_field_qlat){
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
			mul = mul * gauge_field_qlat.getElemsConst(x_qlat)[mu];
			x_qlat = x_qlat + dir_vec[mu];
			mul = mul * gauge_field_qlat.getElemsConst(x_qlat)[nu];
			x_qlat = x_qlat + dir_vec[nu] - dir_vec[mu];
			cps::Matrix dag1; 
			dag1.Dagger(gauge_field_qlat.getElemsConst(x_qlat)[mu]);
			mul = mul * dag1;
			x_qlat = x_qlat - dir_vec[nu];
			cps::Matrix dag2;
			dag2.Dagger(gauge_field_qlat.getElemsConst(x_qlat)[nu]);
			mul = mul * dag2;

			node_sum += mul.ReTr();
		 }}
	}
	double global_sum;
	MPI_Allreduce(&node_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, getComm());

	return global_sum / (18. * getNumNode() * geo_.localVolume());
}

inline double totalPlaq(const qlat::Field<cps::Matrix> &gauge_field_qlat){
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
			mul = mul * gauge_field_qlat.getElemsConst(x_qlat)[mu];
			x_qlat = x_qlat + dir_vec[mu];
			mul = mul * gauge_field_qlat.getElemsConst(x_qlat)[nu];
			x_qlat = x_qlat + dir_vec[nu] - dir_vec[mu];
			cps::Matrix dag1; 
			dag1.Dagger(gauge_field_qlat.getElemsConst(x_qlat)[mu]);
			mul = mul * dag1;
			x_qlat = x_qlat - dir_vec[nu];
			cps::Matrix dag2;
			dag2.Dagger(gauge_field_qlat.getElemsConst(x_qlat)[nu]);
			mul = mul * dag2;

			node_sum += mul.ReTr();
		 }}
	}
	double global_sum;
	MPI_Allreduce(&node_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, getComm());

	return global_sum;
}

inline double avg_real_trace(const qlat::Field<cps::Matrix> &gauge_field_qlat){
	qlat::Geometry geo_ = gauge_field_qlat.geo;
	double tr_node_sum = 0.;
	for(long index = 0; index < geo_.localVolume(); index++){
		 Coordinate x_qlat; geo_.coordinateFromIndex(x_qlat, index);
		 for(int mu = 0; mu < DIM; mu++){
		 	tr_node_sum += \
				(gauge_field_qlat.getElemsConst(x_qlat)[mu]).ReTr();
		 }
	}
	double tr_global_sum = 0.;
	MPI_Allreduce(&tr_node_sum, &tr_global_sum, 1, MPI_DOUBLE, MPI_SUM, getComm());

	return tr_global_sum / (12. * getNumNode() * geo_.localVolume());
}

inline double check_constrained_plaquette(
				const qlat::Field<cps::Matrix> &gauge_field_qlat,
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
			Matrix m;
			vector<int> dir; dir.clear();
			for(int i = 0; i < mag; i++) dir.push_back(mu);
			for(int i = 0; i < mag; i++) dir.push_back(nu);
			for(int i = 0; i < mag; i++) dir.push_back(mu + DIM);
			for(int i = 0; i < mag; i++) dir.push_back(nu + DIM);
			
			getPathOrderedProd(m, gauge_field_qlat, x, dir);
			
			count++;
			node_sum += m.ReTr();
		}}
	}}}}

	double global_sum;
	MPI_Allreduce(&node_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, getComm());

	return global_sum / (3. * count * getNumNode());
}

inline void exp(Matrix &expM, const Matrix &M){
        Matrix mTemp2 = M, mTemp3;
	for(int i = 9; i > 1; i--){
		mTemp3.OneMinusfTimesM(-1. / i, mTemp2);
		mTemp2.DotMEqual(M, mTemp3);
	}
	expM.OneMinusfTimesM(-1., mTemp2);
}

inline void LieA2LieG(Matrix &expiM, const Matrix &M){
	// expiM = exp(i * M)
	Matrix mTemp = M; mTemp *= qlat::Complex(0., 1.);
	exp(expiM, mTemp);
}

inline void getPathOrderedProd(Matrix &prod, const Field<Matrix> &field, 
					const Coordinate &x, const vector<int> &dir){
	Matrix mul; mul.UnitMatrix();
	Matrix dag;
	Coordinate y(x);
	int direction;
	for(unsigned int i = 0; i < dir.size(); i++){
		direction = dir[i];
		assert(direction < DIM * 2 && direction > -1);
		if(direction < DIM){
			mul = mul * field.getElemsConst(y)[direction];
			y[direction]++;
		}else{
			y[direction - DIM]--;
			dag.Dagger(field.getElemsConst(y)[direction - DIM]);
			mul = mul * dag;
		}
	}
	prod = mul;
}

inline void getStapleDagger(Matrix &staple, const Field<Matrix> &field, 
					const Coordinate &x, const int mu){
	vector<int> dir;
	Matrix staple_; staple_.ZeroMatrix();
	Matrix m;
	for(int nu = 0; nu < DIM; nu++){
		if(mu == nu) continue;
		dir.clear();
		dir.push_back(nu); dir.push_back(mu); dir.push_back(nu + DIM);
		getPathOrderedProd(m, field, x, dir);
		staple_ += m;
		dir.clear();
		dir.push_back(nu + DIM); dir.push_back(mu); dir.push_back(nu);
		getPathOrderedProd(m, field, x, dir);
		staple_ += m;
	}
	staple.Dagger(staple_);
}

inline void rnFillingSHA256Gaussian(std::vector<double> &xs)
{
	using namespace qlat;
	static bool initialized = false;
	static Geometry geo;
	static qlat::RngField rf;
	if (false == initialized){
		geo.init(getSizeNode(), 1);
		rf.init(geo, RngState("Ich liebe dich."));
		initialized = true;
	}
	assert(xs.size() % geo.localVolume()== 0);
	const int chunk = xs.size() / geo.localVolume();
#pragma omp parallel for
	for (long index = 0; index < geo.localVolume(); ++index){
		Coordinate xl; geo.coordinateFromIndex(xl, index);
		RngState& rs = rf.getElem(xl);
		for (int i = chunk * index; i < chunk * (index + 1); ++i){
			xs[i] = gRandGen(rs);
		}
	}
}

class argCHmcWilson{
public:
	int mag;
	int length;
	double beta;
	double dt;
	Field<Matrix> *gFieldExt;

	const argCHmcWilson& operator=(const argCHmcWilson &RHS){
		this->mag = RHS.mag;
		this->length = RHS.length;
		this->beta = RHS.beta;
		this->dt = RHS.dt;
		this->gFieldExt = RHS.gFieldExt;
		return *this;
	}
};

class algCHmcWilson{
private:
	
	RngState globalRngState;

	argCHmcWilson arg;
	Field<Matrix> gField;
	Field<Matrix> mField;

	inline int isConstrained(const Coordinate &x, int mu, int mag)
	{
		// return 0: not constrained;
		// return 1: constrained but neither the first nor the last one 
		// on the segment
		// return 10: constrained and the first one on the segment
		// return 100: constrained and the last one on the segment
		
		// debug start
		
		// return 0;

		// debug end

		bool isConstrained_ = true;
		for(int i = 0; i < 4; i++){
			if(i == mu) continue;
			isConstrained_ = isConstrained_ && (x[i] % mag == 0);
		}
		if(isConstrained_){
			if(x[mu] % mag == mag - 1) return 100;
			if(x[mu] % mag == 0) return 10;
			return 1;
		}else{
			return 0;
		}
	}
	
	inline void getForce(Matrix &force, const Coordinate &x, const int mu){
		Matrix mStaple1, mStaple2, mTemp;
		switch(isConstrained(x, mu, arg.mag)){
			case 0: {
				getStapleDagger(mStaple1, gField, x, mu);
				mTemp = gField.getElemsConst(x)[mu] * mStaple1;
				break;
			}
			case 1:
			case 10: {
				Coordinate y(x); y[mu]++;
				getStapleDagger(mStaple1, gField, x, mu);
				getStapleDagger(mStaple2, gField, y, mu);
				mTemp = gField.getElemsConst(y)[mu] * mStaple2 \
					- mStaple1 * gField.getElemsConst(x)[mu];
				break;
			}
			// case 100: force.ZeroMatrix(); break;
			
			// test case start
			case 100: {
				getStapleDagger(mStaple1, gField, x, mu);
				mTemp = mStaple1 * gField.getElemsConst(x)[mu] * -1.;
				break;
			} 
			// test case end
		
			default: assert(false);
		}
		
		mTemp.TrLessAntiHermMatrix(); 
		force = mTemp * qlat::Complex(0., arg.beta / 3.);
	}

	inline void evolveMomentum(double dt_){
		TIMER("algCHmcWilson::evolveMomentum()");
#pragma omp parallel for
		for(long index = 0; index < mField.geo.localVolume(); index++){
			Coordinate x; 
			mField.geo.coordinateFromIndex(x, index);
			Matrix mForce;
			for(int mu = 0; mu < gField.geo.multiplicity; mu++){
			// only works for cps::Matrix
				getForce(mForce, x, mu);
		// 		if(getIdNode() == 0){
		// 			cout << "Before: " << endl;
		// 			cout << mField.getElems(x)[mu] << endl;
		// 		}
				mField.getElems(x)[mu] += mForce * dt_;
		// 		if(getIdNode() == 0){
		// 			cout << "After: " << endl;
		// 			cout << mField.getElems(x)[mu] << endl;
		// 		}

		}}
	}

	inline void evolveGaugeField(double dt_){
		TIMER("algCHmcWilson::evolveGaugeField()");
#pragma omp parallel for
		for(long index = 0; index < gField.geo.localVolume(); index++){
			Coordinate x; 
			gField.geo.coordinateFromIndex(x, index);
			Matrix mTemp;
			Matrix mLeft, mRight;
			for(int mu = 0; mu < gField.geo.multiplicity; mu++){
			// only works for cps::Matrix
				Matrix &U = gField.getElems(x)[mu];
				Coordinate y(x); y[mu]--;
				switch(isConstrained(x, mu, arg.mag)){
				case 0: {
					LieA2LieG(mLeft, mField.getElems(x)[mu] * dt_);
					U = mLeft * U;
					break;
				}
				case 100: // test case
				case 1: {
					LieA2LieG(mLeft, mField.getElems(y)[mu] * dt_);
					LieA2LieG(mRight, \
						mField.getElems(x)[mu] * -dt_);
					U = mLeft * U * mRight;
					break;
				}
				case 10: {
					LieA2LieG(mRight, \
						mField.getElems(x)[mu] * -dt_);
					U = U * mRight;
					break;
				}
		// 		case 100: {
		// 			LieA2LieG(mLeft, mField.getElems(y)[mu] * dt_);
		// 			U = mLeft * U;
		// 			break;
		// 		}
				default: assert(false);
				}
		}}
	}

	inline double getHamiltonian(){
		static double oldKE = 0., oldPE = 0., oldTotalE = 0.;
		double localSum = 0.; // local sum of tr(\pi*\pi^\dagger)
#pragma omp parallel for reduction(+:localSum)
		for(long index = 0; index < mField.geo.localVolume(); index++){
			for(int mu = 0; mu < DIM; mu++){
				Coordinate x; 
				mField.geo.coordinateFromIndex(x, index);
				switch(isConstrained(x, mu, arg.mag)){
					// case 100: break;
					case 100: // test case
					case 0:
					case 1:
					case 10:{
						Matrix mTemp = \
							mField.getElemsConst(x)[mu];
						localSum += (mTemp * mTemp).ReTr();
						break;
					}
					default: assert(false);
				}
		}}
		double globalSum;
		MPI_Allreduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, getComm());
		double kineticEnergy = globalSum / 2.;
		double potentialEnergy = -totalPlaq(gField) * arg.beta / 3.;
		if(getIdNode() == 0){
	// 		cout << "kinEng = \t" << kineticEnergy << endl;
	// 		cout << "potEng = \t" << potentialEnergy << endl;
	// 		cout << "totEng = \t" 
	// 			<< kineticEnergy + potentialEnergy << endl;
	// 		cout << "delKin = \t" << kineticEnergy - oldKE << endl;
	//		cout << "delPot = \t" << potentialEnergy - oldPE << endl;
			// cout << "ratio = " << (kineticEnergy - oldKE) / (potentialEnergy - oldPE) << endl;
	//		cout << "delPer = \t" << (kineticEnergy + potentialEnergy - oldTotalE) / oldTotalE << endl;
		}

		oldKE = kineticEnergy;
		oldPE = potentialEnergy;
		oldTotalE = kineticEnergy + potentialEnergy;

		return kineticEnergy + potentialEnergy;
	}

public:
	double oldH, newH;
	double dieRoll;
	double deltaH;
	double percentDeltaH;
	double acceptProbability;
	double avgPlaq;
	bool doesAccept;
	int numAccept;
	int numReject;

	inline algCHmcWilson(argCHmcWilson arg_):
		globalRngState("By the witness of the martyrs.")
	{
		arg = arg_;
		qlat::Geometry geo_ = arg.gFieldExt->geo;
		
		assert(geo_.expansionLeft[0] > 0 && geo_.expansionRight[0] > 0);
		assert(geo_.expansionLeft[1] > 0 && geo_.expansionRight[1] > 0);
		assert(geo_.expansionLeft[2] > 0 && geo_.expansionRight[2] > 0);
		assert(geo_.expansionLeft[3] > 0 && geo_.expansionRight[3] > 0);
	
		gField.init(geo_); gField = *(arg.gFieldExt);
		mField.init(gField.geo);

		numAccept = 0;
		numReject = 0;
	}

	inline void initMomentum(){
		long rnSize = mField.geo.localVolume() * \
				mField.geo.multiplicity * SU3_NUM_OF_GENERATORS;
		vector<double> omega(rnSize); rnFillingSHA256Gaussian(omega);

#pragma omp parallel for
		for(long index = 0; index < mField.geo.localVolume(); index++){
			Coordinate x; mField.geo.coordinateFromIndex(x, index);
			Matrix mTemp;
			long fund;
			for(int mu = 0; mu < mField.geo.multiplicity; mu++){
				mTemp.ZeroMatrix();
				fund = (index * mField.geo.multiplicity + mu) * \
							SU3_NUM_OF_GENERATORS;
				for(int a = 0; a < SU3_NUM_OF_GENERATORS; a++){
					mTemp += su3Generator[a] * omega[fund + a];
				}
				mField.getElems(x)[mu] = mTemp;
		}}
	}

	inline void runTraj(int trajNum){
		
		TIMER("algCHmcWilson::runTraj()");
		
		initMomentum();
		fetch_expanded(gField);
		oldH = getHamiltonian();
		for(int i = 0; i < arg.length; i++){

			evolveGaugeField(arg.dt / 2.);
			{
			// TIMER_VERBOSE("fetch_expanded");
			fetch_expanded(gField);
			}

			evolveMomentum(arg.dt);

			evolveGaugeField(arg.dt / 2.);
			{
			// TIMER_VERBOSE("fetch_expanded");
			fetch_expanded(gField);
			}

		// 	evolveGaugeField(xi * arg.dt);
		// 	fetch_expanded(gField);
		// 	
		// 	evolveMomentum(arg.dt / 2.);

		// 	evolveGaugeField((1 - 2. * xi) * arg.dt);
		// 	fetch_expanded(gField);

		// 	evolveMomentum(arg.dt / 2.);
		// 	
		// 	evolveGaugeField(xi * arg.dt);
		// 	fetch_expanded(gField);
				
			double avgPlaq_ = avg_plaquette(gField);
			if(getIdNode() == 0){
				cout << "Traj #" << trajNum + 1 
					<<  ", (literal)Step " << i + 1 
					<< ":\tavgPlaq = " << avgPlaq_
					<< endl;
			}

			// reunitarize(gField);
		}
		
		newH = getHamiltonian();
		dieRoll = uRandGen(globalRngState);
		deltaH = newH - oldH;
		percentDeltaH = deltaH / oldH;
		acceptProbability = exp(oldH - newH);
		doesAccept = (dieRoll < acceptProbability);
		MPI_Bcast((void *)&doesAccept, 1, MPI_BYTE, 0, getComm());
		// make sure that all the node make the same decision.
		if(doesAccept){
			if(getIdNode() == 0){
				cout << "End trajectory " << trajNum + 1
						<< ": ACCEPT trajectory." << endl;
				cout << "old Hamiltonian =\t" << oldH << endl;
				cout << "new Hamiltonian =\t" << newH << endl;
				cout << "exp(-Delta H) =  \t" 
							<< exp(oldH - newH) << endl;
				cout << "Die Roll =       \t" << dieRoll << endl; 
				cout << "Delta H =        \t" << deltaH << endl; 
				cout << "Delta H Ratio =  \t" << percentDeltaH << endl; 
				numAccept++;
			}
			*(arg.gFieldExt) = gField;
		}else{
			if(getIdNode() == 0){
					cout << "End trajectory " << trajNum + 1
						<< ": REJECT trajectory." << endl;
				cout << "old Hamiltonian =\t" << oldH << endl;
				cout << "new Hamiltonian =\t" << newH << endl;
				cout << "exp(-Delta H) =  \t" 
							<< exp(oldH - newH) << endl;
				cout << "Die Roll =       \t" << dieRoll << endl; 
				cout << "Delta H =        \t" << deltaH << endl; 
				cout << "Delta H Ratio =  \t" << percentDeltaH << endl; 
				numReject++;
			}
			gField = *(arg.gFieldExt);
		}
		fetch_expanded(gField);
		avgPlaq = avg_plaquette(gField);
	}

};







