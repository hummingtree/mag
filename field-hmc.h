#pragma once

#include <iostream>
#include <fstream>
#include <omp.h>

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
#include <qlat/field-rng.h>

#include "field-matrix.h"

using namespace cps;
using namespace qlat;
using namespace std;

#define SU3_NUM_OF_GENERATORS 8

static const double invSqrt2 = 1. / sqrt(2.);

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
	int trajLength;
	int numTraj;
	double beta;
	double dt;
	gAction gA;
	string exportAddress; // config output
	int outputInterval;
	int forceAccept;
	int outputStart;
};

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

	if(!mag) return 0; // this is not a constrained evolution: always return 0;

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

inline void getForce(Field<Matrix> &fField, const Field<Matrix> &gField,
			const argCHmcWilson &arg){
	TIMER("getFoece()");
	assert(isMatchingGeo(fField.geo, gField.geo));
#pragma omp parallel for
	for(long index = 0; index < fField.geo.localVolume(); index++){
		Coordinate x; 
		Matrix mStaple1, mStaple2, mTemp;
		fField.geo.coordinateFromIndex(x, index);
		for(int mu = 0; mu < fField.geo.multiplicity; mu++){
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
			case 100: mTemp.ZeroMatrix(); break;
		
			// test case start
		// 	case 100: {
		// 		getStapleDagger(mStaple1, gField, x, mu);
		// 		mTemp = mStaple1 * gField.getElemsConst(x)[mu] * -1.;
		// 		break;
		// 	} 
		// 	// test case end
	
		 	default: assert(false);
			}
	
			mTemp.TrLessAntiHermMatrix(); 
			fField.getElems(x)[mu] = \
				mTemp * qlat::Complex(0., arg.beta / 3.);
	}}
}

inline void evolveMomentum(Field<Matrix> &mField, 
				const Field<Matrix> &fField, double dt, 
				const argCHmcWilson &arg){
	TIMER("algCHmcWilson::evolveMomentum()");
	assert(isMatchingGeo(mField.geo, fField.geo));
#pragma omp parallel for
	for(long index = 0; index < mField.geo.localVolume(); index++){
		Coordinate x; 
		mField.geo.coordinateFromIndex(x, index);
		for(int mu = 0; mu < mField.geo.multiplicity; mu++){
			mField.getElems(x)[mu] += fField.getElemsConst(x)[mu] * dt;
	}}
}

inline void evolveGaugeField(Field<Matrix> &gField, 
				const Field<Matrix> &mField, double dt, 
				const argCHmcWilson &arg){
	TIMER("algCHmcWilson::evolveGaugeField()");
	assert(isMatchingGeo(mField.geo, gField.geo));
#pragma omp parallel for
	for(long index = 0; index < gField.geo.localVolume(); index++){
		Coordinate x; 
		gField.geo.coordinateFromIndex(x, index);
		Matrix mL, mR;
		for(int mu = 0; mu < gField.geo.multiplicity; mu++){
		// only works for Matrix
			Matrix &U = gField.getElems(x)[mu];
			Coordinate y(x); y[mu]--;
			switch(isConstrained(x, mu, arg.mag)){
			case 0: {
				LieA2LieG(mL, mField.getElemsConst(x)[mu] * dt);
				U = mL * U;
				break;
			}
			// case 100: // test case
			case 1: {
				LieA2LieG(mL, mField.getElemsConst(y)[mu] * dt);
				LieA2LieG(mR, mField.getElemsConst(x)[mu] * -dt);
				U = mL * U * mR;
				break;
			}
			case 10: {
				LieA2LieG(mR, mField.getElemsConst(x)[mu] * -dt);
				U = U * mR;
			break;
			}
			case 100: {
				LieA2LieG(mL, mField.getElemsConst(y)[mu] * dt);
				U = mL * U;
				break;
			}
			default: assert(false);
			}
	}}
}

inline void forceGradientIntegrator(Field<Matrix> &gField, Field<Matrix> &mField, 
					const argCHmcWilson &arg, Chart<Matrix> &chart){
    	TIMER("forceGradientIntegrator()"); 

	assert(isMatchingGeo(gField.geo, mField.geo));
	const double alpha = (3. - sqrt(3.)) * arg.dt / 6.;
	const double beta = arg.dt / sqrt(3.);
	const double gamma = (2. - sqrt(3.)) * arg.dt * arg.dt / 12.;
	
	static Field<Matrix> gFieldAuxil; gFieldAuxil.init(gField.geo);
	static Field<Matrix> fField; fField.init(mField.geo);

	evolveGaugeField(gField, mField, alpha, arg);
	
	for(int i = 0; i < arg.trajLength; i++){
		fetch_expanded_chart(gField, chart);
		getForce(fField, gField, arg);
		gFieldAuxil = gField;
		evolveGaugeField(gFieldAuxil, fField, gamma, arg);
		fetch_expanded_chart(gFieldAuxil, chart);
		getForce(fField, gFieldAuxil, arg);
		evolveMomentum(mField, fField, 0.5 * arg.dt, arg);

		evolveGaugeField(gField, mField, beta, arg);
	
		fetch_expanded_chart(gField, chart);
		getForce(fField, gField, arg);
		gFieldAuxil = gField;
		evolveGaugeField(gFieldAuxil, fField, gamma, arg);
		fetch_expanded_chart(gFieldAuxil, chart);
		getForce(fField, gFieldAuxil, arg);
		evolveMomentum(mField, fField, 0.5 * arg.dt, arg);

		if(i < arg.trajLength - 1) 
			evolveGaugeField(gField, mField, 2. * alpha, arg);
		else evolveGaugeField(gField, mField, alpha, arg);
	}
	report << "reunitarize: max deviation = " << reunitarize(gField) << endl;
}

inline void leapFrogIntegrator(Field<Matrix> &gField, Field<Matrix> &mField, 
				const argCHmcWilson &arg){
	TIMER("leapFrogIntegrator()");
	assert(isMatchingGeo(gField.geo, mField.geo));
	Geometry geo_; 
	geo_.init(gField.geo.geon, gField.geo.multiplicity, gField.geo.nodeSite);
	static Field<Matrix> fField; fField.init(geo_);
	evolveGaugeField(gField, mField, arg.dt / 2., arg);
	for(int i = 0; i < arg.trajLength; i++){
		fetch_expanded(gField);
		getForce(fField, gField, arg);
		evolveMomentum(mField, fField, arg.dt, arg);
		if(i < arg.trajLength - 1) 
			evolveGaugeField(gField, mField, arg.dt, arg);
		else evolveGaugeField(gField, mField, arg.dt / 2., arg);
	}
}

inline double getHamiltonian(Field<Matrix> &gField, const Field<Matrix> &mField,
				const argCHmcWilson &arg){
	
	TIMER("getHamiltonian()");
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
	for(long index = 0; index < mField.geo.localVolume(); index++){
		for(int mu = 0; mu < DIM; mu++){
			Coordinate x; mField.geo.coordinateFromIndex(x, index);
			switch(isConstrained(x, mu, arg.mag)){
				case 100: break;
				// case 100: // test case
				case 0:
				case 1:
				case 10:{
					Matrix mTemp = mField.getElemsConst(x)[mu];
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
	MPI_Allreduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, getComm());
	double kineticEnergy = globalSum / 2.;
	fetch_expanded(gField);
	double potentialEnergy = -totalPlaq(gField) * arg.beta / 3.;
	report << "POTENTIAL ENERGY = " << potentialEnergy << endl;
	return kineticEnergy + potentialEnergy;
}

inline void initMomentum(Field<Matrix> &mField){
	TIMER("initMomemtum()");

	using namespace qlat;
	static bool initialized = false;
	static Geometry rng_geo;
	static RngField rng_field;
	if(initialized == false){
		rng_geo.init(mField.geo.geon, 1, mField.geo.nodeSite);
		rng_field.init(rng_geo, RngState("Ich liebe dich."));
		initialized = true;
	}

#pragma omp parallel for
	for(long index = 0; index < mField.geo.localVolume(); index++){
		Coordinate x; mField.geo.coordinateFromIndex(x, index);
		Matrix mTemp;
		for(int mu = 0; mu < mField.geo.multiplicity; mu++){
			mTemp.ZeroMatrix();
			for(int a = 0; a < SU3_NUM_OF_GENERATORS; a++){
				mTemp += su3Generator[a] * gRandGen(rng_field.getElem(x));
			}
			mField.getElems(x)[mu] = mTemp;
	}}
}

inline double derivative(const Field<Matrix> &gField, Coordinate &local, int mu, int a){
	Matrix &U = gField.getElemsConst(local)[mu];
	Matrix V_dagger; getStapleDagger(V_dagger, gField, local, mu);
	Matrix temp = su3Generator[a] * U * V_dagger * qlat::Complex(0., 1.);
	return temp.ReTr();
}

inline double derivative_sum(Field<Matrix> &gField, const argCHmcWilson &arg){
	fetch_expanded(gField);
	double local_sum = 0.;
	for(int x = 0; x < gField.geo.nodeSite[0]; x += arg.mag){
	for(int y = 0; y < gField.geo.nodeSite[1]; y += arg.mag){
	for(int z = 0; z < gField.geo.nodeSite[2]; z += arg.mag){
	for(int t = 0; t < gField.geo.nodeSite[3]; t += arg.mag){
	for(int mu = 0; mu < DIM; mu++){
	for(int a = 0; a < 8; a++){
		Coordinate coor(x, y, z, t);
		local_sum += derivative(gField, coor, mu, a);
	}}}}}}

	double global_sum;
	MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, getComm());

	return global_sum;
}

inline void runHMC(Field<Matrix> &gFieldExt, const argCHmcWilson &arg, FILE *pFile){
	TIMER("algCHmcWilson::runHMC()");
	assert(pFile != NULL);
	assert(arg.numTraj > 20);

	RngState globalRngState("By the witness of the martyrs.");

	Geometry geoExpand1; geoExpand1.copyButExpand(gFieldExt.geo, 1);
	Geometry geoLocal; geoLocal.copyOnlyLocal(gFieldExt.geo);
	Field<Matrix> gField; gField.init(geoExpand1); gField = gFieldExt;
	Field<Matrix> mField; mField.init(geoLocal);

	Chart<Matrix> chart;
	produce_chart_envelope(chart, gFieldExt.geo, arg.gA);

	double oldH, newH;
	double dieRoll;
	double deltaH, percentDeltaH;
	double acceptProbability;
	double avgPlaq;
	bool doesAccept;
	int numAccept = 0, numReject = 0;

	for(int i = 0; i < arg.numTraj; i++){
		initMomentum(mField);
		
		oldH = getHamiltonian(gField, mField, arg);
		// leapFrogIntegrator(gField, mField, arg);
		forceGradientIntegrator(gField, mField, arg, chart);
		
		newH = getHamiltonian(gField, mField, arg);
	
		dieRoll = uRandGen(globalRngState);
		deltaH = newH - oldH;
		percentDeltaH = deltaH / oldH;
		acceptProbability = exp(oldH - newH);
		doesAccept = (dieRoll < acceptProbability);
		MPI_Bcast((void *)&doesAccept, 1, MPI_BYTE, 0, getComm());
		// make sure that all the node make the same decision.
		
		if(i < arg.forceAccept){
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

		double dv_sum = derivative_sum(gField, arg);
		report << "FINE DERIVATIVE =\t" << dv_sum << endl;

		if(getIdNode() == 0){
			fprintf(pFile, "%i\t%.6e\t%.6e\t%.12e\t%.12e\t%i\n", i + 1, 
				abs(deltaH), acceptProbability, avgPlaq, dv_sum, doesAccept);
			fflush(pFile);
		}

		if((i + 1) % arg.outputInterval == 0 && i + 1 >= arg.outputStart){
			argExport argExport_;
			argExport_.beta = arg.beta;
			argExport_.sequenceNum = i + 1;
			argExport_.ensembleLabel = "constrained_hmc";
			if(arg.exportAddress.size() > 0){
				string address = arg.exportAddress + "ckpoint." + show(i + 1);
				export_config_nersc(gFieldExt, address, argExport_, true);
			}
			if(getIdNode() == 0) printf("ACCEPT RATE = %.3f\n",
										(double)numAccept / (numAccept + numReject));
		}
	}

	if(getIdNode() == 0){
		fprintf(pFile, "Accept Rate = %.3f\n", 
				(double)numAccept / (numAccept + numReject));
		fflush(pFile);
	}

	Timer::display();
}

