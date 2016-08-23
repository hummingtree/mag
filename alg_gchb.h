#pragma once

#include <iostream>

#include <config.h>
#include <stdlib.h>	
#include <util/qcdio.h>
#include <math.h>
#include <time.h>
#include <alg/alg_base.h>
#include <alg/common_arg.h>
#include <alg/ghb_arg.h>
#include <util/lattice.h>
#include <util/gjp.h>
#include <util/random.h>
#include <util/smalloc.h>
#include <util/vector.h>
#include <util/verbose.h>
#include <util/error.h>

CPS_START_NAMESPACE

Float oneThird = 1. / 3., minusOneThird = -1. / 3.;

//------------------------------------------------------------------
/*!
  \param latt The lattice on which to perform the heatbath
  \param c_arg The common argument structure for all algorithms.
  \param arg The parameters specific to this algorithm.
 */
//------------------------------------------------------------------

Matrix getRandomSU3BySubsu2(double small){

        LRG.SetInterval(1., -1.);

        double a, b, c, d, norm, select;
        Matrix ret;

        a = small * LRG.Urand(); if(a > 0) a *= -1.; a += 1.;
        b = LRG.Urand();
        c = LRG.Urand();
        d = LRG.Urand();

        VRB.Debug("unnormalized:\ta = %+6.5f b = %+6.5f c = %+6.5f d = %+6.5f\n", \
									a, b, c, d);

        norm = sqrt((1. - a * a) / (b * b + c * c + d * d));

        b *= norm;
        c *= norm;
        d *= norm;

        VRB.Debug("normalized:\ta = %+6.5f b = %+6.5f c = %+6.5f d = %+6.5f\n", \
									a, b, c, d);

        select = LRG.Urand();

	Float *mPtr = (Float *)&ret;

        if(select < minusOneThird){
                mPtr[0] = a;   mPtr[2] = c;   mPtr[4] = 0.;
                mPtr[6] = -c;  mPtr[8] = a;   mPtr[10] = 0.;
                mPtr[12] = 0.; mPtr[14] = 0.; mPtr[16] = 1.;

                mPtr[1] = b;   mPtr[3] = d;   mPtr[5] = 0.;
                mPtr[7] = d;   mPtr[9] = -b;  mPtr[11] = 0.;
                mPtr[13] = 0.; mPtr[15] = 0.; mPtr[17] = 0.;
        }else
        if(select > oneThird){
                mPtr[0] = 1.;  mPtr[2] = 0.;  mPtr[4] = 0.;
                mPtr[6] = 0.;  mPtr[8] = a;   mPtr[10] = c;
                mPtr[12] = 0.; mPtr[14] = -c; mPtr[16] = a;

                mPtr[1] = 0.;  mPtr[3] = 0.;  mPtr[5] = 0.;
                mPtr[7] = 0.;  mPtr[9] = b;   mPtr[11] = d;
                mPtr[13] = 0.; mPtr[15] = d;  mPtr[17] = -b;
        }else{
                mPtr[0] = a;   mPtr[2] = 0.;  mPtr[4] = c;
                mPtr[6] = 0.;  mPtr[8] = 1.;  mPtr[10] = 0.;
                mPtr[12] = -c; mPtr[14] = 0.; mPtr[16] = a;

                mPtr[1] = b;   mPtr[3] = 0.;  mPtr[5] = d;
                mPtr[7] = 0.;  mPtr[9] = 0.;  mPtr[11] = 0.;
                mPtr[13] = d;  mPtr[15] = 0.; mPtr[17] = -b;
        }

	if(a * a + b * b + c * c + d * d > 1.01 || a * a + b * b + c * c + d * d < 0.99)
	{
		std::cout << "a = " << a << std::endl;
		std::cout << "b = " << b << std::endl;
		std::cout << "c = " << c << std::endl;
		std::cout << "d = " << d << std::endl;
	}

        return ret;
}

class gchbArg{
public:
	int mag; // what product to preserve
	int numIter;
	int nHits;
	Float small;
};

class algGCtrnHeatBath: public Alg
{
private:
	const char *cname;

	gchbArg *gchbArg_;

	long reject;
	long accept;
	long rejectCons;
	long acceptCons;

	inline bool isConstrained(int *x, int mu, int mag)
	{
		// return false; // test case
		bool isConstrained_ = true;
		for(int i = 0; i < 4; i++){
			if(i == mu) continue;
			isConstrained_ = isConstrained_ && (x[i] % mag == 0);
		}
		return isConstrained_;
	}
	inline void metropolisKernel(Matrix &U, const Matrix &sigma)
	{
		// metropolis updating one links at the same time

		LRG.SetInterval(1., -1.);

		VRB.Debug("beta = %f\n", GJP.Beta());
		Float fBeta = GJP.Beta() / 3.;

		Matrix epsilon;
		Matrix mTemp1, mTemp2;

		Float  oldAction;
		Float  newAction;
		Float  acceptProbability;
		Float  dieRoll;

		VRB.Debug("metropolisKernel: nHits=%d,\tsmall=%f.\n", \
						gchbArg_->nHits, gchbArg_->small);

		for(int hit = 0; hit < gchbArg_->nHits; hit++)
		{
			epsilon = getRandomSU3BySubsu2(gchbArg_->small);

			// Now begin the update process. 

			mTemp1 = U * sigma;
			oldAction = -fBeta * mTemp1.ReTr();

			VRB.Debug("Old action = %e\n", oldAction );

			mTemp1 = U * epsilon;
			mTemp2 = mTemp1 * sigma;

			newAction = -fBeta * mTemp2.ReTr();

			VRB.Debug("New action = %e\n", newAction );

			acceptProbability = exp(oldAction - newAction);

			VRB.Debug("Accept chance = %e\n", acceptProbability);

			dieRoll = LRG.Urand();
			if(dieRoll < 0.) dieRoll *= -1.;

			VRB.Debug("die roll =  %e\n", dieRoll );


			if(dieRoll < acceptProbability){
				VRB.Debug("ACCEPT\n\n");
				U = mTemp1;
				accept++;
			}
			else{
				VRB.Debug("REJECT\n\n");
				reject++;
			}
		}             /* End HIT_CNT loop for multiple hits.           */
	}

	inline void metropolisKernelPProd(Matrix &U1, const Matrix &sigma1,
			Matrix &U2, const Matrix &sigma2)
	{
		// metropolis updating two consequtive links at the same time

		LRG.SetInterval(1., -1.);

		VRB.Debug("beta = %f\n", GJP.Beta());
		Float fBeta = GJP.Beta() / 3.;

		Matrix epsilon, epsilonDagger;
		Matrix mTemp1, mTemp2, mTemp3, mTemp4;

		Float  oldAction;
		Float  newAction;
		Float  acceptProbability;
		Float  dieRoll;

		VRB.Debug("metropolisKernelPProd: nHits=%d,\tsmall=%f.\n", \
						gchbArg_->nHits, gchbArg_->small);

		for(int hit = 0; hit < gchbArg_->nHits; hit++)
		{
			epsilon = getRandomSU3BySubsu2(gchbArg_->small);
			epsilonDagger.Dagger(epsilon);


			// Now begin the update process. 

			mTemp1 = U1 * sigma1;
			mTemp2 = U2 * sigma2;
			oldAction = -fBeta * mTemp1.ReTr();
			oldAction += -fBeta * mTemp2.ReTr();

			VRB.Debug("Old action = %e\n", oldAction );

			mTemp1 = U1 * epsilon;
			mTemp2 = epsilonDagger * U2;

			mTemp3 = mTemp1 * sigma1;
			mTemp4 = mTemp2 * sigma2;

			newAction = -fBeta * mTemp3.ReTr();
			newAction += -fBeta * mTemp4.ReTr();

			VRB.Debug("New action = %e\n", newAction );

			acceptProbability = exp(oldAction - newAction);

			VRB.Debug("Accept chance = %e\n", acceptProbability);

			dieRoll = LRG.Urand();
			if(dieRoll < 0.) dieRoll *= -1.;

			VRB.Debug("die roll =  %e\n", dieRoll );


			if(dieRoll < acceptProbability){
				VRB.Debug("ACCEPT\n\n");
				U1 = mTemp1;
				U2 = mTemp2;
				acceptCons++;
			}
			else{
				VRB.Debug("REJECT\n\n");
				rejectCons++;
			}

		}             /* End HIT_CNT loop for multiple hits.           */
	}

public:
	inline void showInfo(){
	
		cout << "mag     =\t" << gchbArg_->mag << endl;
		cout << "numIter =\t" << gchbArg_->numIter << endl;
		cout << "nHits   =\t" << gchbArg_->nHits << endl;
		cout << "small   =\t" << gchbArg_->small << endl;
		cout << "beta    =\t" << GJP.Beta() << endl;
		
	}
		
	inline void accpetRate(){
		cout << "Normal Metropolis: " 
			<< (double)accept / (accept + reject)
			<< endl;
		cout << "Constrained Metropolis: " 
			<< (double)acceptCons / (acceptCons + rejectCons)
			<< endl;
		cout << "Total: "
			<< (double)(accept + acceptCons) / (accept + acceptCons + reject + rejectCons)
			<< endl;
	}

	inline algGCtrnHeatBath(Lattice &lat, CommonArg *c_arg, gchbArg *arg)
				: Alg(lat, c_arg){
		
		cname = "algGCtrnHeatBath";
  		const char *fname = "algGCtrnHeatBath(L&,CommonArg*,GhbArg*)";
  		VRB.Func(cname, fname);
  		if(arg == NULL) ERR.Pointer(cname, fname, "arg");
		gchbArg_ = arg;

		accept = 0;
		reject = 0;
		acceptCons = 0;
		rejectCons = 0;
		
	}

	inline virtual ~algGCtrnHeatBath(){}

	inline void run()
	{
		TIMER_VERBOSE("algGCtrnHeatBath::run()");

		VRB.Func(cname,fname);

		// Set the Lattice pointer
		//----------------------------------------------------------------
		Lattice& lat = AlgLattice();
 		if (lat.Gclass() != G_CLASS_WILSON){
 			ERR.General(cname, fname, \
 					"Only correct for Wilson gauge action\n");
 		}

		// Run the gauge heat bath
		//----------------------------------------------------------------
		for(int i = 0; i < gchbArg_->numIter; i++){
			int x[4];
			Matrix *mPtr = lat.GaugeField();
			for(int mut = 0; mut < 4; mut++){
			for(x[3] = 0; x[3] < GJP.TnodeSites(); x[3]++){ 
			for(x[2] = 0; x[2] < GJP.ZnodeSites(); x[2]++){
			for(x[1] = 0; x[1] < GJP.YnodeSites(); x[1]++){ 
			for(x[0] = 0; x[0] < GJP.XnodeSites(); x[0]++){
				
				LRG.AssignGenerator(x);
				
				if(isConstrained(x, mut, gchbArg_->mag)){
				if(x[mut] % gchbArg_->mag == gchbArg_->mag - 1){
			// test case start
			
					Matrix mStaple;
					lat.Staple(mStaple, x, mut);
					long localIndexCPS; 
					GJP.localIndexFromPos(x, localIndexCPS);
					metropolisKernel(
						mPtr[mut + 4 * localIndexCPS],
						mStaple
						);
			//
			// test case end
				}
					else{
						int y[4]; memcpy(y, x, 4 * sizeof(int));
						y[mut]++;
						Matrix mStaple1, mStaple2;
						lat.Staple(mStaple1, x, mut);
						lat.Staple(mStaple2, y, mut);
						metropolisKernelPProd(
							mPtr[mut + lat.GsiteOffset(x)],
							mStaple1,
							mPtr[mut + lat.GsiteOffset(y)],
							mStaple2
							);
					}
				}else{

					Matrix mStaple;
					lat.Staple(mStaple, x, mut);
					long localIndexCPS; 
					GJP.localIndexFromPos(x, localIndexCPS);
					metropolisKernel(
						mPtr[mut + 4 * localIndexCPS],
						mStaple
						);
				}			
	 		}}}}}
	 		lat.Reunitarize();

			double avgPlaq = lat.SumReTrPlaq() / (18. * GJP.VolSites());
			double avgConsPlaq = check_constrained_plaquette(lat, gchbArg_->mag);

			if(UniqueID() == 0){
		// 		cout 
		// 		<< "Thermalization Cycle =\t" << i << endl
		// 		<< "avgPlaq              =\t" << avgPlaq << endl
		// 	 	<< "avgConsPlaq          =\t" << avgConsPlaq << endl;
				cout 
				<< i << "\t"
				<< avgPlaq << "\t"
				<< avgConsPlaq << endl;
		}

		}

	}
};

CPS_END_NAMESPACE

