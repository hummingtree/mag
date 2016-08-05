#pragma once

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

        if(select < minusOneThird){
                ret[0] = a;   ret[2] = c;   ret[4] = 0.;
                ret[6] = -c;  ret[8] = a;   ret[10] = 0.;
                ret[12] = 0.; ret[14] = 0.; ret[16] = 1.;

                ret[1] = b;   ret[3] = d;   ret[5] = 0.;
                ret[7] = d;   ret[9] = -b;  ret[11] = 0.;
                ret[13] = 0.; ret[15] = 0.; ret[17] = 0.;
        }else
        if(select > oneThird){
                ret[0] = 1.;  ret[2] = 0.;  ret[4] = 0.;
                ret[6] = 0.;  ret[8] = a;   ret[10] = c;
                ret[12] = 0.; ret[14] = -c; ret[16] = a;

                ret[1] = 1.;  ret[3] = 0.;  ret[5] = 0.;
                ret[7] = 0.;  ret[9] = b;   ret[11] = d;
                ret[13] = 0.; ret[15] = d;  ret[17] = -b;
        }else{
                ret[0] = a;   ret[2] = 0.;  ret[4] = c;
                ret[6] = 0.;  ret[8] = 1.;  ret[10] = 0.;
                ret[12] = -c; ret[14] = 0.; ret[16] = a;

                ret[1] = b;   ret[3] = 0.;  ret[5] = d;
                ret[7] = 0.;  ret[9] = 0.;  ret[11] = 0.;
                ret[13] = d;  ret[15] = 0.; ret[17] = -b;
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

	inline bool isConstrained(int *x, int mu, int mag)
	{
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
			}
			else
				VRB.Debug("REJECT\n\n");

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
			}
			else
				VRB.Debug("REJECT\n\n");

		}             /* End HIT_CNT loop for multiple hits.           */
	}

public:
	inline algGCtrnHeatBath(Lattice &lat, CommonArg *c_arg, gchbArg *arg)
				: Alg(lat, c_arg){
		
		cname = "algGCtrnHeatBath";
  		const char *fname = "algGCtrnHeatBath(L&,CommonArg*,GhbArg*)";
  		VRB.Func(cname, fname);
  		if(arg == NULL) ERR.Pointer(cname, fname, "arg");
		gchbArg_ = arg;
	}

	inline virtual ~algGCtrnHeatBath(){}

	inline void run()
	{
		TIMER("algGCtrnHeatBath::run()");

		VRB.Func(cname,fname);

		// Set the Lattice pointer
		//----------------------------------------------------------------
		Lattice& lat = AlgLattice();
		if (lat.Gclass() != G_CLASS_WILSON){
			ERR.General(cname, fname, \
					"Only correct for Wilson gauge action\n");
		}

		// relocate the heatbath kernal and set up rand seeds
		//--------------------------------------------------

		// Run the gauge heat bath
		//----------------------------------------------------------------
		for(int i = 0; i < gchbArg_->numIter; i++){

			// Heat bath
			// Checkerboard everything, do even or odd sites
			// Scan over local subvolume doing all even(odd) links
			// index for "x" is consistant with GsiteOffset: "x,y,z,t" order

			// The y[4] are the local coordinates for the 2^4 cube. 
			// We traverse the points on this cube, and do the corresponding
			// points on every other hypercube before moving on.

			int x[4];
			Matrix *mPtr = lat.GaugeField();
			for(int mu = 0; mu < 4; mu++){
			for(x[3] = 0; x[3] < GJP.TnodeSites(); x[3]++){ 
			for(x[2] = 0; x[2] < GJP.ZnodeSites(); x[2]++){
			for(x[1] = 0; x[1] < GJP.YnodeSites(); x[1]++){ 
			for(x[0] = 0; x[0] < GJP.XnodeSites(); x[0]++){
				LRG.AssignGenerator(x);
				if(isConstrained(x, mu, gchbArg_->mag)){
					if(x[mu] % gchbArg_->mag == gchbArg_->mag - 1){}
					else{
						int y[4]; memcpy(y, x, 4 * sizeof(int));
						y[mu]++;
						Matrix mStaple1, mStaple2;
						lat.Staple(mStaple1, x, mu);
						lat.Staple(mStaple2, y, mu);
						metropolisKernelPProd(
							mPtr[mu + lat.GsiteOffset(x)],
							mStaple1,
							mPtr[mu + lat.GsiteOffset(y)],
							mStaple2
							);
					}
				}else{
					Matrix mStaple;
					lat.Staple(mStaple, x, mu);
					metropolisKernel(
						mPtr[mu + lat.GsiteOffset(x)],
						mStaple
						);
				}			
			}}}}}
			lat.Reunitarize();
		}
	}
};

CPS_END_NAMESPACE

