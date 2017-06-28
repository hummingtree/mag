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
#include <cmath>

using namespace cps;
using namespace qlat;
using namespace std;

QLAT_START_NAMESPACE

// notations follow hep-lat/0311018
inline cps::Matrix get_U(Field<cps::Matrix>& fine_gField, const qlat::Coordinate& x, int mu){
	cps::Coordinate y = x; y[mu]++;
	return fine_gField.get_elems(x)[mu] * fine_gField.get_elems(y)[mu];
}

inline cps::Matrix get_Q(Field<cps::Matrix>& fine_gField, const qlat::Coordinate& x, int mu, double rho){
	// assuming properly communicated.
	cps::Matrix stp;
	get_staple_2x1(stp, fine_gField, x, mu);
	cps::Matrix dg; dg.Dagger(get_U(fine_gField, x, mu));
	return rho*dg*stp;
}

inline cps::Matrix hermitian_traceless(const cps::Matrix& M){
	cps::Matrix one; one.UnitMatrix();
	cps::Matrix dg; dg.Dagger();
	return ((M+dg)-(M+dg).Tr()/3.*one)/2.;
}

inline cps::Matrix compute_Lambda(const cps::Matrix& Q, const cps::Matrix& SigmaP, const cps::Matrix& U){
	double c0 = (Q * Q * Q).ReTr() / 3.;
	bool reflect = false;
	if(c0 < 0.){
		c0 = -c0;
		reflect = true;
	}
	double c1 = (Q * Q).ReTr() / 2.;
	double c0m = 2.* pow(c1/3., 1.5);
	double theta = acos(c0/c0m);
	double u = sqrt(c1/3.) * cos(theta/3.);
	double w = sqrt(c1) * sin(theta/3.);

	// qlat::Printf("u=%.12f\tw=%.12f\n", u, w);	

	double xi0;
	double xi1;
	if(w*w < 0.05*0.05){
		xi0 = 1.-w*w/6.*(1.-w*w/20.*(1.-w*w/42.));
		xi1 = -1./2.*(1.-w*w/12.*(1.-w*w/30.)) + 1./6.*(1.-w*w/20.*(1.-w*w/42.));
	}
	else{
		xi0 = sin(w)/w;
		xi1 = cos(w)/(w*w)-sin(w)/(w*w*w);
	}
	qlat::Complex h0 = (u*u-w*w)*expix(2.*u) + expix(-u)*(8.*u*u*cos(w)+i()*2.*u*(3.*u*u+w*w)*xi0);
	qlat::Complex h1 = 2.*u*expix(2.*u) - expix(-u)*(2.*u*cos(w)-i()*(3.*u*u-w*w)*xi0);
	qlat::Complex h2 = expix(2.*u)-expix(-u)*(cos(w)+i()*3.*u*xi0);
	qlat::Complex f0 = h0 / (9.*u*u-w*w);
	qlat::Complex f1 = h1 / (9.*u*u-w*w);
	qlat::Complex f2 = h2 / (9.*u*u-w*w);

	qlat::Complex r01 = 2.*(u+i()*(u*u-w*w))*expix(2.*u)+2.*expix(-u)*(4.*u*(2.-i()*u)*cos(w)+i()*xi0*(9.*u*u+w*w-i()*u*(3.*u*u+w*w)));
	qlat::Complex r11 = 2.*(1.+2.*i()*u)*expix(2.*u) + expix(-u)*( -2.*(1.-i()*u)*cos(w) + i()*xi0*(6.*u+i()*(w*w-3.*u*u)) );
	qlat::Complex r21 = 2.*i()*expix(2.*u)+i()*expix(-u)*( cos(w)-3.*xi0*(1.-i()*u) );
	qlat::Complex r02 = -2.*expix(2.*u) + 2.*i()*u*expix(-u)*( cos(w) + (1.+4.*i()*u*xi0) + 3.*u*u*xi1 );
	qlat::Complex r12 = -i()*expix(-u) * ( cos(w) + (1.+2.*i()*u)*xi0 - 3.*u*u*xi1 );
	qlat::Complex r22 = expix(-u) * (xi0 - 3.*i()*u*xi1);

	b10 = (2.*u*r01 + (3.*u*u-w*w)*r02 - 2.*(15.*u*u+w*w)*f0) / ( 2.*(9.*u*u-w*w)*(9.*u*u-w*w) );
	b11 = (2.*u*r11 + (3.*u*u-w*w)*r12 - 2.*(15.*u*u+w*w)*f1) / ( 2.*(9.*u*u-w*w)*(9.*u*u-w*w) );
	b12 = (2.*u*r21 + (3.*u*u-w*w)*r22 - 2.*(15.*u*u+w*w)*f2) / ( 2.*(9.*u*u-w*w)*(9.*u*u-w*w) );

	b20 = (r01 + 3.*u*r02 - 24.*u*f0) / ( 2.*(9.*u*u-w*w)*(9.*u*u-w*w) );
	b21 = (r11 + 3.*u*r12 - 24.*u*f1) / ( 2.*(9.*u*u-w*w)*(9.*u*u-w*w) );
	b22 = (r21 + 3.*u*r22 - 24.*u*f2) / ( 2.*(9.*u*u-w*w)*(9.*u*u-w*w) );

	if(reflect){
		f0 = conj(f0);
		f1 = -conj(f1);
		f2 = conj(f2);
	
		b10 = conj(b10);
		b11 = -conj(b11);
		b12 = conj(b12);
		b20 = -conj(b20);
		b21 = conj(b21);
		b22 = -conj(b22);
	}

	// qlat::Printf("f0=%.12f\tf0=%.12f\n", f0.real(), f0.imag());	
	// qlat::Printf("f1=%.12f\tf1=%.12f\n", f1.real(), f1.imag());	
	// qlat::Printf("f2=%.12f\tf2=%.12f\n", f2.real(), f2.imag());	
	
	cps::Matrix one; one.UnitMatrix();
	cps::Matrix B1 = b10 * one + b11 * Q + b12 * Q * Q;
	cps::Matrix B2 = b20 * one + b21 * Q + b22 * Q * Q;

	cps::Matrix Gamma = (SigmaP*B1*U).Tr() * Q + (SigmaP*B2*U).Tr() * Q * Q + f1*U*SigmaP + f2*Q*U*SigmaP + f2*Q*SigmaP*Q;
	return hermitian_traceless(Gamma);
}

inline void force_gradient_integrator_multi(
	Field<cps::Matrix>& FgField, Field<cps::Matrix>& FmField, 
	Field<cps::Matrix>& FgFieldAuxil, Field<cps::Matrix>& FfField,
	const Arg_chmc& Farg, 
	Chart<cps::Matrix>& Fchart,
	Field<cps::Matrix>& CgField, Field<cps::Matrix>& CmField, 
	Field<cps::Matrix>& CgFieldAuxil, Field<cps::Matrix>& CfField,
	Chart<cps::Matrix>& Cchart
){
	// See mag.pdf for notations.

	sync_node();
	TIMER("force_gradient_integrator_multi()"); 

	assert(is_matching_geo(FgField.geo, FmField.geo));
	assert(is_matching_geo(FgField.geo, FfField.geo));
	assert(is_matching_geo(FgField.geo, FgFieldAuxil.geo));
	assert(is_matching_geo(CgField.geo, CmField.geo));
	assert(is_matching_geo(CgField.geo, CfField.geo));
	assert(is_matching_geo(CgField.geo, CgFieldAuxil.geo));
	const double alpha = (3.-sqrt(3.))*Farg.dt/6.;
	const double beta = Farg.dt/sqrt(3.);
	const double gamma = (2.-sqrt(3.))*Farg.dt*Farg.dt/12.;

	evolve_gauge_field(FgField, FmField, alpha, Farg);
	evolve_gauge_field(CgField, CmField, alpha, Farg);

	sync_node();
	for(int i = 0; i < Farg.trajectory_length; i++){
		fetch_expanded_chart(FgField, Fchart);
		fetch_expanded_chart(CgField, Cchart);
		// TODO!!!
		get_Fforce(FfField, FgField, CgField, Farg);
		get_Cforce(CfField, FgField, CgField, Farg);
		FgFieldAuxil = FgField;
		CgFieldAuxil = CgField;
		evolve_gauge_field(FgFieldAuxil, FfField, gamma, Farg);
		evolve_gauge_field(CgFieldAuxil, CfField, gamma, Farg);
		fetch_expanded_chart(FgFieldAuxil, Cchart);
		fetch_expanded_chart(CgFieldAuxil, Cchart);
		// TODO!!!
		get_Fforce(FfField, FgFieldAuxil, CgFieldAuxil, Farg);
		get_Cforce(CfField, FgFieldAuxil, CgFieldAuxil, Farg);
		evolve_momentum(FmField, FfField, 0.5 * Farg.dt, Farg);
		evolve_momentum(CmField, CfField, 0.5 * Farg.dt, Farg);

		evolve_gauge_field(FgField, FmField, beta, Farg);
		evolve_gauge_field(CgField, CmField, beta, Farg);
	
		fetch_expanded_chart(FgField, Fchart);
		fetch_expanded_chart(CgField, Cchart);
		// TODO!!!
		get_Fforce(FfField, FgField, CgField, Farg);
		get_Cforce(CfField, FgField, CgField, Farg);
		FgFieldAuxil = FgField;
		CgFieldAuxil = CgField;
		evolve_gauge_field(FgFieldAuxil, FfField, gamma, Farg);
		evolve_gauge_field(CgFieldAuxil, CfField, gamma, Farg);
		fetch_expanded_chart(FgFieldAuxil, Cchart);
		fetch_expanded_chart(CgFieldAuxil, Cchart);
		// TODO!!!
		get_Fforce(FfField, FgFieldAuxil, CgFieldAuxil, Farg);
		get_Cforce(CfField, FgFieldAuxil, CgFieldAuxil, Farg);
		evolve_momentum(FmField, FfField, 0.5 * Farg.dt, Farg);
		evolve_momentum(CmField, CfField, 0.5 * Farg.dt, Farg);

		if(i < arg.trajectory_length - 1){
			evolve_gauge_field(FgField, FmField, 2. * alpha, Farg);
			evolve_gauge_field(CgField, CmField, 2. * alpha, Farg);
		} 
		else{
			evolve_gauge_field(FgField, FmField, alpha, Farg);
			evolve_gauge_field(CgField, CmField, alpha, Farg);
		}
	}

	qlat::Printf("reunitarize FgField: max deviation = %.8e\n", reunitarize(FgField));
	qlat::Printf("reunitarize CgField: max deviation = %.8e\n", reunitarize(CgField));
}

inline void run_hmc_multi(
	Field<cps::Matrix>& FgField_ext, const Arg_chmc &Farg,
	Field<cps::Matrix>& CgField_ext,
){
	
	// perform a number of fine updates and them a number of coarse updates.
	
	TIMER("run_hmc_multi");
	
	FILE *p = NULL;
	if(arg.summary_dir_stem.size() > 0){
		p = Fopen((Farg.summary_dir_stem + "/summary.dat").c_str(), "a");
	}

	if(!get_id_node()) assert(p != NULL);

	time_t now = time(NULL);
	Fprintf(p, "# %s", 							ctime(&now));
	Fprintf(p, "# %s\n", 						show(gField_ext.geo).c_str());
	Fprintf(p, "# mag               = %i\n", 	Farg.mag);
	Fprintf(p, "# trajectory_length = %i\n", 	Farg.trajectory_length);
	Fprintf(p, "# num_trajectory    = %i\n", 	Farg.num_trajectory);
	Fprintf(p, "# beta              = %.6f\n", 	Farg.beta);
	Fprintf(p, "# dt                = %.5f\n", 	Farg.dt);
	Fprintf(p, "# c1                = %.5f\n", 	Farg.gauge.c1);
	Fprintf(p, "# GAUGE_TYPE        = %d\n", 	Farg.gauge.type);
	Fprintf(p, "# traj. number\texp(-DeltaH)\tavgPlaq\taccept/reject\n");
	Fflush(p);
	qlat::Printf("p opened.");

	assert(arg.num_trajectory > 20);
	RngState globalRngState("By the witness of the martyrs.");

	qlat::Coordinate expansion(2, 2, 2, 2);

	// declare fine lattice variables
	Geometry Fgeo_expanded = FgField_ext.geo; Fgeo_expanded.resize(expansion, expansion);
	Geometry Fgeo_local = FgField_ext.geo;
	Field<cps::Matrix> FgField; 		FgField.init(Fgeo_expanded); FgField = FgField_ext;
	Field<cps::Matrix> FgField_auxil; 	FgField_auxil.init(Fgeo_expanded);
	Field<cps::Matrix> FmField; 		FmField.init(Fgeo_expanded);
	Field<cps::Matrix> FfField; 		FfField.init(Fgeo_expanded);

	Geometry Frng_geo; 
	Frng_geo.init(FmField.geo.geon, 1, FmField.geo.node_site);
	RngField Frng_field; 
	Frng_field.init(Frng_geo, RngState("Ich liebe dich."));

	//declare coarse lattice variables
	Geometry Cgeo_expanded = CgField_ext.geo; Cgeo_expanded.resize(expansion, expansion);

	Field<cps::Matrix> CgField; CgField.init(Cgeo_expanded); CgField = CgField_ext;
	Field<cps::Matrix> CgField_auxil; CgField_auxil.init(Cgeo_expanded);
	Field<cps::Matrix> CmField; CmField.init(Cgeo_expanded);
	Field<cps::Matrix> CfField; CfField.init(Cgeo_expanded);

	fetch_expanded(CgField);
	fetch_expanded(FgField);
    qlat::Printf("COARSE Plaquette = %.12f\n", avg_plaquette(FgField));	
    qlat::Printf("FINE   Plaquette = %.12f\n", avg_plaquette(CgField));	

	Geometry Crng_geo; 
	Crng_geo.init(CmField.geo.geon, 1, CmField.geo.node_site);
	RngField Crng_field; 
	Crng_field.init(Crng_geo, RngState("Tut mir leid."));

	// declare the communication patterns
	Chart<cps::Matrix> Fchart;
	produce_chart_envelope(Fchart, FgField_ext.geo, Farg.gauge);
	Chart<cps::Matrix> Cchart;
	produce_chart_envelope(Cchart, CgField_ext.geo, Farg.gauge);

	// start the hmc 
	double old_hamiltonian;
	double new_hamiltonian;
	double die_roll;
	double del_hamiltonian;
	double accept_probability;
	double average_plaquette;
	
	vector<double> old_energy_partition;
	vector<double> new_energy_partition;
	bool does_accept;
	int num_accept = 0;
	int num_reject = 0;

	// update fine and coarse lattices in one single hmc.
	for(int i = 0; i < Farg.num_trajectory; i++){

		init_momentum(FmField, Frng_field);
		init_momentum(CmField, Crng_field);
		
		// TODO!!!
		old_hamiltonian = get_hamiltonian(FgField, FmField, Farg, Fchart, CgField, CmField, Cchart, old_energy_partition);
		force_gradient_integrator_multi(FgField, FmField, FgField_auxil FfField, Farg, Fchart,
										CgField, CmField, CgField, CfField, Cchart);
		new_hamiltonian = get_hamiltonian(FgField, FmField, Farg, Fchart, CgField, CmField, Cchart, new_energy_partition);
	
		die_roll = u_rand_gen(globalRngState);
		del_hamiltonian = new_hamiltonian - old_hamiltonian;
		accept_probability = exp(old_hamiltonian - new_hamiltonian);
		
		does_accept = die_roll < accept_probability;
		// make sure that all the node make the same decision.
		MPI_Bcast((void *)&does_accept, 1, MPI_BYTE, 0, get_comm());
		
		if(i < Farg.num_forced_accept_step){
			qlat::Printf("End Trajectory %d:\t FORCE ACCEPT.\n", i + 1);
			FgField_ext = FgField;
			CgField_ext = CgField;
			does_accept = true;
		}else{
			if(does_accept){
				qlat::Printf("End Trajectory %d:\t ACCEPT.\n", i + 1);
				num_accept++;
				FgField_ext = FgField;
				CgField_ext = CgField;
			}else{
				qlat::Printf("End Trajectory %d:\t REJECT.\n", i + 1);
				num_reject++;
				FgField = FgField_ext;	
				CgField = CgField_ext;	
			}	
		}
		
		qlat::Printf("Old Hamiltonian =\t%+.12e\n", old_hamiltonian);
		qlat::Printf("New Hamiltonian =\t%+.12e\n", new_hamiltonian);
		qlat::Printf("Delta H         =\t%+.12e\n", del_hamiltonian); 
		qlat::Printf("exp(-Delta H)   =\t%12.6f\n", accept_probability);
		qlat::Printf("Die Roll        =\t%12.6f\n", die_roll); 	
	
		fetch_expanded_chart(FgField, Fchart);
		average_plaquette = avg_plaquette(FgField);
		qlat::Printf("Avg Plaquette   =\t%+.12e\n", average_plaquette); 
		qlat::Printf("ACCEPT RATE     =\t%+.4f\n", (double)num_accept / (num_accept + num_reject));	

		if(Farg.summary_dir_stem.size() > 0){
			Fprintf(p, "%i\t%.6e\t%.6e\t%.12e\t%i\t%.12e\n", 
					i + 1, abs(deltaH), accept_probability, average_plaquette, does_accept, 
					does_accept?new_energy_partition_new[1]:old_energy_partition_old[1]);
			Fflush(p);
		}

		if((i+1) % Farg.num_step_between_output == 0 && i+1 >= Farg.num_step_before_output){
			Arg_export arg_export;
			arg_export.beta = 			Farg.beta;
			arg_export.sequence_num = 	i+1;
			arg_export.ensemble_label = "multi";
			
			if(Farg.export_dir_stem.size() > 0){
				string address = Farg.export_dir_stem + "ckpoint." + show(i + 1);
				export_config_nersc(FgFieldExt, address, arg_export, true);
			}
			
			sync_node();
		}

	Timer::display();
}




QLAT_END_NAMESPACE
