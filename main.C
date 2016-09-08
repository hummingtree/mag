#include <hash-cpp/crc32.h>

#include <iostream>
#include <fstream>

#include <sys/stat.h>
#include <unistd.h>

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

#include "alg_gchmc.h"
#include "alg_gchb.h"
#include "cps_util.h"
#include "stat.h"


// #include<qmp.h>
// #include<mpi.h>

#define PARALLEL_READING_THREADS 16

using namespace std;
using namespace cps;
using namespace qlat;

void naive_field_expansion(const cps::Lattice &lat, 
				qlat::Field<cps::Matrix> &gauge_field_qlat, 
				int mag)
{
	// from CPS to qlat
	syncNode();
#pragma omp parallel for
	for(long local_index = 0; local_index < GJP.VolNodeSites(); local_index++){
		int x_cps[4]; GJP.LocalIndex(local_index, x_cps);
		Coordinate x_qlat(mag * x_cps[0], mag * x_cps[1],
					mag * x_cps[2], mag * x_cps[3]);
                qlat::Vector<cps::Matrix> vec_qlat(gauge_field_qlat.getElems(x_qlat));
                vec_qlat[0] = *lat.GetLink(x_cps, 0);
                vec_qlat[1] = *lat.GetLink(x_cps, 1);
                vec_qlat[2] = *lat.GetLink(x_cps, 2);
                vec_qlat[3] = *lat.GetLink(x_cps, 3);	
	}	
	syncNode();
	if(UniqueID() == 0) std::cout << "Field expansion finished." << std::endl;
}

void hmc_in_qlat(const Coordinate &totalSize, 
                        string config_addr, const argCHmcWilson &argHMC,
			int argc, char *argv[]){
// 	Start(&argc, &argv);
// 	int totalSite[] = {totalSize[0], totalSize[1], totalSize[2], totalSize[3]};
// 	DoArg do_arg_coarse;
// 	setDoArg(do_arg_coarse, totalSite);
// 	GJP.Initialize(do_arg_coarse);
// 	LRG.Initialize();
// 
// 	// load config in CPS
// 	// load_config(config_addr.c_str());
// 
// 	// Lattice &lat = LatticeFactory::Create(F_CLASS_NONE, G_CLASS_NONE);
// 
// 	// Set up LQPS
// 
// 	begin(QMP_COMM_WORLD, Coordinate(SizeX(), SizeY(), SizeZ(), SizeT()));
// 	Coordinate totalSite_qlat(argHMC.mag * GJP.NodeSites(0) * SizeX(), 
// 				argHMC.mag * GJP.NodeSites(1) * SizeY(), 
// 				argHMC.mag * GJP.NodeSites(2) * SizeZ(), 
// 				argHMC.mag * GJP.NodeSites(3) * SizeT());
	begin(&argc, &argv);

	Geometry geoOrigin;
        geoOrigin.init(totalSize, DIM);
	Coordinate expansion(1, 1, 1, 1);
	geoOrigin.resize(expansion, expansion);

	Field<Matrix> gFieldOrigin;
	gFieldOrigin.init(geoOrigin);
	
	import_config_nersc(gFieldOrigin, config_addr, 16, true);
	fetch_expanded(gFieldOrigin);
 	report << "average plaquette origin = \t" 
		<< avg_plaquette(gFieldOrigin) << endl;

	Geometry geoExpanded;
	geoExpanded.init(argHMC.mag * totalSize, DIM);
	geoExpanded.resize(expansion, expansion);

	Field<Matrix> gFieldExpanded;
	gFieldExpanded.init(geoExpanded);

#pragma omp parallel for
	for(long index = 0; index < geoExpanded.localVolume(); index++){
		Coordinate x; geoExpanded.coordinateFromIndex(x, index);
		for(int mu = 0; mu < geoExpanded.multiplicity; mu++){
			gFieldExpanded.getElems(x)[mu].UnitMatrix();
	}}
	syncNode();
#pragma omp parallel for
	for(long index = 0; index < geoOrigin.localVolume(); index++){
		Coordinate x; geoOrigin.coordinateFromIndex(x, index);
		Coordinate xExpanded = argHMC.mag * x;
		for(int mu = 0; mu < DIM; mu++){
			gFieldExpanded.getElems(xExpanded)[mu] = 
						gFieldOrigin.getElemsConst(x)[mu]; 
	}}	
	syncNode();
	report << "Field expansion finished." << std::endl;
	
	Chart<Matrix> chart;
	produce_chart_envelope(chart, gFieldExpanded.geo, argHMC.gA);
	
	fetch_expanded(gFieldExpanded);
	report << "average plaquette = \t" << avg_plaquette(gFieldExpanded) << endl;
	report << "constrained plaquette = \t" 
		<< check_constrained_plaquette(gFieldExpanded, argHMC.mag) << endl;	

//  start hmc 
	 FILE *pFile = fopen("/bgusr/home/jtu/mag/data/alg_gchmc_test.dat", "a");

	if(getIdNode() == 0){
		time_t now = time(NULL);
		fputs("# ", pFile);
		fputs(ctime(&now), pFile);
		fputs(show(gFieldExpanded.geo).c_str(), pFile); fputs("\n", pFile);
		fprintf(pFile, "# mag =        %i\n", argHMC.mag);
		fprintf(pFile, "# trajLength = %i\n", argHMC.trajLength);
		fprintf(pFile, "# numTraj =    %i\n", argHMC.numTraj);
		fprintf(pFile, "# beta =       %.3f\n", argHMC.beta);
		fprintf(pFile, "# dt =         %.5f\n", argHMC.dt);
		fprintf(pFile, 
			"# traj. number\texp(-DeltaH)\tavgPlaq\taccept/reject\n");
		fflush(pFile);
		report << "pFile opened." << endl;
	}

	runHMC(gFieldExpanded, argHMC, pFile);
	
	fetch_expanded(gFieldExpanded);
	report << "average plaquette = \t" << avg_plaquette(gFieldExpanded) << endl;
	report << "constrained plaquette = \t" 
		<< check_constrained_plaquette(gFieldExpanded, argHMC.mag) << endl;	
}

bool doesFileExist(const char *fn){
  struct stat sb;
  return 1 + stat(fn, &sb);
}

int main(int argc, char* argv[]){
	cout.precision(12);
	cout.setf(ios::showpoint);
	cout.setf(ios::showpos);
	cout.setf(ios::scientific);

	Coordinate total_size(24, 24, 24, 64);
	int mag_factor = 2;
	
	int origin_start = 300;
	int origin_end = 700;
	int origin_interval = 20; 

	string cps_config;
	string expanded_config;

	argCHmcWilson argHMC;

	for(int i = origin_start; i < origin_end; i += origin_interval){
		argHMC.mag = mag_factor;
		argHMC.trajLength = 11;
		argHMC.numTraj = 200;
		argHMC.beta = 6.05;
		argHMC.dt = 1. / argHMC.trajLength;
		argHMC.outputInterval = 10;
		
		gAction gA_; gA_.type = qlat::WILSON;
		argHMC.gA = gA_;

		cps_config = "/bgusr/data09/qcddata/DWF/2+1f/24nt64/IWASAKI+DSDR/"
		"b1.633/ls24/M1.8/ms0.0850/ml0.00107/evol1/configurations/"
		"ckpoint_lat." + show((long)i);

		expanded_config = "/bgusr/home/jtu/config/"
		"2+1f_24nt64_IWASAKI+DSDR_b1.633_ls24_M1.8_ms0.0850_ml0.00107/"
		"ckpoint_lat." + show((long)i) + "_mag" + show((long)mag_factor) + 
		"_b6.05_WILSON/";

		mkdir(expanded_config.c_str(), 0777);	

		argHMC.exportAddress = expanded_config;
			
		hmc_in_qlat(total_size, cps_config, argHMC, argc, argv);
	}

	syncNode();
	cout << "Program ended normally." << endl;

	return 0;
}

