// #include <gf/inverter.h>
// #include <gf/rng_state.h>

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

// #include<qmp.h>
// #include<mpi.h>

#define PARALLEL_READING_THREADS 1

using namespace std;
using namespace cps;
using namespace qlat;

void naive_field_expansion(const cps::Lattice &lat, 
			qlat::Field<cps::Matrix> &gauge_field_qlat, int mag)
{
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

void CPS2QLAT2File(const Coordinate &totalSize, int mag,
			string config_addr, string export_addr,
			int argc, char *argv[])
{
	
	// Supposed to:
	// 1. Set up (coarse) CPS to read in a configuration
	// 2. Transfer that to LQPS
	// 3. Export
		
	// Set up the (coarse) CPS 

	// static const char* fname = "CPS2LQPS2CPS(totalSize, config_addr, mag, &argc, &argv)";
	Start(&argc, &argv);
	int totalSite[] = {totalSize[0], totalSize[1], totalSize[2], totalSize[3]};
	DoArg do_arg_coarse;
	setDoArg(do_arg_coarse, totalSite);
	GJP.Initialize(do_arg_coarse);
	LRG.Initialize();

	load_config(config_addr.c_str());

	Lattice &lat = LatticeFactory::Create(F_CLASS_NONE, G_CLASS_NONE);

	// Set up LQPS

	begin(QMP_COMM_WORLD, Coordinate(SizeX(), SizeY(), SizeZ(), SizeT()));
	Coordinate totalSite_qlat(mag * GJP.NodeSites(0) * SizeX(), \
				mag * GJP.NodeSites(1) * SizeY(), \
				mag * GJP.NodeSites(2) * SizeZ(), \
				mag * GJP.NodeSites(3) * SizeT());
        Geometry geo_;
        geo_.init(totalSite_qlat, DIM);
	// expand geometry to test the communication.
	Coordinate expansion(2, 2, 2, 2);
	geo_.resize(expansion, expansion);

	qlat::Field<cps::Matrix> gauge_field_qlat;
	gauge_field_qlat.init(geo_);

#pragma omp parallel for
	for(long index = 0; index < geo_.localVolume(); index++){
		Coordinate x_qlat; geo_.coordinateFromIndex(x_qlat, index);
		qlat::Vector<cps::Matrix> vec_qlat(gauge_field_qlat.getElems(x_qlat));
		for(int mu = 0; mu < geo_.multiplicity; mu++){
		// only works for cps::Matrix
		vec_qlat[mu].UnitMatrix();
	}}

	// Transfer to LQPS

	// int x_qlat[4];

	syncNode();
	Coordinate coorNode; coorNodeFromIdNode(coorNode, getIdNode());
	std::cout << "cps UniqueID(): " << UniqueID() << "; "
		<< CoorX() << "x" << CoorY() << "x"<< CoorZ() << "x"<< CoorT() << "."
			 << "\tqlat: getIdNode(): " << getIdNode() << "; "
				<< show(coorNode) << "." << std::endl;

	naive_field_expansion(lat, gauge_field_qlat, mag);
 	LatticeFactory::Destroy();

	fetch_expanded(gauge_field_qlat);

	// cout << avg_plaquette(gauge_field_qlat) << endl;
	// cout << avg_real_trace(gauge_field_qlat) << endl;
	cout << check_constrained_plaquette(gauge_field_qlat, mag) << endl;	

	// HMC in qlat ------------------- start -------------------
	
	argCHmcWilson argHMC;
	argHMC.mag = mag;
	argHMC.length = 5;
	argHMC.beta = 2.13;
	argHMC.dt = 0.1;
	argHMC.gFieldExt = &gauge_field_qlat;
	
	algCHmcWilson algHMC(argHMC);
	algHMC.runTraj();

	// HMC in qlat -------------------- end --------------------

//	Field<MatrixTruncatedSU3> gauge_field_truncated;
//	fieldCastTruncated(gauge_field_truncated, gauge_field_qlat);
//	sophisticated_serial_write(gauge_field_truncated, export_addr, false, true);

 	// if(mag == 1) load_config(export_addr);
// 
// 	std::cout << "Start to read config." << std::endl;
// 
// 	qlat::Field<cps::Matrix> gauge_field_qlat_read; gauge_field_qlat_read.init(geo_);
// 	gauge_field_qlat_read = gauge_field_qlat;
// 	sophisticated_serial_read(gauge_field_qlat_read, export_addr, PARALLEL_READING_THREADS);
// 
// #pragma omp parallel for
// 	for(long index = 0; index < geo_.localVolume(); index++){
// 		Coordinate x_qlat; geo_.coordinateFromIndex(x_qlat, index);
// 		qlat::Vector<cps::Matrix> vec_qlat(gauge_field_qlat.getElems(x_qlat));
// 		qlat::Vector<cps::Matrix> vec_qlat_read(gauge_field_qlat_read.getElems(x_qlat));
// 
// 		for(int mu = 0; mu < geo_.multiplicity; mu++){
// 		// only works for cps::Matrix
// 		double sum = 0.;
// 		for(int i = 0; i < 9; i++){
// 		sum += std::norm(vec_qlat[mu][i] - vec_qlat_read[mu][i]);
// 		}
// 		assert(sum < 1e-5);
// 	}}
// 
// 	syncNode();
// 
// 	if(UniqueID() == 0) cout << "Matching Verified." << endl;
	
	End();

	// end();

}

void start_cps_qlat(const Coordinate &totalSize, double beta,
			int argc, char *argv[])
{
	int totalSize_cps[] = {totalSize[0], totalSize[1], totalSize[2], totalSize[3]};
	Start(&argc, &argv);
	DoArg do_arg;
	setDoArg(do_arg, totalSize_cps);
	do_arg.beta = beta;
	GJP.Initialize(do_arg);
	LRG.Initialize();

	begin(QMP_COMM_WORLD, Coordinate(SizeX(), SizeY(), SizeZ(), SizeT()));

}

void File2QLAT2CPS(const Coordinate &totalSize, int mag,
			string export_addr,
			int argc, char *argv[])
{
	// Suppose to read in the expanded file and transfer that to cps.
	//
	
	start_cps_qlat(mag * totalSize, 1.633, argc, argv);

	//	VRB.Level(VERBOSE_DEBUG_LEVEL);

	Lattice &lat = LatticeFactory::Create(F_CLASS_NONE, G_CLASS_WILSON);
        
	syncNode();
	cps::Matrix *ptr = lat.GaugeField();
#pragma omp parallel for
        for(long localIndex = 0; localIndex < GJP.VolNodeSites(); localIndex++){
		for(int mu = 0; mu < 4; mu++){
			(ptr + 4 * localIndex + mu)->UnitMatrix();
		}                
        }
        syncNode();


        Geometry geo_;
        geo_.init(totalSize, DIM);
	qlat::Field<MatrixTruncatedSU3> gf_qlat_trunc;
	gf_qlat_trunc.init(geo_);

	sophisticated_serial_read(gf_qlat_trunc, export_addr, \
						PARALLEL_READING_THREADS );

#pragma omp parallel for
	for(long local_index = 0; local_index < geo_.localVolume(); local_index++){
		Coordinate x_qlat; 
		geo_.coordinateFromIndex(x_qlat, local_index);
		int x_cps[] = {mag * x_qlat[0], mag * x_qlat[1], \
						mag * x_qlat[2], mag * x_qlat[3]};
		long localIndexCPS; GJP.localIndexFromPos(x_cps, localIndexCPS);
		cps::Matrix *ptr = lat.GaugeField(); 	
		for(int mu = 0; mu < DIM; mu++){
			memcpy((void *)(ptr + 4 * localIndexCPS + mu), \
				(void *)(gf_qlat_trunc.getElems(x_qlat).data() + mu), \
				sizeof(MatrixTruncatedSU3));
		}
	}
	
	lat.Reunitarize();

	double avgPlaq, avgConsPlaq;

	cout.precision(16);
	
	avgPlaq = lat.SumReTrPlaq() / (18. * GJP.VolSites());
	avgConsPlaq = check_constrained_plaquette(lat, mag);

	if(UniqueID() == 0){
		cout << "avgPlaq = " << avgPlaq << ",\t"
			<< "avgConsPlaq = " << avgConsPlaq << endl;
	}
		
	syncNode();
	if(UniqueID() == 0) std::cout << "Transfer to cps ready." << std::endl;
	syncNode();

	gchbArg gchbArg_;
	gchbArg_.mag = mag;
	gchbArg_.numIter = 50;
	gchbArg_.nHits = 10;
	gchbArg_.small = 0.2;

	CommonArg commonArg_;

	algGCtrnHeatBath algGCtrnHeatBath_(lat, &commonArg_, &gchbArg_);
	if(UniqueID() == 0) algGCtrnHeatBath_.showInfo();

	for(int i = 0; i < 20; i++){
		algGCtrnHeatBath_.run();
		avgPlaq = lat.SumReTrPlaq() / (18. * GJP.VolSites());
		avgConsPlaq = check_constrained_plaquette(lat, mag);

		if(UniqueID() == 0){
			cout << "Thermalization Cycle = " << i << ",\t"
				<< "avgPlaq = " << avgPlaq << ",\t"
			 	<< "avgConsPlaq = " << avgConsPlaq << endl;
			algGCtrnHeatBath_.accpetRate();
		}
	}

}



bool doesFileExist(const char *fn){
  struct stat sb;
  return 1 + stat(fn, &sb);
}

int main(int argc, char* argv[]){
	
	Coordinate totalSize(24, 24, 24, 64);
	int mag_factor = 2;
	string cps_config = "/bgusr/data09/qcddata/DWF/2+1f/24nt64/IWASAKI+DSDR/"
		"b1.633/ls24/M1.8/ms0.0850/ml0.00107/evol1/configurations/"
		"ckpoint_lat.300";
		// "/bgusr/home/ljin/qcdarchive/DWF_iwa_nf2p1/24c64/"
		// "2plus1_24nt64_IWASAKI_b2p13_ls16_M1p8_ms0p04_mu0p005_rhmc_H_R_G/"
		// "ckpoint_lat.IEEE64BIG.5000";	
	
	string expanded_config = "/bgusr/home/jtu/config/"
		"2+1f_24nt64_IWASAKI+DSDR_b1.633_ls24_M1.8_ms0.0850_ml0.00107/"
			"ckpoint_lat.300_mag" + show((long)mag_factor);
	
	// if(!doesFileExist(expanded_config.c_str())){
		CPS2QLAT2File(totalSize, mag_factor, cps_config, expanded_config, argc, argv);
		if(UniqueID() == 0) cout << "Program ended normally." << endl;
		return 0;
	// }

	File2QLAT2CPS(totalSize, mag_factor, cps_config, argc, argv);

	syncNode();
	
	if(UniqueID() == 0) cout << "Program ended normally." << endl;

	return 0;
}
