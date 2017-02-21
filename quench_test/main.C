#include <hash-cpp/crc32.h>

#include <iostream>
#include <fstream>

#include <sys/stat.h>
#include <unistd.h>

#include <qlat/qlat.h>
#include "../field-hmc.h"

#define PARALLEL_READING_THREADS 16

using namespace std;
using namespace cps;
using namespace qlat;

void naive_field_expansion(const cps::Lattice &lat, 
				qlat::Field<Matrix> &gauge_field_qlat, 
				int mag)
{
	// from CPS to qlat
	sync_node();
#pragma omp parallel for
	for(long local_index = 0; local_index < GJP.VolNodeSites(); local_index++){
		int x_cps[4]; GJP.LocalIndex(local_index, x_cps);
		Coordinate x_qlat(mag * x_cps[0], mag * x_cps[1],
					mag * x_cps[2], mag * x_cps[3]);
                qlat::Vector<Matrix> vec_qlat(gauge_field_qlat.get_elems(x_qlat));
                vec_qlat[0] = *lat.GetLink(x_cps, 0);
                vec_qlat[1] = *lat.GetLink(x_cps, 1);
                vec_qlat[2] = *lat.GetLink(x_cps, 2);
                vec_qlat[3] = *lat.GetLink(x_cps, 3);	
	}	
	sync_node();
	if(UniqueID() == 0) std::cout << "Field expansion finished." << std::endl;
}

void hmc_in_qlat(const Coordinate &totalSize, string config_addr, const Arg_chmc &arg,
					const Arg_chmc &arg_original, int argc, char *argv[]){
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
// 	Coordinate totalSite_qlat(arg.mag * GJP.NodeSites(0) * SizeX(), 
// 				arg.mag * GJP.NodeSites(1) * SizeY(), 
// 				arg.mag * GJP.NodeSites(2) * SizeZ(), 
// 				arg.mag * GJP.NodeSites(3) * SizeT());
	begin(&argc, &argv);

	Geometry geoOrigin; geoOrigin.init(totalSize, DIM);
	Coordinate expansion(2, 2, 2, 2);
	geoOrigin.resize(expansion, expansion);

	Field<Matrix> gFieldOrigin;
	gFieldOrigin.init(geoOrigin);
	
	import_config_nersc(gFieldOrigin, config_addr, 16);
	fetch_expanded(gFieldOrigin);
	qlat::Printf("COARSE Configuration: %s\n", config_addr.c_str());
	qlat::Printf("AVERAGE Plaquette Original = %.8f\n", avg_plaquette(gFieldOrigin));

	Geometry geoExpanded;
	geoExpanded.init(arg.mag * totalSize, DIM);
	geoExpanded.resize(expansion, expansion);

	Field<Matrix> gFieldExpanded;
	gFieldExpanded.init(geoExpanded);

	Field<double> dField; dField.init(geoOrigin, DIM * SU3_NUM_OF_GENERATORS);

//	Arg_chmc arg_original;
//	arg_original.mag = 1;
//	report << "COARSE DERIVATIVE = \t" << derivative_sum(gFieldOrigin, arg_original) << endl;
	derivative_field(dField, gFieldOrigin, arg_original);

	Field<double> dField_output; dField_output.init(geoOrigin, DIM * SU3_NUM_OF_GENERATORS);
	sophisticated_make_to_order(dField_output, dField);
	sophisticated_serial_write(dField_output, arg.summary_dir_stem + "dev_dump_coarse");

#pragma omp parallel for
	for(long index = 0; index < geoExpanded.local_volume(); index++){
		Coordinate x = geoExpanded.coordinate_from_index(index);
		for(int mu = 0; mu < geoExpanded.multiplicity; mu++){
			gFieldExpanded.get_elems(x)[mu].UnitMatrix();
	}}
	sync_node();
#pragma omp parallel for
	for(long index = 0; index < geoOrigin.local_volume(); index++){
		Coordinate x = geoOrigin.coordinate_from_index(index);
		Coordinate xExpanded = arg.mag * x;
		for(int mu = 0; mu < DIM; mu++){
			gFieldExpanded.get_elems(xExpanded)[mu] = 
						gFieldOrigin.get_elems_const(x)[mu]; 
	}}	
	sync_node();
	qlat::Printf("Field Expansion Finished.\n");
	
	Chart<Matrix> chart;
	produce_chart_envelope(chart, gFieldExpanded.geo, arg.gauge);

	fetch_expanded(gFieldExpanded);
	double avg_plaq = avg_plaquette(gFieldExpanded);
	double c_plaq_before = check_constrained_plaquette(gFieldExpanded, arg.mag);
	qlat::Printf("AVERAGE Plaquette     = %.12f\n", avg_plaq);
	qlat::Printf("CONSTRAINED Plaquette = %.12f\n", c_plaq_before);	

//  start hmc 
	FILE *pFile = Fopen(
			str_printf("./summary/8c8_iwasaki_b8.8188_c1_0.07151_16c16_iwasaki_b3.40_c1_-0.331.dat").c_str(), "a");
	
	time_t now = time(NULL);
	Fprintf(pFile, "# %s", ctime(&now));
	Fprintf(pFile, "# %s\n", show(gFieldExpanded.geo).c_str());
	Fprintf(pFile, "# Coarse Config     = %s\n", config_addr.c_str());
	Fprintf(pFile, "# mag               = %i\n", arg.mag);
	Fprintf(pFile, "# trajectory_length = %i\n", arg.trajectory_length);
	Fprintf(pFile, "# num_trajectory    = %i\n", arg.num_trajectory);
	Fprintf(pFile, "# beta              = %.6f\n", arg.beta);
	Fprintf(pFile, "# dt                = %.5f\n", arg.dt);
	Fprintf(pFile, "# c1                = %.5f\n", arg.gauge.c1);
	Fprintf(pFile, "# GAUGE_TYPE        = %d\n", arg.gauge.type);
	Fprintf(pFile, "# traj. number\texp(-DeltaH)\tavgPlaq\taccept/reject\n");
	Fflush(pFile);
	qlat::Printf("pFile opened.");

	run_chmc(gFieldExpanded, arg, pFile);
	
	avg_plaq = avg_plaquette(gFieldExpanded);
	double c_plaq_after = check_constrained_plaquette(gFieldExpanded, arg.mag);
	fetch_expanded(gFieldExpanded);
	qlat::Printf("AVERAGE Plaquette     = %.12f\n", avg_plaq);
	qlat::Printf("CONSTRAINED Plaquette = %.12f\n", c_plaq_after);	
	qlat::Printf("CONSTRAINED Diff      = %.12f\n", c_plaq_after - c_plaq_before);
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

	Coordinate total_size(8, 8, 8, 8);
	int mag = 2;
	
	int origin_start = 		1000;
	int origin_end = 		1000;
	int origin_interval = 	100; 

	string cps_config;
	string expanded_config;

	Arg_chmc arg;
	Arg_chmc arg_original;

	for(int i = origin_start; i <= origin_end; i += origin_interval){
		arg.mag = mag;
		arg.trajectory_length = 9;
		arg.num_trajectory = 10000;
		arg.beta = 3.40;
		arg.dt = 1. / arg.trajectory_length;
		arg.num_step_between_output = 10;
		arg.num_forced_accept_step = 20;
		arg.num_step_before_output = 100;

		Gauge gauge; gauge.type = qlat::IWASAKI;
		gauge.c1 = -0.331;
		arg.gauge = gauge;

		arg_original.mag = 1;
//		arg_original.trajectory_length = 7;
//		arg_original.num_trajectory = 10000;
		arg_original.beta = 8.8188;
//		arg_original.dt = 1. / arg.trajectory_length;
//		arg_original.num_step_between_output = 10;
//		arg_original.num_forced_accept_step = 20;
//		arg_original.num_step_before_output = 100;

		arg_original.gauge.type = qlat::IWASAKI;
		arg_original.gauge.c1 = 0.07151;

//		cps_config = 			str_printf("/bgusr/home/jtu/quench_evolution/configurations/quench_iwasaki_4c4_b2.60_c1_-0.331/ckpoint.%d", i);
		cps_config = 			str_printf("/bgusr/home/jtu/quench_evolution/configurations/quench_iwasaki_8c8_b8.8188_c1_0.07151/ckpoint.%d", i);

// 		expanded_config = 		str_printf("./configurations/quench_wilson_4c4_b6.05_ckpoint.%d_mag%d_b%.2f_wilson/", i, mag, arg.beta);
		expanded_config = "";
		arg.summary_dir_stem = 	str_printf("./results/8c8_iwasaki_b8.8188_c1_0.07151_16c16_iwasaki_b3.40_c1_-0.331_config_%d/", i);
//		arg.summary_dir_stem = "";

		mkdir(expanded_config.c_str(), 0777);	
		mkdir(arg.summary_dir_stem.c_str(), 0777);	

		arg.export_dir_stem = expanded_config;
			
		hmc_in_qlat(total_size, cps_config, arg, arg_original, argc, argv);
	}

	sync_node();
	printf("[%3d]:Program ENDED Normally.\n", get_id_node());

	return 0;
}

