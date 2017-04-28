#include <hash-cpp/crc32.h>

#include <iostream>
#include <fstream>

#include <sys/stat.h>
#include <unistd.h>

#include <qlat/qlat.h>
#include "../field-hmc.h"
#include <string>

#define PARALLEL_READING_THREADS 16

using namespace std;
using namespace cps;
using namespace qlat;

void lattice_expansion(int mag, const Geometry& original_geo, const Geometry& expanded_geo,
						const string& original_config, const string& expanded_config){

	Field<Matrix> original_g_field;
	original_g_field.init(original_geo);
	
	import_config_nersc(original_g_field, original_config, PARALLEL_READING_THREADS);
	fetch_expanded(original_g_field);
	qlat::Printf("Coarse configuration: %s\n", original_config.c_str());
	qlat::Printf("Average plaquette original = %.12f\n", avg_plaquette(original_g_field));

	Field<Matrix> expanded_g_field;
	expanded_g_field.init(expanded_geo);

#pragma omp parallel for
	for(long index = 0; index < expanded_geo.local_volume(); index++){
		qlat::Coordinate x = expanded_geo.coordinate_from_index(index);
		for(int mu = 0; mu < expanded_geo.multiplicity; mu++){
			expanded_g_field.get_elems(x)[mu].UnitMatrix();
	}}
	sync_node();
#pragma omp parallel for
	for(long index = 0; index < original_geo.local_volume(); index++){
		qlat::Coordinate x = original_geo.coordinate_from_index(index);
		qlat::Coordinate expanded_x = mag * x;
		for(int mu = 0; mu < DIM; mu++){
			expanded_g_field.get_elems(expanded_x)[mu] = original_g_field.get_elems_const(x)[mu]; 
	}}	
	sync_node();
	qlat::Printf("Field expansion finished.\n");
	
	fetch_expanded(expanded_g_field);
	double avg_plaq = avg_plaquette(expanded_g_field);
	double c_plaq_before = check_constrained_plaquette(expanded_g_field, mag);
	qlat::Printf("Average plaquette     = %.12f\n", avg_plaq);
	qlat::Printf("Constrained plaquette = %.12f\n", c_plaq_before);	

	Arg_export ex;
	ex.beta = 0.;
	ex.sequence_num = 0;
	ex.ensemble_label = "lattice expansion";

	export_config_nersc(expanded_g_field, expanded_config, ex, true);
}

int main(int argc, char* argv[]){

	begin(&argc, &argv);
	
	int mag = 2;

	qlat::Coordinate expansion(2, 2, 2, 2);

	qlat::Coordinate original_size(12, 12, 12, 32);
	qlat::Coordinate expanded_size(24, 24, 24, 64);

	Geometry original_geo; original_geo.init(original_size, DIM);
	original_geo.resize(expansion, expansion);

	Geometry expanded_geo; expanded_geo.init(expanded_size, DIM);
	expanded_geo.resize(expansion, expansion);

	int start = 	130;
	int end = 		211;
	int interval = 	10; 

	string original_config;
	string expanded_config;

	for(int i = start; i < end; i += interval){
		original_config = str_printf("/bgusr/home/jtu/weird_ens/12x32ID/configurations/ckpoint_lat.%d", i);

		expanded_config = str_printf("./configurations/12x32ID_24x64/ckpoint_lat.%d", i);

		mkdir("./configurations", 0777);	
		mkdir("./configurations/12x32ID_24x64", 0777);	
			
		lattice_expansion(mag, original_geo, expanded_geo, original_config, expanded_config);
	}

	sync_node();
	printf("[%3d]:Program ENDED Normally.\n", get_id_node());

	return 0;
}

