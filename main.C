// #include <gf/inverter.h>
// #include <gf/rng_state.h>

#include <qlat/config.h>
#include <qlat/utils.h>
#include <qlat/mpi.h>
#include <qlat/field.h>
#include <qlat/field-io.h>

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

// #include<qmp.h>
// #include<mpi.h>

using namespace std;
using namespace cps;
using namespace qlat;

extern MPI_Comm QMP_COMM_WORLD;

void load_config(string lat_file)
{
    const char *cname = "cps";
    const char *fname = "load_checkpoint()";

    Lattice &lat = LatticeFactory::Create(F_CLASS_NONE, G_CLASS_NONE);
    // sprintf(lat_file, "%s.%d", meas_arg.GaugeStem, traj);
    QioArg rd_arg(lat_file.c_str(), 0.001);
    rd_arg.ConcurIONumber = 1;
    ReadLatticeParallel rl;
    rl.read(lat, rd_arg);
    if(!rl.good()) ERR.General(cname, fname, ("Failed read lattice" + lat_file + "\n").c_str());
    LatticeFactory::Destroy();
}

void setDoArg(DoArg& do_arg, const int totalSite[4])
{
	do_arg.x_sites = totalSite[0];
	do_arg.y_sites = totalSite[1];
	do_arg.z_sites = totalSite[2];
	do_arg.t_sites = totalSite[3];
	do_arg.s_sites = 2;
	do_arg.dwf_height = 1.0;
	do_arg.x_bc = BND_CND_PRD;
	do_arg.y_bc = BND_CND_PRD;
	do_arg.z_bc = BND_CND_PRD;
	do_arg.t_bc = BND_CND_PRD;
	do_arg.start_conf_kind = START_CONF_ORD;
	do_arg.start_seed_kind = START_SEED_INPUT;
	do_arg.start_seed_value = 123121;
	do_arg.x_nodes = 0;
	do_arg.y_nodes = 0;
	do_arg.z_nodes = 0;
	do_arg.t_nodes = 0;
	do_arg.s_nodes = 0;
	do_arg.x_node_sites = 0;
	do_arg.y_node_sites = 0;
	do_arg.z_node_sites = 0;
	do_arg.t_node_sites = 0;
	do_arg.s_node_sites = 0;
	do_arg.gfix_chkb = 1;
}

void CPS2LQPS2File(const Coordinate &totalSize, int mag,
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
	Coordinate totalSite_qlat(mag * GJP.NodeSites(0) * SizeX(), mag * GJP.NodeSites(1) * SizeY(), mag * GJP.NodeSites(2) * SizeZ(), mag * GJP.NodeSites(3) * SizeT());
        Geometry geo_;
        geo_.init(totalSite_qlat, DIM);

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

	int x_cps[4];
	// int x_qlat[4];

	syncNode();
	Coordinate coorNode; coorNodeFromIdNode(coorNode, getIdNode());
	std::cout << "cps UniqueID(): " << UniqueID() << "; "
		<< CoorX() << "x" << CoorY() << "x"<< CoorZ() << "x"<< CoorT() << "x" << "."
			 << "qlat: getIdNode(): " << getIdNode() << "; "
				<< show(coorNode) << "." << std::endl;

	for(int i = 0; i < GJP.NodeSites(0); i++){ 
	for(int j = 0; j < GJP.NodeSites(1); j++){
	for(int k = 0; k < GJP.NodeSites(2); k++){
	for(int l = 0; l < GJP.NodeSites(3); l++){	

		x_cps[0] = i; x_cps[1] = j; x_cps[2] = k; x_cps[3] = l;
		Coordinate x_qlat(mag * i, mag * j, mag * k, mag * l);
		qlat::Vector<cps::Matrix> vec_qlat(gauge_field_qlat.getElems(x_qlat));
		vec_qlat[0] = *lat.GetLink(x_cps, 0);
		vec_qlat[1] = *lat.GetLink(x_cps, 1);
		vec_qlat[2] = *lat.GetLink(x_cps, 2);
		vec_qlat[3] = *lat.GetLink(x_cps, 3);

	}}}}

	syncNode();
	
	sophisticated_serial_write(gauge_field_qlat, export_addr, false, true);

	LatticeFactory::Destroy();

	if(mag == 1) load_config(export_addr);

	End();

	// end();

}

bool doesFileExist(const char *fn){
  struct stat sb;
  return 1 + stat(fn, &sb);
}

int main(int argc, char* argv[]){
	
	Coordinate totalSize(24, 24, 24, 64);
	int mag_factor = 4;
	string config = "/bgusr/home/ljin/qcdarchive/DWF_iwa_nf2p1/24c64/"
		"2plus1_24nt64_IWASAKI_b2p13_ls16_M1p8_ms0p04_mu0p005_rhmc_H_R_G/"
		"ckpoint_lat.IEEE64BIG.5000";	
	
	string output = "/bgusr/home/jtu/config/ckpoint_lat.IEEE64BIG.5000_mag" + show((long)mag_factor);
	
	// if(!doesFileExist(output.c_str()))
		CPS2LQPS2File(totalSize, mag_factor, config, output, argc, argv);

	cout << "Program ended normally." << endl;

	return 0;
}
