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

#include <iostream>

#include <qlat/coordinate.h>

using namespace std;
using namespace cps;

double oneThird = 1. / 3., minusOneThird = -1. / 3.;

extern MPI_Comm QMP_COMM_WORLD;

void load_config(string lat_file)
{
    const char *cname = "cps";
    const char *fname = "load_config";

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

double check_constrained_plaquette(cps::Lattice &lat, int mag){
        long count = 0;
        double node_sum = 0.;
        for(int x0 = 0; x0 < GJP.XnodeSites(); x0 += mag){
        for(int x1 = 0; x1 < GJP.YnodeSites(); x1 += mag){
        for(int x2 = 0; x2 < GJP.ZnodeSites(); x2 += mag){
        for(int x3 = 0; x3 < GJP.TnodeSites(); x3 += mag){
                int x[] = {x0, x1, x2, x3};
                for(int mu = 0; mu < 4; mu++){
                for(int nu = 0; nu < mu; nu++){
			cps::Matrix mul; mul.UnitMatrix();
                        for(int i = 0; i < mag; i++){
                                mul = mul * *lat.GetLink(x, mu);
                                x[mu]++;
                        }
                        for(int i = 0; i < mag; i++){
                                mul = mul * *lat.GetLink(x, nu);
                                x[nu]++;
                        }
                        cps::Matrix dag;
                        for(int i = 0; i < mag; i++){
                                x[mu]--;
                                dag.Dagger(*lat.GetLink(x, mu));
                                mul = mul * dag;
                        }
                        for(int i = 0; i < mag; i++){
                                x[nu]--;
                                dag.Dagger(*lat.GetLink(x, nu));
                                mul = mul * dag;
                        }
                        count++;
                        node_sum += mul.ReTr();	
		
		// 	for(int i = 0; i < mag; i++) dir.push_back(mu);
		// 	for(int i = 0; i < mag; i++) dir.push_back(nu);
		// 	for(int i = 0; i < mag; i++) dir.push_back(mu + 4);
		// 	for(int i = 0; i < mag; i++) dir.push_back(nu + 4);
                // 	cps::Matrix add; add.ZeroMatrix();
		// 	lat.PathOrdProd(add, x, dir.data(), 4 * mag);
		// 	node_sum += add.ReTr();
		// 	count++;
                }}
        }}}}

        double global_sum;
        MPI_Allreduce(&node_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, QMP_COMM_WORLD);

        return global_sum / (3. * count * NumNodes());
}


