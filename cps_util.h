#pragma once

#include "alg_gchmc.h"

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
using namespace qlat;

double oneThird = 1. / 3., minusOneThird = -1. / 3.;

inline double avg_real_trace(const qlat::Field<cps::Matrix> &gauge_field_qlat);
inline double avg_plaquette(const qlat::Field<cps::Matrix> &gauge_field_qlat);

extern MPI_Comm QMP_COMM_WORLD;

double norm(const Matrix &m){
	double sum = 0.;
	for(int i = 0; i < 9; i++){
		sum += norm(m[i]); // squared norm
	}
	return sqrt(sum);
}

double reunitarize(Field<Matrix> &field){
	double maxDev = 0.;
	Matrix oldElem;
        for(long index = 0; index < field.geo.localVolume(); index++){
                Coordinate x; field.geo.coordinateFromIndex(x, index);
                for(int mu = 0; mu < field.geo.multiplicity; mu++){
			Matrix &newElem = field.getElems(x)[mu];
			oldElem = newElem;
			newElem.Unitarize();
			maxDev = max(maxDev, norm(newElem - oldElem));
	}}
	return maxDev;
}

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

class argExport{
public:
	double beta;
	int sequenceNum;
	string ensembleLabel;
};

void export_config_nersc(const Field<Matrix> &field, const string exportAddr,
			const argExport &arg,
			const bool doesSkipThird = false){
	FILE *pExport;

	Geometry geo_expand_one;
	geo_expand_one.init(field.geo.geon, field.geo.multiplicity,  
		field.geo.nodeSite, Coordinate(1, 1, 1, 1), Coordinate(1, 1, 1, 1));

	Field<Matrix> field_cp; field_cp.init(geo_expand_one);
	field_cp = field;

	fetch_expanded(field_cp);
	double avgPlaq = avg_plaquette(field_cp);
	double avgReTr = avg_real_trace(field_cp);

	if(doesSkipThird){
		Field<array<complex<double>, 6> > field_trunc, field_write;
		fieldCastTruncated(field_trunc, field_cp);
		sophisticated_make_to_order(field_write, field_trunc);
	string crc32Hash = field_hash_crc32(field_write); 
	
	std::ostringstream checksumSum; 
	checksumSum.setf(std::ios::hex, std::ios::basefield);
	checksumSum << fieldChecksumSum32(field_write);

	if(getIdNode() == 0){
		cout << "Node 0 open file!" << endl;
		pExport = fopen(exportAddr.c_str(), "w");
		assert(pExport != NULL);

		std::ostringstream header_stream;

		header_stream << "BEGIN_HEADER" << endl;
		header_stream << "HDR_VERSION = 1.0" << endl;
		if(doesSkipThird) header_stream << "DATATYPE = 4D_SU3_GAUGE" << endl;
		else header_stream << "DATATYPE = 4D_SU3_GAUGE_3x3" << endl;
		header_stream << "DIMENSION_1 = " << field.geo.totalSite(0) << endl;
		header_stream << "DIMENSION_2 = " << field.geo.totalSite(1) << endl;
		header_stream << "DIMENSION_3 = " << field.geo.totalSite(2) << endl;
		header_stream << "DIMENSION_4 = " << field.geo.totalSite(3) << endl;
		header_stream << "CRC32HASH = " << crc32Hash << endl;
		header_stream << "CHECKSUM = " << checksumSum.str() << endl;
		
		header_stream.precision(12);
		header_stream << "LINK_TRACE = " << avgReTr << endl;
		header_stream << "PLAQUETTE = " << avgPlaq << endl;
		header_stream << "CREATOR = RBC" << endl;
		time_t now = std::time(NULL);	
		header_stream << "ARCHIVE_DATE = " << std::ctime(&now);
		header_stream << "ENSEMBLE_LABEL = " << arg.ensembleLabel << endl;
		header_stream << "FLOATING_POINT = IEEE64BIG" << endl;
		header_stream << "ENSEMBLE_ID = NOT yet implemented" << endl;
		header_stream << "SEQUENCE_NUMBER = " << arg.sequenceNum << endl;
		header_stream << "BETA = " << arg.beta << endl; 
		header_stream << "END_HEADER" << endl;

		fputs(header_stream.str().c_str(), pExport);
		fclose(pExport);
	}

	sophisticated_serial_write(field_write, exportAddr, true);

	}else{
		Field<Matrix> field_write;
		sophisticated_make_to_order(field_write, field_cp);
		string crc32Hash = field_hash_crc32(field_write); 

		std::ostringstream checksumSum; 
		checksumSum.setf(std::ios::hex, std::ios::basefield);
		checksumSum << fieldChecksumSum32(field_write);

		if(getIdNode() == 0){
			cout << "Node 0 open file!" << endl;
			pExport = fopen(exportAddr.c_str(), "w");
			assert(pExport != NULL);

			std::ostringstream header_stream;

			header_stream << "BEGIN_HEADER" << endl;
			header_stream << "HDR_VERSION = 1.0" << endl;
			if(doesSkipThird) header_stream << "DATATYPE = 4D_SU3_GAUGE" << endl;
			else header_stream << "DATATYPE = 4D_SU3_GAUGE_3x3" << endl;
			header_stream << "DIMENSION_1 = " << field.geo.totalSite(0) << endl;
			header_stream << "DIMENSION_2 = " << field.geo.totalSite(1) << endl;
			header_stream << "DIMENSION_3 = " << field.geo.totalSite(2) << endl;
			header_stream << "DIMENSION_4 = " << field.geo.totalSite(3) << endl;
			header_stream << "CRC32HASH = " << crc32Hash << endl;
			header_stream << "CHECKSUM = " << checksumSum.str() << endl;
			header_stream << "LINK_TRACE = " << avgReTr << endl;
			header_stream << "PLAQUETTE = " << avgPlaq << endl;
			header_stream << "CREATOR = RBC" << endl;
			time_t now = std::time(NULL);	
			header_stream << "ARCHIVE_DATE = " << std::ctime(&now);
			header_stream << "ENSEMBLE_LABEL = " << arg.ensembleLabel << endl;
			header_stream << "FLOATING_POINT = IEEE64BIG" << endl;
			header_stream << "ENSEMBLE_ID = NOT yet implemented" << endl;
			header_stream << "SEQUENCE_NUMBER = " << arg.sequenceNum << endl;
			header_stream << "BETA = " << arg.beta << endl; 
			header_stream << "END_HEADER" << endl;

			fputs(header_stream.str().c_str(), pExport);
			fclose(pExport);
		}

		sophisticated_serial_write(field_write, exportAddr, true);
		

	}

}

void import_config_nersc(Field<Matrix> &field, const string importAddr,
                        const int num_of_reading_threads = 0,
			const bool doesSkipThird = false){

	if(doesSkipThird){
		Geometry geo_;
		geo_.init(field.geo);
		Field<MatrixTruncatedSU3> gf_qlat_trunc;
		gf_qlat_trunc.init(geo_);

		sophisticated_serial_read(gf_qlat_trunc, importAddr, 
						num_of_reading_threads);

		for(long index = 0; index < geo_.localVolume(); index++){
			Coordinate x; geo_.coordinateFromIndex(x, index);
			for(int mu = 0; mu < DIM; mu++){
				memcpy((void *)(field.getElems(x).data() + mu), 
				(void *)(gf_qlat_trunc.getElemsConst(x).data() + mu), 
				sizeof(MatrixTruncatedSU3));
			}
		}

		reunitarize(field);
	}else{
		sophisticated_serial_read(field, importAddr, num_of_reading_threads);
	}
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

template<class CharT, class Traits>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
                   const cps::Matrix &M){
        std::basic_ostringstream<CharT, Traits> os_;
        os_.flags(os.flags());
        os_.imbue(os.getloc());
        os_.precision(4);
        os_.setf(std::ios::showpoint);
        os_.setf(std::ios::showpos);
        os_.setf(std::ios::scientific);
        os_ << "|" << M(0, 0) << " " << M(0, 1) << " " << M(0, 2) << "|" << endl;
        os_ << "|" << M(1, 0) << " " << M(1, 1) << " " << M(1, 2) << "|" << endl;
        os_ << "|" << M(2, 0) << " " << M(2, 1) << " " << M(2, 2) << "|" << endl;
        return os << os_.str();
}

