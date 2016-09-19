#pragma once

#include <iostream>

#include <util/lattice.h>
#include <util/vector.h>

#include <qlat/config.h>
#include <qlat/coordinate.h>
#include <qlat/utils.h>
#include <qlat/mpi.h>
#include <qlat/field.h>
#include <qlat/field-comm.h>
#include <qlat/field-rng.h>

using namespace cps;
using namespace qlat;
using namespace std;

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

inline void getPathOrderedProd(Matrix &prod, const Field<Matrix> &field, 
					const Coordinate &x, const vector<int> &dir){
	Matrix mul; mul.UnitMatrix();
	Matrix dag;
	Coordinate y(x);
	int direction;
	for(unsigned int i = 0; i < dir.size(); i++){
		direction = dir[i];
		assert(direction < DIM * 2 && direction > -1);
		if(direction < DIM){
			mul = mul * field.getElemsConst(y)[direction];
			y[direction]++;
		}else{
			y[direction - DIM]--;
			dag.Dagger(field.getElemsConst(y)[direction - DIM]);
			mul = mul * dag;
		}
	}
	prod = mul;
}

inline double avg_plaquette(const qlat::Field<Matrix> &gauge_field_qlat){
	std::vector<Coordinate> dir_vec(4);
	dir_vec[0] = Coordinate(1, 0, 0, 0);
	dir_vec[1] = Coordinate(0, 1, 0, 0);
	dir_vec[2] = Coordinate(0, 0, 1, 0);
	dir_vec[3] = Coordinate(0, 0, 0, 1);

	qlat::Geometry geo_ = gauge_field_qlat.geo;

	double node_sum = 0.;
	
	for(long index = 0; index < geo_.localVolume(); index++){
		 Coordinate x_qlat; geo_.coordinateFromIndex(x_qlat, index);
		 for(int mu = 0; mu < DIM; mu++){
		 for(int nu = 0; nu < mu; nu++){	
		 	Matrix mul; mul.UnitMatrix();
			mul = mul * gauge_field_qlat.getElemsConst(x_qlat)[mu];
			x_qlat = x_qlat + dir_vec[mu];
			mul = mul * gauge_field_qlat.getElemsConst(x_qlat)[nu];
			x_qlat = x_qlat + dir_vec[nu] - dir_vec[mu];
			Matrix dag1; 
			dag1.Dagger(gauge_field_qlat.getElemsConst(x_qlat)[mu]);
			mul = mul * dag1;
			x_qlat = x_qlat - dir_vec[nu];
			Matrix dag2;
			dag2.Dagger(gauge_field_qlat.getElemsConst(x_qlat)[nu]);
			mul = mul * dag2;

			node_sum += mul.ReTr();
		 }}
	}
	double global_sum;
	MPI_Allreduce(&node_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, getComm());

	return global_sum / (18. * getNumNode() * geo_.localVolume());
}

inline double totalPlaq(const qlat::Field<Matrix> &gauge_field_qlat){
	std::vector<Coordinate> dir_vec(4);
	dir_vec[0] = Coordinate(1, 0, 0, 0);
	dir_vec[1] = Coordinate(0, 1, 0, 0);
	dir_vec[2] = Coordinate(0, 0, 1, 0);
	dir_vec[3] = Coordinate(0, 0, 0, 1);

	qlat::Geometry geo_ = gauge_field_qlat.geo;

	double node_sum = 0.;
	
	for(long index = 0; index < geo_.localVolume(); index++){
		 Coordinate x_qlat; geo_.coordinateFromIndex(x_qlat, index);
		 for(int mu = 0; mu < DIM; mu++){
		 for(int nu = 0; nu < mu; nu++){	
		 	Matrix mul; mul.UnitMatrix();
			mul = mul * gauge_field_qlat.getElemsConst(x_qlat)[mu];
			x_qlat = x_qlat + dir_vec[mu];
			mul = mul * gauge_field_qlat.getElemsConst(x_qlat)[nu];
			x_qlat = x_qlat + dir_vec[nu] - dir_vec[mu];
			Matrix dag1; 
			dag1.Dagger(gauge_field_qlat.getElemsConst(x_qlat)[mu]);
			mul = mul * dag1;
			x_qlat = x_qlat - dir_vec[nu];
			Matrix dag2;
			dag2.Dagger(gauge_field_qlat.getElemsConst(x_qlat)[nu]);
			mul = mul * dag2;

			node_sum += mul.ReTr();
		 }}
	}
	double global_sum;
	MPI_Allreduce(&node_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, getComm());

	return global_sum;
}

inline double avg_real_trace(const qlat::Field<Matrix> &gauge_field_qlat){
	qlat::Geometry geo_ = gauge_field_qlat.geo;
	double tr_node_sum = 0.;
	for(long index = 0; index < geo_.localVolume(); index++){
		 Coordinate x_qlat; geo_.coordinateFromIndex(x_qlat, index);
		 for(int mu = 0; mu < DIM; mu++){
		 	tr_node_sum += \
				(gauge_field_qlat.getElemsConst(x_qlat)[mu]).ReTr();
		 }
	}
	double tr_global_sum = 0.;
	MPI_Allreduce(&tr_node_sum, &tr_global_sum, 1, MPI_DOUBLE, MPI_SUM, getComm());

	return tr_global_sum / (12. * getNumNode() * geo_.localVolume());
}

inline double check_constrained_plaquette(
				const qlat::Field<Matrix> &gauge_field_qlat,
				int mag){
	std::vector<Coordinate> dir_vec(4);
	dir_vec[0] = Coordinate(1, 0, 0, 0);
	dir_vec[1] = Coordinate(0, 1, 0, 0);
	dir_vec[2] = Coordinate(0, 0, 1, 0);
	dir_vec[3] = Coordinate(0, 0, 0, 1);

	qlat::Geometry geo_ = gauge_field_qlat.geo;
	
	long count = 0;
	double node_sum = 0.;
	for(int x0 = 0; x0 < geo_.nodeSite[0]; x0 += mag){
	for(int x1 = 0; x1 < geo_.nodeSite[1]; x1 += mag){
	for(int x2 = 0; x2 < geo_.nodeSite[2]; x2 += mag){
	for(int x3 = 0; x3 < geo_.nodeSite[3]; x3 += mag){
		Coordinate x(x0, x1, x2, x3);
		for(int mu = 0; mu < DIM; mu++){
		for(int nu = 0; nu < mu; nu++){
			Matrix m;
			vector<int> dir; dir.clear();
			for(int i = 0; i < mag; i++) dir.push_back(mu);
			for(int i = 0; i < mag; i++) dir.push_back(nu);
			for(int i = 0; i < mag; i++) dir.push_back(mu + DIM);
			for(int i = 0; i < mag; i++) dir.push_back(nu + DIM);
			
			getPathOrderedProd(m, gauge_field_qlat, x, dir);
			
			count++;
			node_sum += m.ReTr();
		}}
	}}}}

	double global_sum;
	MPI_Allreduce(&node_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, getComm());

	return global_sum / (3. * count * getNumNode());
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


