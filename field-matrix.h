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
#include <qlat/field-io.h>

#include <timer.h>

using namespace cps;
using namespace qlat;
using namespace std;

// So in principle this file only uses the Matrix class and its member functions in cps.

inline double norm(const Matrix &m){
	double sum = 0.;
	for(int i = 0; i < 9; i++){
		sum += norm(m[i]); // squared norm
	}
	return sqrt(sum);
}

inline double reunitarize(Field<Matrix> &field){
	double maxDev = 0.;
	Matrix oldElem;
        for(long index = 0; index < field.geo.local_volume(); index++){
                Coordinate x = field.geo.coordinate_from_index(index);
                for(int mu = 0; mu < field.geo.multiplicity; mu++){
			Matrix &newElem = field.get_elems(x)[mu];
			oldElem = newElem;
			newElem.Unitarize();
			maxDev = max(maxDev, norm(newElem - oldElem));
	}}
	return maxDev;
}

inline void get_path_ordered_product(Matrix &prod, const Field<Matrix> &field, 
					const Coordinate &x, const vector<int> &dir){
	Matrix mul; mul.UnitMatrix();
	Matrix dag;
	Coordinate y(x);
	int direction;
	for(unsigned int i = 0; i < dir.size(); i++){
		direction = dir[i];
		assert(direction < DIM * 2 && direction > -1);
		if(direction < DIM){
			mul = mul * field.get_elems_const(y)[direction];
			y[direction]++;
		}else{
			y[direction - DIM]--;
			dag.Dagger(field.get_elems_const(y)[direction - DIM]);
			mul = mul * dag;
		}
	}
	prod = mul;
}

inline double get_plaq(const Field<Matrix> &f, const Coordinate &x){
	// assuming properly communicated.
	
	const qlat::Vector<Matrix> gx = f.get_elems_const(x);
	vector<qlat::Vector<Matrix> > gxex(DIM);
	Coordinate y;
	for(int mu = 0; mu < DIM; mu++){
		y = x; y[mu]++;
		gxex[mu] = f.get_elems_const(y);
	}
	Matrix m, n;
	double ret = 0.;
	for(int mu = 0; mu < DIM; mu++){
	for(int nu = 0; nu < mu; nu++){
		m = gx[mu] * gxex[mu][nu];
		n.Dagger(gx[nu] * gxex[nu][mu]);
		ret += (m * n).ReTr();
	}}
	return ret;
}

inline int symmetric_index_mapping(int mu, int nu){
	return DIM * nu + mu - (nu + 1) * (nu + 2) / 2;
}

inline double get_rectangular(const Field<Matrix> &f, const Coordinate &x){
	// assuming properly communicated.

	TIMER("get_rectangular()");
	const qlat::Vector<Matrix> gx_0_0 = f.get_elems_const(x);
	vector<qlat::Vector<Matrix> > gx_1_0(DIM);
	vector<qlat::Vector<Matrix> > gx_2_0(DIM);
	vector<qlat::Vector<Matrix> > gx_1_1(6);

	Coordinate y;
	for(int mu = 0; mu < DIM; mu++){
		y = x; y[mu]++;
		gx_1_0[mu] = f.get_elems_const(y);
	}

	for(int mu = 0; mu < DIM; mu++){
		y = x; y[mu] += 2;
		gx_2_0[mu] = f.get_elems_const(y);
	}

	for(int mu = 0; mu < DIM; mu++){
	for(int nu = 0; nu < mu; nu++){
		y = x; y[mu]++; y[nu]++;
//		printf("gx_1_1 index = %d", symmetric_index_mapping(mu, nu));
		gx_1_1[symmetric_index_mapping(mu, nu)] = f.get_elems_const(y);
	}}
	
	//        mu
	//      0 1 2 3
	//    0 x 0 1 2
	//    1   x 3 4
	// nu 2     x 5
	//    3       x
	double sum = 0.;
	Matrix m;
	for(int mu = 0; mu < DIM; mu++){
	for(int nu = 0; nu < mu; nu++){
		int symmetric_index = symmetric_index_mapping(mu, nu);
		
		m.Dagger(gx_0_0[mu] * gx_1_0[mu][nu] * gx_1_1[symmetric_index][nu]);
		sum += (gx_0_0[nu] * gx_1_0[nu][nu] * gx_2_0[nu][mu] * m).ReTr();
	
		m.Dagger(gx_0_0[mu] * gx_1_0[mu][mu] * gx_2_0[mu][nu]);
		sum += (gx_0_0[nu] * gx_1_0[nu][mu] * gx_1_1[symmetric_index][mu] * m).ReTr();
	}}

	return sum;
}

inline double total_plaq(const qlat::Field<Matrix> &f){
	// assuming properly communicated.
	TIMER("total_plaq()");
	double local_sum = 0.;
	int num_threads;
	vector<double> pp_local_sum;

#pragma omp parallel
{
	if(omp_get_thread_num() == 0){
		num_threads = omp_get_num_threads();
		pp_local_sum.resize(num_threads);
	}
	double p_local_sum = 0.;
#pragma omp barrier
#pragma omp for
	for(long index = 0; index < f.geo.local_volume(); index++){
		Coordinate x = f.geo.coordinate_from_index(index);
			p_local_sum += get_plaq(f, x);
	}
		pp_local_sum[omp_get_thread_num()] = p_local_sum;
}

	for(int i = 0; i < num_threads; i++){
		local_sum += pp_local_sum[i];
	}

	double global_sum;
	MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, get_comm());

	return global_sum;
}

inline double total_rectangular(const qlat::Field<Matrix> &f){
	// assuming properly communicated.
	TIMER("total_rectangular()");
	double local_sum = 0.;
	int num_threads;
	vector<double> pp_local_sum;

#pragma omp parallel
{
	if(omp_get_thread_num() == 0){
		num_threads = omp_get_num_threads();
		pp_local_sum.resize(num_threads);
	}
	double p_local_sum = 0.;
#pragma omp barrier
#pragma omp for
	for(long index = 0; index < f.geo.local_volume(); index++){
		Coordinate x = f.geo.coordinate_from_index(index);
			p_local_sum += get_rectangular(f, x);
	}
		pp_local_sum[omp_get_thread_num()] = p_local_sum;
}

	for(int i = 0; i < num_threads; i++){
		local_sum += pp_local_sum[i];
	}

	double global_sum;
	MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, get_comm());

	return global_sum;
}

inline double avg_plaquette(const qlat::Field<Matrix> &f){
	return total_plaq(f) / (18. * get_num_node() * f.geo.local_volume());
}

inline double avg_real_trace(const qlat::Field<Matrix> &gauge_field_qlat){
	qlat::Geometry geo_ = gauge_field_qlat.geo;
	double tr_node_sum = 0.;
	for(long index = 0; index < geo_.local_volume(); index++){
		 Coordinate x_qlat = geo_.coordinate_from_index(index);
		 for(int mu = 0; mu < DIM; mu++){
		 	tr_node_sum += \
				(gauge_field_qlat.get_elems_const(x_qlat)[mu]).ReTr();
		 }
	}
	double tr_global_sum = 0.;
	MPI_Allreduce(&tr_node_sum, &tr_global_sum, 1, MPI_DOUBLE, MPI_SUM, get_comm());

	return tr_global_sum / (12. * get_num_node() * geo_.local_volume());
}

inline double check_constrained_plaquette(const qlat::Field<Matrix> &gField, int mag){
	std::vector<Coordinate> dir_vec(4);
	dir_vec[0] = Coordinate(1, 0, 0, 0);
	dir_vec[1] = Coordinate(0, 1, 0, 0);
	dir_vec[2] = Coordinate(0, 0, 1, 0);
	dir_vec[3] = Coordinate(0, 0, 0, 1);

	qlat::Geometry geo_ = gField.geo;
	
	long count = 0;
	double node_sum = 0.;
	for(int x0 = 0; x0 < geo_.node_site[0]; x0 += mag){
	for(int x1 = 0; x1 < geo_.node_site[1]; x1 += mag){
	for(int x2 = 0; x2 < geo_.node_site[2]; x2 += mag){
	for(int x3 = 0; x3 < geo_.node_site[3]; x3 += mag){
		Coordinate x(x0, x1, x2, x3);
		for(int mu = 0; mu < DIM; mu++){
		for(int nu = 0; nu < mu; nu++){
			Matrix m;
			vector<int> dir; dir.clear();
			for(int i = 0; i < mag; i++) dir.push_back(mu);
			for(int i = 0; i < mag; i++) dir.push_back(nu);
			for(int i = 0; i < mag; i++) dir.push_back(mu + DIM);
			for(int i = 0; i < mag; i++) dir.push_back(nu + DIM);
			
			get_path_ordered_product(m, gField, x, dir);
			
			count++;
			node_sum += m.ReTr();
		}}
	}}}}

	double global_sum;
	MPI_Allreduce(&node_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, get_comm());

	return global_sum / (3. * count * get_num_node());
}

class Arg_export{
public:
	double beta;
	int sequence_num;
	string ensemble_label;
};

inline void export_config_nersc(const Field<Matrix> &field, const string &dir,
					const Arg_export &arg, const bool doesSkipThird = false){
	FILE *pExport;

	Geometry geo_expand_one;
	geo_expand_one.init(field.geo.geon, field.geo.multiplicity, field.geo.node_site);
	Coordinate expansion(1, 1, 1, 1);
	geo_expand_one.resize(expansion, expansion);

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

	if(get_id_node() == 0){
		cout << "Node 0 open file!" << endl;
		pExport = fopen(dir.c_str(), "w");
		assert(pExport != NULL);

		std::ostringstream header_stream;

		header_stream << "BEGIN_HEADER" << endl;
		header_stream << "HDR_VERSION = 1.0" << endl;
		if(doesSkipThird) header_stream << "DATATYPE = 4D_SU3_GAUGE" << endl;
		else header_stream << "DATATYPE = 4D_SU3_GAUGE_3x3" << endl;
		header_stream << "DIMENSION_1 = " << field.geo.total_site()[0] << endl;
		header_stream << "DIMENSION_2 = " << field.geo.total_site()[1] << endl;
		header_stream << "DIMENSION_3 = " << field.geo.total_site()[2] << endl;
		header_stream << "DIMENSION_4 = " << field.geo.total_site()[3] << endl;
		header_stream << "CRC32HASH = " << crc32Hash << endl;
		header_stream << "CHECKSUM = " << checksumSum.str() << endl;
		
		header_stream.precision(12);
		header_stream << "LINK_TRACE = " << avgReTr << endl;
		header_stream << "PLAQUETTE = " << avgPlaq << endl;
		header_stream << "CREATOR = RBC" << endl;
		time_t now = std::time(NULL);	
		header_stream << "ARCHIVE_DATE = " << std::ctime(&now);
		header_stream << "ENSEMBLE_LABEL = " << arg.ensemble_label << endl;
		header_stream << "FLOATING_POINT = IEEE64BIG" << endl;
		header_stream << "ENSEMBLE_ID = NOT yet implemented" << endl;
		header_stream << "SEQUENCE_NUMBER = " << arg.sequence_num << endl;
		header_stream << "BETA = " << arg.beta << endl; 
		header_stream << "END_HEADER" << endl;

		fputs(header_stream.str().c_str(), pExport);
		fclose(pExport);
	}

	sophisticated_serial_write(field_write, dir, true);

	}else{
		Field<Matrix> field_write;
		sophisticated_make_to_order(field_write, field_cp);
		string crc32Hash = field_hash_crc32(field_write); 

		std::ostringstream checksumSum; 
		checksumSum.setf(std::ios::hex, std::ios::basefield);
		checksumSum << fieldChecksumSum32(field_write);

		if(get_id_node() == 0){
			cout << "Node 0 open file!" << endl;
			pExport = fopen(dir.c_str(), "w");
			assert(pExport != NULL);

			std::ostringstream header_stream;

			header_stream << "BEGIN_HEADER" << endl;
			header_stream << "HDR_VERSION = 1.0" << endl;
			if(doesSkipThird) header_stream << "DATATYPE = 4D_SU3_GAUGE" << endl;
			else header_stream << "DATATYPE = 4D_SU3_GAUGE_3x3" << endl;
			header_stream << "DIMENSION_1 = " << field.geo.total_site()[0] << endl;
			header_stream << "DIMENSION_2 = " << field.geo.total_site()[1] << endl;
			header_stream << "DIMENSION_3 = " << field.geo.total_site()[2] << endl;
			header_stream << "DIMENSION_4 = " << field.geo.total_site()[3] << endl;
			header_stream << "CRC32HASH = " << crc32Hash << endl;
			header_stream << "CHECKSUM = " << checksumSum.str() << endl;
			header_stream << "LINK_TRACE = " << avgReTr << endl;
			header_stream << "PLAQUETTE = " << avgPlaq << endl;
			header_stream << "CREATOR = RBC" << endl;
			time_t now = std::time(NULL);	
			header_stream << "ARCHIVE_DATE = " << std::ctime(&now);
			header_stream << "ENSEMBLE_LABEL = " << arg.ensemble_label << endl;
			header_stream << "FLOATING_POINT = IEEE64BIG" << endl;
			header_stream << "ENSEMBLE_ID = NOT yet implemented" << endl;
			header_stream << "SEQUENCE_NUMBER = " << arg.sequence_num << endl;
			header_stream << "BETA = " << arg.beta << endl; 
			header_stream << "END_HEADER" << endl;

			fputs(header_stream.str().c_str(), pExport);
			fclose(pExport);
		}

		sophisticated_serial_write(field_write, dir, true);
		

	}

}

inline void import_config_nersc(Field<Matrix> &field, const string importAddr,
                        		const int num_of_reading_threads = 0,
					const bool doesSkipThird = false){

	if(doesSkipThird){
		Geometry geo_ = field.geo;
		Field<MatrixTruncatedSU3> gf_qlat_trunc;
		gf_qlat_trunc.init(geo_);

		sophisticated_serial_read(gf_qlat_trunc, importAddr, 
						num_of_reading_threads);

		for(long index = 0; index < geo_.local_volume(); index++){
			Coordinate x = geo_.coordinate_from_index(index);
			for(int mu = 0; mu < DIM; mu++){
				memcpy((void *)(field.get_elems(x).data() + mu), 
				(void *)(gf_qlat_trunc.get_elems_const(x).data() + mu), 
				sizeof(MatrixTruncatedSU3));
			}
		}

		reunitarize(field);
	}else{
		sophisticated_serial_read(field, importAddr, num_of_reading_threads);
	}
}


