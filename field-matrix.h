#pragma once

#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <cstring>

#include <cmath>

#include <util/lattice.h>
#include <util/vector.h>

#include <alg/alg_smear.h>

#include <qlat/config.h>
#include <qlat/coordinate.h>
#include <qlat/utils.h>
#include <qlat/mpi.h>
#include <qlat/field.h>
#include <qlat/field-comm.h>
#include <qlat/field-rng.h>
#include <qlat/field-io.h>

#include <timer.h>

#define Matrix cps::Matrix

#define CSTRING_MAX 500

using namespace cps;
using namespace qlat;
using namespace std;

// So in principle this file only uses the Matrix class and its member functions in cps.

inline Matrix random_su3_from_su2(double small, RngState &rng){
	
	double a0 = u_rand_gen(rng, 1., small);
	double a1 = u_rand_gen(rng, 1., -1.);
	double a2 = u_rand_gen(rng, 1., -1.);
	double a3 = u_rand_gen(rng, 1., -1.);

//	qlat::Printf("Proposed: (%.3f, %.3f, %.3f, %.3f)\n", a0, a1, a2, a3);
	double norm = sqrt((1 - a0 * a0) / (a1 * a1 + a2 * a2 + a3 * a3));

	// a0 /= norm; 
	a1 *= norm; a2 *= norm; a3 *= norm;

//	qlat::Printf("Proposed: (%.3f, %.3f, %.3f, %.3f)\n", a0, a1, a2, a3);

	Matrix m; m.UnitMatrix();

	switch(rand_gen(rng) % 3){
		case 0: m(1, 1) = qlat::Complex(a0, a3);  m(1, 2) = qlat::Complex(a2, a1);
				m(2, 1) = qlat::Complex(-a2, a1); m(2, 2) = qlat::Complex(a0, -a3); break;
		
		case 1: m(0, 0) = qlat::Complex(a0, a3);  m(0, 2) = qlat::Complex(a2, a1);
				m(2, 0) = qlat::Complex(-a2, a1); m(2, 2) = qlat::Complex(a0, -a3); break;
		
		case 2: m(0, 0) = qlat::Complex(a0, a3);  m(0, 1) = qlat::Complex(a2, a1);
				m(1, 0) = qlat::Complex(-a2, a1); m(1, 1) = qlat::Complex(a0, -a3); break;
	}
	
	return m;
}

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
// For the purpose of running updating algorithms on qcdserver, communication is 
// NOT needed if running on single node. Thus when crossing boundary to get the 
// link variables we need to regularize the coordinates. 
#ifdef USE_SINGLE_NODE
			regularize(y, field.geo.node_site);	
#endif
			mul = mul * field.get_elems_const(y)[direction];
			y[direction]++;
		}else{
			y[direction - DIM]--;
#ifdef USE_SINGLE_NODE
			regularize(y, field.geo.node_site);
#endif
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

// change the reading function such that it will check 
// 1. dimension
// 2. check sum, i.e. literally the sum ...
// 3. plaquette

inline bool snatch_keyword(char* line, const char* key, char* des){
    char* value;
	if(strstr(line, key) != NULL){
    // found the keyword and try to find the '='.
		value = strchr(line, '=');
		assert(value != NULL);

	// remove the '=' and any space and tab
        value++;
		while(value[0] == ' ' || value[0] == '\t') value++;
		if(value[strlen(value)-1] == '\n') value[strlen(value)-1] = '\0';
		strcpy(des, value);
        return true;
    }else{
        return false;
    }
}

inline void import_config_nersc(Field<Matrix> &field, const string import,
                        		const int num_of_reading_threads = 0){
	
	FILE *input = fopen(import.c_str(), "rb");
	assert(input != NULL);
	assert(!ferror(input));

	char line[CSTRING_MAX];
	char desc[CSTRING_MAX];

	Coordinate dim;
	uint32_t checksum = 0;
	double plaquette = 0.;
	char type[CSTRING_MAX];

	int pos = -1;
	rewind(input);
	while(fgets(line, CSTRING_MAX, input) != NULL){
		if(snatch_keyword(line, "DATATYPE", desc)){
			strcpy(type, desc);
			qlat::Printf("DATATYPE = %s\n", type);
		}
		if(snatch_keyword(line, "DIMENSION_1", desc)){
			dim[0] = atoi(desc);
			qlat::Printf("DIMENSION_1 = %d\n", dim[0]);
		}
		if(snatch_keyword(line, "DIMENSION_2", desc)){
			dim[1] = atoi(desc);
			qlat::Printf("DIMENSION_2 = %d\n", dim[1]);
		}
		if(snatch_keyword(line, "DIMENSION_3", desc)){
			dim[2] = atoi(desc);
			qlat::Printf("DIMENSION_3 = %d\n", dim[2]);
		}
		if(snatch_keyword(line, "DIMENSION_4", desc)){
			dim[3] = atoi(desc);
			qlat::Printf("DIMENSION_4 = %d\n", dim[3]);
		}

		if(snatch_keyword(line, "CHECKSUM", desc)){
			puts(desc);
			checksum = stol(desc, 0, 16);
			qlat::Printf("CHECKSUM = %x\n", checksum);
		}
		if(snatch_keyword(line, "PLAQUETTE", desc)){
			plaquette = strtod(desc, NULL);
			qlat::Printf("PLAQUETTE = %.8f\n", plaquette);
		}
		if(strstr(line, "END_HEADER") != NULL){ 
			pos = ftell(input); 
			break;
		}
	}

	assert(pos > -1);
	assert(!feof(input));

	bool does_skip_third;
	if(!strcmp(type, "4D_SU3_GAUGE")){
		does_skip_third = true;
	}else if(!strcmp(type, "4D_SU3_GAUGE_3x3")){
		does_skip_third = false;
	}else{
		qlat::Printf("WRONG DATATYPE!!!\n");
		assert(false);
	}

	if(dim == field.geo.total_site()){}
	else{
		qlat::Printf("WRONG Lattice Size!!!\n");
		assert(false);
	}

	if(does_skip_third){
		Geometry geo_ = field.geo;
		Field<MatrixTruncatedSU3> field_truncated;
		field_truncated.init(geo_);

		sophisticated_serial_read(field_truncated, import, pos, num_of_reading_threads);
		from_big_endian_64((char*)field_truncated.field.data(), 
								sizeof(MatrixTruncatedSU3) * field_truncated.field.size());

		uint32_t computed_checksum = fieldChecksumSum32(field_truncated);
		if(computed_checksum != checksum){
			qlat::Printf("WRONG Checksum: %x(labeled) vs %x(computed)\n", checksum, computed_checksum);
			assert(false);
		}

		for(long index = 0; index < geo_.local_volume(); index++){
			Coordinate x = geo_.coordinate_from_index(index);
			qlat::Vector<MatrixTruncatedSU3> p_from = field_truncated.get_elems(x);
			qlat::Vector<Matrix> p_to = field.get_elems(x);
			for(int mu = 0; mu < geo_.multiplicity; mu++){
				memcpy((char*)(p_to.data() + mu), (char*)(p_from.data() + mu), 
							sizeof(MatrixTruncatedSU3));
		}}
	
		reunitarize(field);

	}else{
		sophisticated_serial_read(field, import, pos, num_of_reading_threads);
		from_big_endian_64((char*)field.field.data(), sizeof(Matrix) * field.field.size());
	
		uint32_t computed_checksum = fieldChecksumSum32(field);
		if(computed_checksum != checksum){
			qlat::Printf("WRONG Checksum: %x(labeled) vs %x(computed)\n", checksum, computed_checksum);
			assert(false);
		}
	}

	fetch_expanded(field);
	double average_plaquette = avg_plaquette(field);
	qlat::Printf("Plaquette: %.8f(labeled) vs %.8f(computed)\n", plaquette, average_plaquette);
	if(abs(average_plaquette - plaquette) > 1e-4) assert(false);
}


