#include <cmath>

using namespace std;

double autoCorrelation(double* data, int num);
double mean(double* data, int num, double& average);
double mean1(double* data, int num, double& average);
double correlator(double* data, int num, int m, double average);
double jackknife(double* data, int num, int binSize, double& average);
double jackknife_2_log_dividing(double* data1, double* data2, int num, int binSize, double& average);
double jackknife_2_power_dividing(double* data1, double* data2, int num, int binSize, double power1, double power2, double& average);
double jackknife_data_output(double* data, int num, int binSize, double& average, double* modified, int& modified_num);

double autoCorrelation(double* data, int num)
{
    double average = 0.0;
    double tauSum = 0.0;
    double std = 0.0;
   	
    std = mean(data, num, average);
    
    double C_0 = std * std;

	for(int n = 0; n < num / 10; n++){
		double C_n = correlator(data, num, n, average);
		if(C_n < 0)	break;
		tauSum += C_n / (2 * C_0);
	}

	for(int n = 1; n < num / 10; n++){
		double C_n = correlator(data, num, -n, average);
		if(C_n < 0)	break;
		tauSum += C_n / (2 * C_0);
	}
	
    return tauSum;
}

double correlator(double* data, int num, int m, double average){
	
	double sum = 0.0;
	double C_m = 0.0;
	
    if (m > 0) {
        for (int j = 0; j < num - m; j++) {
            sum += (data[j + m] - average) * (data[j] - average);
        }
        C_m = (sum / (num - m));
    }
    else {
        for (int j = - m; j < num; j++) {
            sum += (data[j + m] - average) * (data[j] - average);
        }
        C_m = (sum / (num + m));
    }
	
	return C_m;
}

double mean(double* data, int num, double& average)
{
    double sum = 0;
    double sqrSum = 0;
    double std = 0.0;
    
    for(int i = 0; i < num; i++)
    {
        sum += data[i];
        sqrSum += data[i] * data[i];
    }
    
    average = sum / num;
    std = sqrt((sqrSum - num * average * average) / num);
    
    return std;
}

double mean1(double* data, int num, double& average)
{
    double sum = 0;
    double sqrSum = 0;
    double std = 0.0;
    
    for(int i = 0; i < num; i++)
    {
        sum += data[i];
        sqrSum += data[i] * data[i];
    }
    
    average = sum / num;
    std = sqrt((sqrSum - num * average * average) / (num - 1));
    
    return std;
}

double jackknife(double* data, int num, int binSize, double& average){
    
    double sum1 = 0.;
    
    double vbin[num / binSize];
    double vjack[num / binSize];
    
	for(int i = 0; i < num; i++){
		sum1 += data[i];
		
		if((i + 1) % binSize == 0){
			vbin[(i + 1) / binSize - 1] = sum1 / binSize;			
			sum1 = 0.;
		}
	}
		
	double sum2 = 0.;
	for (int k = 0; k < num / binSize; k++) {
        sum2 += vbin[k];
    }
		
    for (int k = 0; k < num / binSize; k++) {
        vjack[k] = (sum2 - vbin[k]) / (num / binSize - 1);
    }
	
    double sum3 = 0.;
    double sqrsum = 0.;
	
    for (int k = 0; k < num / binSize; k++) {
        sum3 += vjack[k];
        sqrsum += vjack[k] * vjack[k];
    }
    
    average = sum3 / (num / binSize);
    return sqrt((sqrsum - (num / binSize) * average * average) * ((num / binSize) - 1) / (num / binSize));
}

double jackknife_data_output(double* data, int num, int binSize, double& average, double* modified, int& modified_num){
    double sum1 = 0.;
    
    double vbin[num / binSize];
    double vjack[num / binSize];
    
	for(int i = 0; i < num; i++){
		sum1 += data[i];
		
		if((i + 1) % binSize == 0){
			vbin[(i + 1) / binSize - 1] = sum1 / binSize;			
			sum1 = 0.;
		}
	}
		
	double sum2 = 0.;
	for (int k = 0; k < num / binSize; k++) {
        sum2 += vbin[k];
    }
		
    for (int k = 0; k < num / binSize; k++) {
        vjack[k] = (sum2 - vbin[k]) / (num / binSize - 1);
    	modified[k] = vjack[k];
	}
	
    double sum3 = 0.;
    double sqrsum = 0.;
	
    for (int k = 0; k < num / binSize; k++) {
        sum3 += vjack[k];
        sqrsum += vjack[k] * vjack[k];
    }
    
	modified_num = num / binSize;
    average = sum3 / (num / binSize);
    return sqrt((sqrsum - (num / binSize) * average * average) * ((num / binSize) - 1) / (num / binSize));
}

double jackknife_2_log_dividing(double* data1, double* data2, int num, int binSize, double& average){
    
    double sum1_1 = 0.;
	double sum1_2 = 0.;
    
    double vbin1[num / binSize];
    double vjack1[num / binSize];
	
    double vbin2[num / binSize];
    double vjack2[num / binSize];
    
	for(int i = 0; i < num; i++){
		sum1_1 += data1[i];
		
		if((i + 1) % binSize == 0){
			vbin1[(i + 1) / binSize - 1] = sum1_1 / binSize;			
			sum1_1 = 0.;
		}
	}
	
	for(int i = 0; i < num; i++){
		sum1_2 += data2[i];
		
		if((i + 1) % binSize == 0){
			vbin2[(i + 1) / binSize - 1] = sum1_2 / binSize;			
			sum1_2 = 0.;
		}
	}
		
	double sum2_1 = 0.;
	for (int k = 0; k < num / binSize; k++) {
        sum2_1 += vbin1[k];
    }
	
	double sum2_2 = 0.;
	for (int k = 0; k < num / binSize; k++) {
        sum2_2 += vbin2[k];
    }
		
    for (int k = 0; k < num / binSize; k++) {
        vjack1[k] = (sum2_1 - vbin1[k]) / (num / binSize - 1);
    }
    for (int k = 0; k < num / binSize; k++) {
        vjack2[k] = (sum2_2 - vbin2[k]) / (num / binSize - 1);
    }
	
    double sqrsum = 0.;
	
	double log_dividing_avg = log(sum2_1 / sum2_2);
	double dif;
	
    for (int k = 0; k < num / binSize; k++) {
		dif = log(vjack1[k] / vjack2[k]) - log_dividing_avg;
        sqrsum += dif * dif;
    }
    
    average = log_dividing_avg;
    return sqrt(sqrsum * (num / binSize - 1) / (num / binSize));
}

double jackknife_2_power_dividing(double* data1, double* data2, int num, int binSize, double power1, double power2, double& average){
    double sum1_1 = 0.;
	double sum1_2 = 0.;
    
    double vbin1[num / binSize];
    double vjack1[num / binSize];
	
    double vbin2[num / binSize];
    double vjack2[num / binSize];
    
	for(int i = 0; i < num; i++){
		sum1_1 += data1[i];
		
		if((i + 1) % binSize == 0){
			vbin1[(i + 1) / binSize - 1] = sum1_1 / binSize;			
			sum1_1 = 0.;
		}
	}
	
	for(int i = 0; i < num; i++){
		sum1_2 += data2[i];
		
		if((i + 1) % binSize == 0){
			vbin2[(i + 1) / binSize - 1] = sum1_2 / binSize;			
			sum1_2 = 0.;
		}
	}
		
	double sum2_1 = 0.;
	for (int k = 0; k < num / binSize; k++) {
        sum2_1 += vbin1[k];
    }
	
	double sum2_2 = 0.;
	for (int k = 0; k < num / binSize; k++) {
        sum2_2 += vbin2[k];
    }
		
    for (int k = 0; k < num / binSize; k++) {
        vjack1[k] = (sum2_1 - vbin1[k]) / (num / binSize - 1);
    }
    for (int k = 0; k < num / binSize; k++) {
        vjack2[k] = (sum2_2 - vbin2[k]) / (num / binSize - 1);
    }
	
    double sqrsum = 0.;
	
	double power_dividing_avg = pow(sum2_1 / (num / binSize), power1) / pow(sum2_2 / (num / binSize), power2);
	double dif;
	
    for (int k = 0; k < num / binSize; k++) {
		dif = pow(vjack1[k], power1) / pow(vjack2[k], power2) - power_dividing_avg;
        sqrsum += dif * dif;
    }
    
    average = power_dividing_avg;
    return sqrt(sqrsum * (num / binSize - 1) / (num / binSize));
}
