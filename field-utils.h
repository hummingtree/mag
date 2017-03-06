#include <vector>



class linearmodel2d
{
public:
    linearmodel2d(const vector<double> &x, const vector<double> &y){
		assert(x.size() == y.size());


	};
	
	void input(const vector<double> &x, const vector<double> &y)
	{
		assert(x.size() == y.size());
		
		mean(x, num, xavg);
		mean(y, num, yavg);
		
		double sumxsqr = 0.0, sumysqr = 0.0, sumxy = 0.0;
		
		for(int i = 0; i < num; i++){
		    sumxsqr += x[i] * x[i];
		    sumysqr += y[i] * y[i];
		    sumxy += x[i] * y[i];
		}
		
		sxx = sumxsqr - num * xavg * xavg;
		syy = sumysqr - num * yavg * yavg;
		sxy = sumxy - num * xavg * yavg;
		
		s = sqrt((syy - sxy * sxy / sxx) / (num - 2));
	}
	
    double ob();
    double oa(); // y = a + b * x
    double oberr();
    double oaerr();
    double orsqr();
private:
    double sxx, sxy, syy, xavg, yavg, s;
    int n;
};

linearmodel2d::linearmodel2d(double* x, double* y, int num){
    n = num;
    double sumx = 0.;
    double sumy = 0.;
    
    mean(x, num, xavg);
    mean(y, num, yavg);
    
    double sumxsqr = 0.0, sumysqr = 0.0, sumxy = 0.0;
    
    for(int i = 0; i < num; i++){
        if(!std::isnan(x[i]) && !std::isnan(y[i])){
            sumx += x[i];
            sumy += y[i];
            sumxsqr += x[i] * x[i];
            sumysqr += y[i] * y[i];
            sumxy += x[i] * y[i];
        }
    }
    
    xavg = sumx / num;
    yavg = sumy / num;
    
    sxx = sumxsqr - num * xavg * xavg;
    syy = sumysqr - num * yavg * yavg;
    sxy = sumxy - num * xavg * yavg;
    
    s = sqrt((syy - sxy * sxy / sxx) / (num - 2));
}

void linearmodel2d::input(double* x, double* y, int num){
    n = num;
    
    mean(x, num, xavg);
    mean(y, num, yavg);
    
    double sumxsqr = 0.0, sumysqr = 0.0, sumxy = 0.0;
    
    for(int i = 0; i < num; i++){
        sumxsqr += x[i] * x[i];
        sumysqr += y[i] * y[i];
        sumxy += x[i] * y[i];
    }
    
    sxx = sumxsqr - num * xavg * xavg;
    syy = sumysqr - num * yavg * yavg;
    sxy = sumxy - num * xavg * yavg;
    
    s = sqrt((syy - sxy * sxy / sxx) / (num - 2));
}

double linearmodel2d::ob(){
    return sxy / sxx;
}

double linearmodel2d::oa(){
    return yavg - xavg * sxy / sxx;
}

double linearmodel2d::oberr(){
    return s / sqrt(sxx);
}

double linearmodel2d::oaerr(){
    return s * sqrt(1.0 / n + xavg * xavg / sxx);
}

double linearmodel2d::orsqr(){
    return sxy * sxy / (sxx * syy);
}

