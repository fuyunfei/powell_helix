#include <vector>
#include "ceres/ceres.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
struct F1 {
	F1(double x, double y)
      : x_(x), y_(y) {}
  template <typename T> bool operator()(const T* const b,                              
                                        T* residual) const {
    // f1 = y-(b1*b5*cos(x/b3)+b7*x+b4)
    residual[0] = T(y_)-(b[0]*b[5]*cos(T(x_)/b[2])+b[6]*T(x_)+b[3]);
    return true;
  }
  private:
  const double x_;
  const double y_;
};
struct F2 {
	F2(double x, double y)
      : x_(x), y_(y) {}
  template <typename T> bool operator()(const T* const b,
                                        T* residual) const {
    // f2 = y-(b2*sin(x/b3)+b5)
    residual[0] = T(y_)-(b[1]*sin(T(x_)/b[2])+b[4]);
    return true;
  }
private:
  const double x_;
  const double y_;
};



struct F3 {
  template <typename T> bool operator()(const T* const b,
                                        T* residual) const {
    // f3 = (x2 - 2 x3)^2
    residual[0] = (T(1)-b[5]*b[5]-b[6]*b[6])*T(100);
    return true;
  }
};



struct F4 {
	F4(double x)
      : x_(x) {}
  template <typename T> bool operator()(const T* const b,
                                        T* residual) const {
    // f2 = y-(b2*sin(x/b3)+b5)
    residual[0] =b[1]*sin(T(x_)/b[2])/(-b[0]*b[5]*sin(T(x_)/b[2])+b[2]*b[6])-cos(T(x_));
    return true;
  }
private:
  const double x_;
};


DEFINE_string(minimizer, "trust_region",
              "Minimizer type to use, choices are: line_search & trust_region");

std::vector<double> linspace(double a, double b, int n) {
    std::vector<double> array;
    double step = (b-a) / (n-1);

    while(a <= b) {
        array.push_back(a);
        a += step;           // could recode to better handle rounding errors
    }
    return array;
}



std::vector<double> linspace_sin(double a, double b, int n) {
    std::vector<double> array;
    double step = (b-a) / (n-1);

    while(a <= b) {
        array.push_back(sin(a));
        a += step;           // could recode to better handle rounding errors
    }
    return array;
}




int main(int argc, char** argv) {
 double pi=3.1415926;
 int data_count=10000;
 std::vector<double>  x= linspace(0,3*pi,data_count);
 std::vector<double>  y= linspace_sin(0,3*pi,data_count);
 std::vector<double>  T= linspace(0,1,data_count);
 double b[]={0.5,0.5,0.5,0.5,0.5,0.0005,1};
  Problem problem;
  // Add residual terms to the problem using the using the autodiff
  // wrapper to get the derivatives automatically. The parameters, x1 through
  // x4, are modified in place.

for (int i = 0; i < data_count; ++i)
{
	// problem.AddResidualBlock(new AutoDiffCostFunction<F1, 1, 7>(new F1(T[i],x[i])),
 //                           NULL,
 //                           b);
 //  	problem.AddResidualBlock(new AutoDiffCostFunction<F2, 1, 7>(new F2(T[i],y[i])),
 //                           NULL,
 //                           b);

  	problem.AddResidualBlock(new AutoDiffCostFunction<F3, 1, 7>(new F3()),
                           NULL,
                           b);
  	problem.AddResidualBlock(new AutoDiffCostFunction<F4, 1, 7>(new F4(T[i])),
                           NULL,
                           b);
};


  
  Solver::Options options;

  options.max_num_iterations = 500;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  for (int i = 0; i < 7; ++i)
  {
  	std::cout << "Initial b"<<i <<" "<< b[i]<<std::endl;
  }
  
 
  // Run the solver!
  Solver::Summary summary;
  Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";
  for (int i = 0; i < 7; ++i)
  {
  	std::cout <<" "<< b[i] ;
  };

}
