#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {

  //Calculating the RMSE
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  for(unsigned int i = 0; i < estimations.size(); i++)
  {
      VectorXd residual = estimations[i] - ground_truth[i];

      residual = residual.array()*residual.array();
      rmse += residual;
  }

  rmse = rmse/estimations.size();
  rmse = rmse.array().sqrt();

  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd H_j(3,4);

  //Calculating the Jacobian - Careful not to divide with 0
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  float c1 = pow(px, 2)+pow(py, 2);

  if (fabs(c1) < 0.00001)
  {
     c1 = 0.00001;
  }
  float c2 = sqrt(c1);
  float c3 = (c1*c2);

  H_j << (px/c2), (py/c2), 0, 0,
         -(py/c1), (px/c1), 0, 0,
         py*(vx*py - vy*px)/c3, px*(px*vy-py*vx)/c3, px/c2, py/c2;
  return H_j;
}
