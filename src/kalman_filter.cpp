#include "kalman_filter.h"

#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  //Predict the following state
  //using the covariance matrix, noise matrix and state transition matrix
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  //Use the Kalman-filter update - laser
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd H_t = H_.transpose();
  MatrixXd S = H_ * P_ * H_t + R_;
  MatrixXd S_i = S.inverse();
  MatrixXd PH_t = P_ * H_t;
  MatrixXd K = PH_t * S_i;

  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  //Use the extended Kalman-filter update - radar
  float help;
  VectorXd z_pred = VectorXd(3);
  //Careful not to divide with 0 and setting the value
  // to 0.00001 if it would be the case
  z_pred[0] = sqrt(pow(x_[0], 2)+pow(x_[1], 2));
  if (0.00001<x_[0]<0.00001)
  {
      help = 0.00001;
  }
  else
  {
      help = x_[0];
  }
  z_pred[1] = atan2(x_[1], help);
  if (0.00001<(pow(x_[0], 2)+pow(x_[1], 2))<0.00001)
  {
      help = 0.00001;
  }
  else
  {
      help = sqrt(pow(x_[0], 2)+pow(x_[1], 2));
  }
  z_pred[2] = (x_[2]*x_[0]+x_[3]*x_[1])/help;
  VectorXd y = z - z_pred;
  MatrixXd H_t = H_.transpose();
  MatrixXd S = H_ * P_ * H_t + R_;
  MatrixXd S_i = S.inverse();
  MatrixXd PH_t = P_ * H_t;
  MatrixXd K = PH_t * S_i;

  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}
