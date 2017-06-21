#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  H_j_ = MatrixXd(3, 4);
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.Q_ = MatrixXd(4, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  //measurement function - laser
  H_laser_ << 1, 0, 0, 0,
             0, 1, 0, 0;

  //state transition matrix
  ekf_.F_ << 1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1;

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    // first measurement
    cout << "EKF: " << endl;

    //initialize Vector
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    //initialize covariance matrix
    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ << 1, 0, 0, 0,
               0, 1, 0, 0,
               0, 0, 1000, 0,
               0, 0, 0, 1000;

    //convert from polar to cartesian, if first measurement is radar
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
        ekf_.x_[0] = measurement_pack.raw_measurements_[0]*cos(measurement_pack.raw_measurements_[1]);
        ekf_.x_[1] = measurement_pack.raw_measurements_[0]*sin(measurement_pack.raw_measurements_[1]);
        ekf_.x_[2] = measurement_pack.raw_measurements_[2]*cos(measurement_pack.raw_measurements_[1]);
        ekf_.x_[3] = measurement_pack.raw_measurements_[2]*sin(measurement_pack.raw_measurements_[1]);

    }
    //just use first measurement, if it is radar (use velocity = 0 in booth directions)
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      ekf_.x_[0] = measurement_pack.raw_measurements_[0];
      ekf_.x_[1] = measurement_pack.raw_measurements_[1];
      ekf_.x_[2] = 0;
      ekf_.x_[3] = 0;
    }

    //Set first timestamp and set initialization to true
    previous_timestamp_ = measurement_pack.timestamp_;
    is_initialized_ = true;
    return;
  }

    //measure time between timesteps
    float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
    previous_timestamp_ = measurement_pack.timestamp_;

    //Updating correct values to state transition matrix
    ekf_.F_(0, 2) = dt;
    ekf_.F_(1, 3) = dt;

    float dt_2 = pow(dt, 2);
    float dt_3 = pow(dt, 3);
    float dt_4 = pow(dt, 4);

    //set noise values
    int noise_ax = 9;
    int noise_ay = 9;

    //Use noise values for Q-Matix
    ekf_.Q_ << (dt_4/4)*noise_ax, 0, (dt_3/2)*noise_ax, 0,
                0, (dt_4/4)*noise_ay, 0, (dt_3/2)*noise_ay,
                (dt_3/2)*noise_ax, 0, (dt_2)*noise_ax, 0,
                0, (dt_3/2)*noise_ay, 0, (dt_2)*noise_ay;

  //Use Kalman Prediction
  ekf_.Predict();


  //Use Update or UpdateEKF (for laser and radar respectively
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    //Using CalculateJacobian to get correct measurement function for radar
    //Update measurement function and measurement covariance matrix in booth cases
    H_j_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.H_ = H_j_;
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
