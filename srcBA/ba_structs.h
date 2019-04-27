//
// Created by libaoyu on 19-4-24.
//

#ifndef CERES_LEARNING_SETTINGS_H
#define CERES_LEARNING_SETTINGS_H
#include <iostream>
#include <sophus/se3.hpp>
#include "ceres/ceres.h"
#include "ceres/rotation.h"
using namespace std;
// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 9 parameters: 3 for rotation, 3 for translation, 1 for
// focal length and 2 for radial distortion. The principal point is not modeled
// (i.e. it is assumed be located at the image center).


class SE3CostFunction : public ceres::SizedCostFunction<2,9,3> {
public:
    typedef Sophus::SE3<double>::Tangent Tangent;
    typedef Eigen::Vector3d Point;
    typedef Eigen::Vector2d Point2d;
    SE3CostFunction(double observed_x, double observed_y,double *oricam,double *oripoint)
            : observed_x(observed_x), observed_y(observed_y),original_cam_data(oricam),original_point_data(oripoint) {}
    virtual ~SE3CostFunction() {}
private:
    //tmp vars
    mutable Eigen::Matrix<double,2,2,Eigen::RowMajor> jdrdpc;
    mutable Eigen::Matrix<double,2,3,Eigen::RowMajor> jdpcdPc;
    mutable Eigen::Matrix<double,2,3,Eigen::RowMajor> jdrdPc;
    mutable Eigen::Matrix<double,3,6,Eigen::RowMajor> jdPcdxi;
public:
    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const {
        double const *camera = parameters[0];
        double const *point = parameters[1];
        Eigen::Map<const Eigen::Vector3d> angleaxisData(camera);
        Eigen::AngleAxisd rotation(angleaxisData.norm(),angleaxisData/angleaxisData.norm());
        Eigen::Map<const Eigen::Vector3d> translation(camera+3);
        Sophus::SE3<double> se3State(rotation.toRotationMatrix(),translation);
        Eigen::Map<const Eigen::Vector3d> Pw(point);
        Eigen::Vector3d Pc = se3State*Pw;

        Eigen::Vector2d pc;
        pc[0] = -Pc[0]/Pc[2];
        pc[1] = -Pc[1]/Pc[2];

        // Apply second and fourth order radial distortion.
        const double& l1 = camera[7];
        const double& l2 = camera[8];
        double r2 = pc.transpose()*pc;
        double distortion = 1.0 + r2  * (l1 + l2  * r2);
        const double& focal = camera[6];

        // Compute final projected point position.
        Eigen::Vector2d pfinal = focal*distortion*pc;
        // The error is the difference between the predicted and observed position.
        residuals[0] = pfinal[0] - observed_x;
        residuals[1] = pfinal[1] - observed_y;
        if(jacobians){
            double A = r2;
            double B = (l1 + A * l2);
            jdrdpc = focal*(
                    2.*(B + A*l2)*pc*pc.transpose()
                    + distortion*Eigen::Matrix<double,2,2>::Identity()
                    );
            jdpcdPc<<
            -1./Pc(2),  0,          Pc(0)/(Pc(2)*Pc(2)),
            0,          -1./Pc(2),  Pc(1)/(Pc(2)*Pc(2));
            jdrdPc = jdrdpc * jdpcdPc;
            if(jacobians[0]){ //jdrdcam
                Eigen::Map<Eigen::Matrix<double,2,9,Eigen::RowMajor>> jdrdcam(jacobians[0]);
//                Eigen::Matrix<double,2,6> jdrdxi = jdrdcam.block(0,0,2,6);
                jdPcdxi.block(0,0,3,3).setIdentity();
                jdPcdxi.block(0,3,3,3)=Sophus::SO3<double>::hat(-Pc);
                jdrdcam.block(0,0,2,6) = jdrdPc*jdPcdxi;

                jdrdcam.block(0,6,2,1) = distortion*pc; //drdf
                jdrdcam.block(0,7,2,1) = focal*r2*pc; //drdl1
                jdrdcam.block(0,8,2,1) = focal*r2*r2*pc; //drdl2
            }
            if(jacobians[1]){//jdrdpoint
                Eigen::Map<Eigen::Matrix<double,2,3,Eigen::RowMajor>> jdrdPw(jacobians[1]);
                jdrdPw = jdrdPc*rotation.toRotationMatrix();
            }
        }
        return true;
    }
private:
    // Observations for a sample.
    double observed_x;
    double observed_y;
    double *original_cam_data;
    double *original_point_data;
};

class PoseSE3Parameterization : public ceres::LocalParameterization{
    virtual ~PoseSE3Parameterization() {}
    virtual bool Plus(const double* x,
                      const double* delta,
                      double* x_plus_delta) const{
        Eigen::Map<const Eigen::Vector3d> x_trans(x + 3);
        Eigen::Matrix<double,6,1> se3log;
//        se3log<<delta[3],delta[4],delta[5],delta[0],delta[1],delta[2];
        Sophus::SE3<double> se3_delta = Sophus::SE3<double>::exp(Eigen::Map<const Eigen::Matrix<double,6,1>>(delta));
//        Sophus::SE3<double> se3_delta = Sophus::SE3<double>::exp(se3log);
        Eigen::Map<const Eigen::Vector3d> angleaxisvec(x);
        Eigen::AngleAxisd x_angleaxis = Eigen::AngleAxisd(angleaxisvec.norm(),angleaxisvec/angleaxisvec.norm());
        Sophus::SE3<double> se3_x = Sophus::SE3<double>(Eigen::Quaterniond(x_angleaxis),x_trans);
        Sophus::SE3<double> se3_x_plus_delta = se3_delta*se3_x;


        Eigen::Map<Eigen::Matrix<double,3,1>> x_plus_delta_rot(x_plus_delta);
        Eigen::Map<Eigen::Matrix<double,3,1>> x_plus_delta_trans(x_plus_delta+3);

        Eigen::AngleAxisd angleAxisd_x_plus_delta(se3_x_plus_delta.rotationMatrix());
        x_plus_delta_rot = angleAxisd_x_plus_delta.angle()*angleAxisd_x_plus_delta.axis();
        x_plus_delta_trans = se3_x_plus_delta.translation();
        x_plus_delta[6] = x[6] + delta[6];
        x_plus_delta[7] = x[7] + delta[7];
        x_plus_delta[8] = x[8] + delta[8];
        return true;
    };
    virtual bool ComputeJacobian(const double* x, double* jacobian) const {
        Eigen::Map<Eigen::Matrix<double, 9, 9, Eigen::RowMajor> > J(jacobian);
        J.setIdentity();
        return true;
    };

    virtual int GlobalSize() const {
        return 9;
    };
    virtual int LocalSize() const {
        return 9;
    };
};
#endif //CERES_LEARNING_SETTINGS_H
