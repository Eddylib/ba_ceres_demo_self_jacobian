//
// Created by libaoyu on 19-3-31.
//

#include <cmath>
#include <cstdio>
#include <iostream>
#include "ba_structs.h"
#include "ceres/rotation.h"
#include <mutex>
#include <sophus/se3.hpp>
#include <fstream>
#include "../utils/baproblem.h"
#include "../utils/utils.h"
#include <BAPublisher.h>

struct camera_parm{
    double state[9]{};

    explicit camera_parm(double *data){
        // angle axis --> se3 tangent vector
        Eigen::Vector3d angleAxisData(data[0],data[1],data[2]);
        Eigen::Vector3d translation(data[3],data[4],data[5]);
        double radian = angleAxisData.norm();
        angleAxisData /= radian;
        Eigen::Quaterniond quaternion;
        quaternion = Eigen::AngleAxisd(radian,angleAxisData);
        Sophus::SE3<double > se3(quaternion,translation);
        for (int i = 0; i < 6; ++i) {
            state[i] = se3.log()(i);
        }
        for (int i = 6; i < 9; ++i) {
            state[i] = data[i];
        }
    }
    operator double *(){
        return state;
    }
};
void dump_data(const char * file,std::vector<camera_parm*> &all_camera_parms, BALProblem &bal_problem){
    std::cout<<"dumpoing to "<<file<<std::endl;
    std::ofstream file_to_save(file);
    file_to_save<<all_camera_parms.size()<<std::endl;
    for (int i = 0; i < all_camera_parms.size(); ++i) {
        for (int j = 0; j < 9; ++j) {
            file_to_save<<all_camera_parms[i]->state[j]<<" ";
        }
        file_to_save<<std::endl;
    }

    file_to_save<<bal_problem.num_points()<<std::endl;
    for (int i = 0; i < bal_problem.num_points(); ++i) {
        double *point = bal_problem.mutable_point_for_pointidx(i);
        for (int j = 0; j < 3; ++j) {
            file_to_save<<point[j]<<" ";
        }
        file_to_save<<std::endl;
    }
    std::cout<<"dumpoing end "<<file<<std::endl;
}
int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    if (argc != 2) {
        std::cerr << "usage: simple_bundle_adjuster <bal_problem>\n";
        return 1;
    }
    BAPublisher publisher = BAPublisher::createInstance("lby_ba",argc,argv);
    BALProblem bal_problem;
    if (!bal_problem.LoadFile(argv[1])) {
        std::cerr << "ERROR: unable to open file " << argv[1] << "\n";
        return 1;
    }
    const double* observations = bal_problem.observations();
    // Create residuals for each observation in the bundle adjustment problem. The
    // parameters for cameras and points are added automatically.

    ceres::Problem problem;
    std::vector<double*> all_camera_parms;
    for (int i = 0; i < bal_problem.num_points(); ++i) {
        problem.AddParameterBlock(bal_problem.mutable_point_for_pointidx(i),3);
    }
    for (int i = 0; i < bal_problem.num_cameras(); ++i) {
        all_camera_parms.push_back(bal_problem.mutable_camera_for_cameraidx(i));
        problem.AddParameterBlock(bal_problem.mutable_camera_for_cameraidx(i),9, new PoseSE3Parameterization());
    }
    for (int i = 0; i < bal_problem.num_observations(); ++i) {
        ceres::CostFunction *costFunction = new SE3CostFunction(
                bal_problem.observations()[i*2],
                bal_problem.observations()[i*2+1],
                bal_problem.mutable_camera_for_observation(i),
                bal_problem.mutable_point_for_observation(i)
                );
//        ceres::LossFunction* loss_function =new ceres::HuberLoss(1.0);
        problem.AddResidualBlock(costFunction,
                nullptr,
                bal_problem.mutable_camera_for_observation(i),
                bal_problem.mutable_point_for_observation(i));
    }
    // Make Ceres automatically detect the bundle structure. Note that the
    // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
    // for standard bundle adjustment problems.
    ceres::Solver::Options options;
//    options.gradient_tolerance = 1e-16;
//    options.function_tolerance = 1e-16;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    draw_data(publisher,"lby/before",all_camera_parms,bal_problem);
    ceres::Solve(options, &problem, &summary);
    draw_data(publisher,"lby/after",all_camera_parms,bal_problem);
    std::cout << summary.FullReport() << "\n";
    while(1){
        sleep(1);
    }
    return 0;
}