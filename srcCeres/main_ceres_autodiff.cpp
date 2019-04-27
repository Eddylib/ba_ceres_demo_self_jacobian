//
// Created by libaoyu on 19-4-24.
//

// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: keir@google.com (Keir Mierle)
//
// A minimal, self-contained bundle adjuster using Ceres, that reads
// files from University of Washington' Bundle Adjustment in the Large dataset:
// http://grail.cs.washington.edu/projects/bal
//
// This does not use the best configuration for solving; see the more involved
// bundle_adjuster.cc file for details.
#include <cmath>
#include <cstdio>
#include <iostream>
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "../utils/utils.h"
#include "../utils/baproblem.h"
// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 9 parameters: 3 for rotation, 3 for translation, 1 for
// focal length and 2 for radial distortion. The principal point is not modeled
// (i.e. it is assumed be located at the image center).
struct SnavelyReprojectionError {
    SnavelyReprojectionError(double observed_x, double observed_y)
            : observed_x(observed_x), observed_y(observed_y) {}
    template <typename T>
    bool operator()(const T* const camera,
                    const T* const point,
                    T* residuals) const {
        // camera[0,1,2] are the angle-axis rotation.
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);
        // camera[3,4,5] are the translation.
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];
        // Compute the center of distortion. The sign change comes from
        // the camera model that Noah Snavely's Bundler assumes, whereby
        // the camera coordinate system has a negative z axis.
        T xp = - p[0] / p[2];
        T yp = - p[1] / p[2];
        // Apply second and fourth order radial distortion.
        const T& l1 = camera[7];
        const T& l2 = camera[8];
        T r2 = xp*xp + yp*yp;
        T distortion = 1.0 + r2  * (l1 + l2  * r2);
        // Compute final projected point position.
        const T& focal = camera[6];
        T predicted_x = focal * distortion * xp;
        T predicted_y = focal * distortion * yp;
        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - observed_x;
        residuals[1] = predicted_y - observed_y;

//        if(abs(-2.21315 - point[0]) < 0.0000001)
//        {
//            for (int i = 0; i < 9; ++i) {
//                cout<<camera[i]<<endl;
//            }
//            cout<<"---------"<<endl;
//            for (int i = 0; i < 3; ++i) {
//                cout<<point[i]<<endl;
//            }
//            cout<<"---------"<<endl;
//            for (int i = 0; i < 3; ++i) {
//                cout<<p[i]<<endl;
//            }
//            cout<<"---------"<<endl;
//            cout<<residuals[0]<<endl;
//            cout<<residuals[1]<<endl;
//            exit(-1);
//        }
        return true;
    }
    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const double observed_x,
                                       const double observed_y) {
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
                new SnavelyReprojectionError(observed_x, observed_y)));
    }
    double observed_x;
    double observed_y;
};
int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    if (argc != 2) {
        std::cerr << "usage: simple_bundle_adjuster <bal_problem>\n";
        return 1;
    }
    BAPublisher publisher = BAPublisher::createInstance("ceres_ba",argc,argv);
    BALProblem bal_problem;
    if (!bal_problem.LoadFile(argv[1])) {
        std::cerr << "ERROR: unable to open file " << argv[1] << "\n";
        return 1;
    }
    std::vector<double*> all_camera_parms;
    for (int i = 0; i < bal_problem.num_cameras(); ++i) {
        all_camera_parms.push_back(bal_problem.mutable_camera_for_cameraidx(i));
    }
    const double* observations = bal_problem.observations();

    ceres::Problem problem;
    // Create residuals for each observation in the bundle adjustment problem. The
    // parameters for cameras and points are added automatically.
    for (int i = 0; i < bal_problem.num_points(); ++i) {
        problem.AddParameterBlock(bal_problem.mutable_point_for_pointidx(i),3);
    }

    for (int i = 0; i < bal_problem.num_cameras(); ++i) {
        problem.AddParameterBlock(bal_problem.mutable_camera_for_cameraidx(i),9);
    }
    for (int i = 0; i < bal_problem.num_observations(); ++i) {
        // Each Residual block takes a point and a camera as input and outputs a 2
        // dimensional residual. Internally, the cost function stores the observed
        // image location and compares the reprojection against the observation.
        ceres::CostFunction* cost_function =
                SnavelyReprojectionError::Create(observations[2 * i + 0],
                                                 observations[2 * i + 1]);
//        ceres::LossFunction* loss_function =new ceres::HuberLoss(1.0);
        problem.AddResidualBlock(cost_function,
                                 nullptr /* squared loss */,
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
    draw_data(publisher,"ceres/before",all_camera_parms,bal_problem);
    ceres::Solve(options, &problem, &summary);
    draw_data(publisher,"ceres/after",all_camera_parms,bal_problem);
    std::cout << summary.FullReport() << "\n";
    while(1){
        sleep(1);
    }
    return 0;
}