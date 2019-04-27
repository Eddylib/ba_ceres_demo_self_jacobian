//
// Created by libaoyu on 19-4-24.
//

#ifndef CERES_LEARNING_BAPROBLEM_H
#define CERES_LEARNING_BAPROBLEM_H


// Read a Bundle Adjustment in the Large dataset.
#include <iostream>
#include <cassert>
#include <BAPublisher.h>

class BALProblem {
public:
    ~BALProblem() {
        delete[] point_index_;
        delete[] camera_index_;
        delete[] observations_;
        delete[] parameters_;
    }
    int num_observations()       const { return num_observations_;               }
    const double* observations() const { return observations_;                   }
    double* mutable_cameras()          { return parameters_;                     }
    double* mutable_points()           { return parameters_  + 9 * num_cameras_; }
    double* mutable_camera_for_observation(int i) {
        return mutable_cameras() + camera_index_[i] * 9;
    }
    double* mutable_camera_for_cameraidx(int i) {
        return mutable_cameras() + i * 9;
    }
    double* mutable_point_for_observation(int i) {
        return mutable_points() + point_index_[i] * 3;
    }
    double* mutable_point_for_pointidx(int i) {
        return mutable_points() + i * 3;
    }
    bool LoadFile(const char* filename) {
        FILE* fptr = fopen(filename, "r");
        if (fptr == NULL) {
            return false;
        };
        FscanfOrDie(fptr, "%d", &num_cameras_);
        FscanfOrDie(fptr, "%d", &num_points_);
        FscanfOrDie(fptr, "%d", &num_observations_);
        point_index_ = new int[num_observations_];
        camera_index_ = new int[num_observations_];
        observations_ = new double[2 * num_observations_];
        num_parameters_ = 9 * num_cameras_ + 3 * num_points_;
        parameters_ = new double[num_parameters_];
        for (int i = 0; i < num_observations_; ++i) {
            FscanfOrDie(fptr, "%d", camera_index_ + i);
            FscanfOrDie(fptr, "%d", point_index_ + i);
            for (int j = 0; j < 2; ++j) {
                FscanfOrDie(fptr, "%lf", observations_ + 2*i + j);
            }
        }
        for (int i = 0; i < num_parameters_; ++i) {
            FscanfOrDie(fptr, "%lf", parameters_ + i);
        }
        return true;
    }
    int num_points()const{return num_points_;}
    int num_cameras()const{return num_cameras_;}
    int point_idx(int obidx) const {return point_index_[obidx];}
    int camera_idx(int obidx) const {return camera_index_[obidx];}
private:
    template<typename T>
    void FscanfOrDie(FILE *fptr, const char *format, T *value) {
        int num_scanned = fscanf(fptr, format, value);
//        assert(num_scanned != 1 &&  "Invalid UW data file.");
    }
    int num_cameras_;
    int num_points_;
    int num_observations_;
    int num_parameters_;
    int* point_index_;
    int* camera_index_;
    double* observations_;
    double* parameters_;
};

#endif //CERES_LEARNING_BAPROBLEM_H
