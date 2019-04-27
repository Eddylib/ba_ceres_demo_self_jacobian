//
// Created by libaoyu on 19-4-27.
//

#ifndef CERES_LEARNING_UTILS_H
#define CERES_LEARNING_UTILS_H

#include <BAPublisher.h>
#include <vector>
#include "baproblem.h"
void convert_cam_to_publis_format(double *state, double *published);
void draw_data(BAPublisher &publisher,const string &name,std::vector<double*> &all_camera_parms, BALProblem &bal_problem);
#endif //CERES_LEARNING_UTILS_H
