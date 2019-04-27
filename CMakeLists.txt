cmake_minimum_required(VERSION 3.12)
project(ceres_learning)

set(CMAKE_CXX_STANDARD 14)


include_directories("/home/libaoyu/Desktop/deplib/eigen3")
include_directories("${PROJECT_SOURCE_DIR}/third_party/Sophus")

set(Eigen3_DIR /home/libaoyu/Desktop/dist_root/usr/local/share/eigen3/cmake/)
find_package(Eigen3 REQUIRED)

set(Ceres_DIR /home/libaoyu/Data/ceres-solver/dist/usr/local/lib/cmake/Ceres)
find_package(Ceres REQUIRED)
message(STATUS ${CERES_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})

set(ROS_WS /home/libaoyu/Data/catkin_ws)
set(LIB_PUBLISHER ${ROS_WS}/devel/lib/libba_publisher.so)
set(INC_PUBLISHER ${ROS_WS}/src/showpath/include/showpath)

add_library(utils utils/utils.h utils/utils.cpp)
target_include_directories(utils PUBLIC ${INC_PUBLISHER})

add_executable(main_my_jacobian srcBA/main_my_jacobian.cpp)
target_include_directories(main_my_jacobian PUBLIC /home/libaoyu/Data/catkin_ws/src/showpath/include/showpath)
target_link_libraries(main_my_jacobian ${CERES_LIBRARIES})
target_link_libraries(main_my_jacobian ${LIB_PUBLISHER})
target_link_libraries(main_my_jacobian utils)

add_executable(main_ceres_autodiff srcCeres/main_ceres_autodiff.cpp)
target_include_directories(main_ceres_autodiff PUBLIC ${INC_PUBLISHER})
target_link_libraries(main_ceres_autodiff ${CERES_LIBRARIES})
target_link_libraries(main_ceres_autodiff ${LIB_PUBLISHER})
target_link_libraries(main_ceres_autodiff utils)
