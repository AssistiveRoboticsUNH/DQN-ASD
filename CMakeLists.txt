cmake_minimum_required(VERSION 2.8.3)
project(deep_q_network)

set(PCL_DIR ~/pcl)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  roslib
  rosbag
  std_srvs
  message_generation
)

find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(
  ${catkin_INCLUDE_DIRS}
)

###########
## Build ##
###########

add_service_files(
  FILES
  PointCloud2Array.srv
  DQNGetNextAction.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime
)

#################
## Executables ##
#################

add_executable(pcl_server src/pcl_server.cpp)
target_link_libraries(pcl_server ${PCL_LIBRARIES} ${catkin_LIBRARIES})
add_dependencies( pcl_server ${PROJECT_NAME}_generate_messages_cpp )

add_executable(dqn_executor src/main.cpp src/dqn_executor.cpp src/executor.cpp)
target_link_libraries(dqn_executor ${catkin_LIBRARIES})
add_dependencies( dqn_executor ${PROJECT_NAME}_generate_messages_cpp )