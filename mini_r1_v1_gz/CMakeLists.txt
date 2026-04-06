cmake_minimum_required(VERSION 3.8)
project(mini_r1_v1_gz)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)  
find_package(rclpy REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

# Remove or Comment out this line if you don't have a 
# folder named 'mini_r1_v1_gz' with an __init__.py inside it.
# ament_python_install_package(${PROJECT_NAME})

# Correctly install the launch and config directories
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

# Install Python scripts so they are executable
# Note: Ensure the path to 2d_goal_to_action.py is correct relative to CMakeLists.txt
install(PROGRAMS
  scripts/2d_goal_to_action.py
  scripts/obstacle_mover.py 
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()