#!/bin/sh

set -v

closure_deb_download_install VAR_DEB_NAME="ros2-apt-source"

closure_apt_get_install_package VAR_PACKAGE_NAME="ros-humble-ament-cmake"
closure_apt_get_install_package VAR_PACKAGE_NAME="ros-humble-rclcpp"
closure_apt_get_install_package VAR_PACKAGE_NAME="ros-humble-rosidl-default-generators"

closure_apt_get_install_package VAR_PACKAGE_NAME="python3-colcon-common-extensions"

closure_apt_get_remove_package VAR_PACKAGE_NAME="gfortran-12"

. /opt/ros/humble/setup.sh
. "${CUSTOM_ENV_PROJECT_DIRECTORY_PATH}"/shared-file-system/script/s-40-build-component.sh
