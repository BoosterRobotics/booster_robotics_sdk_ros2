set(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(CMAKE_CXX_STANDARD 17)


set(EXTRA_RPATH "."
        ".."
        "../lib"
        "./lib"
        "../frame_work/lib"
        "/opt/booster/frame_work/lib"
)


message("EXTRA_PATH: ${EXTRA_RPATH}")


# 跳过编译的RPath
set(CMAKE_SKIP_BUILD_RPATH false)
# 跳过安装的RPath
set(CMAKE_SKIP_INSTALL_RPATH false)

set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH}" "${EXTRA_RPATH}")
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${EXTRA_RPATH}")
