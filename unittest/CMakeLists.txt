cmake_minimum_required(VERSION 3.15)
set (CMAKE_CXX_STANDARD 17)
project(asv_test)
set(SOURCE asv_test.cpp)
INCLUDE_DIRECTORIES(
    target_include_directories(${PROJECT_NAME}
        SYSTEM PRIVATE
        /usr/local/include/opencv4
		${WORKSPACE_SOURCE_DIR}/src
    ))
LINK_DIRECTORIES(/usr/local/lib)
add_executable(${PROJECT_NAME} ${SOURCE})
TARGET_LINK_LIBRARIES(${PROJECT_NAME}
	libopencv_core.dylib
	libopencv_highgui.dylib
	libopencv_imgcodecs.dylib
	libopencv_imgproc.dylib
	libopencv_features2d.dylib
	libopencv_flann.dylib
)

# Treat warnings as errors and enable common warnings (especially for GCC)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	target_compile_options(${PROJECT_NAME} PRIVATE
		-Wall -Wextra -Wpedantic -Wconversion -Werror
	)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	# Clang: include explicit implicit-int-float warning to catch double->int
	target_compile_options(${PROJECT_NAME} PRIVATE
		-Wall -Werror -Wconversion
	)
elseif (MSVC)
	target_compile_options(${PROJECT_NAME} PRIVATE /W4 /WX)
endif()