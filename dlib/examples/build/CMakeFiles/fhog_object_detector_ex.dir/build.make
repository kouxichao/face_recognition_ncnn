# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/xichao/miniconda3/bin/cmake

# The command to remove a file.
RM = /home/xichao/miniconda3/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/xichao/dlib-19.16/examples

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/xichao/dlib-19.16/examples/build

# Include any dependencies generated for this target.
include CMakeFiles/fhog_object_detector_ex.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/fhog_object_detector_ex.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/fhog_object_detector_ex.dir/flags.make

CMakeFiles/fhog_object_detector_ex.dir/fhog_object_detector_ex.cpp.o: CMakeFiles/fhog_object_detector_ex.dir/flags.make
CMakeFiles/fhog_object_detector_ex.dir/fhog_object_detector_ex.cpp.o: ../fhog_object_detector_ex.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xichao/dlib-19.16/examples/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/fhog_object_detector_ex.dir/fhog_object_detector_ex.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fhog_object_detector_ex.dir/fhog_object_detector_ex.cpp.o -c /home/xichao/dlib-19.16/examples/fhog_object_detector_ex.cpp

CMakeFiles/fhog_object_detector_ex.dir/fhog_object_detector_ex.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fhog_object_detector_ex.dir/fhog_object_detector_ex.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xichao/dlib-19.16/examples/fhog_object_detector_ex.cpp > CMakeFiles/fhog_object_detector_ex.dir/fhog_object_detector_ex.cpp.i

CMakeFiles/fhog_object_detector_ex.dir/fhog_object_detector_ex.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fhog_object_detector_ex.dir/fhog_object_detector_ex.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xichao/dlib-19.16/examples/fhog_object_detector_ex.cpp -o CMakeFiles/fhog_object_detector_ex.dir/fhog_object_detector_ex.cpp.s

# Object files for target fhog_object_detector_ex
fhog_object_detector_ex_OBJECTS = \
"CMakeFiles/fhog_object_detector_ex.dir/fhog_object_detector_ex.cpp.o"

# External object files for target fhog_object_detector_ex
fhog_object_detector_ex_EXTERNAL_OBJECTS =

fhog_object_detector_ex: CMakeFiles/fhog_object_detector_ex.dir/fhog_object_detector_ex.cpp.o
fhog_object_detector_ex: CMakeFiles/fhog_object_detector_ex.dir/build.make
fhog_object_detector_ex: dlib_build/libdlib.a
fhog_object_detector_ex: /usr/local/cuda/lib64/libcudart_static.a
fhog_object_detector_ex: /usr/lib/x86_64-linux-gnu/librt.so
fhog_object_detector_ex: /usr/lib/x86_64-linux-gnu/librt.so
fhog_object_detector_ex: /usr/lib/x86_64-linux-gnu/libnsl.so
fhog_object_detector_ex: /usr/lib/x86_64-linux-gnu/libSM.so
fhog_object_detector_ex: /usr/lib/x86_64-linux-gnu/libICE.so
fhog_object_detector_ex: /usr/lib/x86_64-linux-gnu/libX11.so
fhog_object_detector_ex: /usr/lib/x86_64-linux-gnu/libXext.so
fhog_object_detector_ex: /usr/lib/x86_64-linux-gnu/libpng.so
fhog_object_detector_ex: /usr/lib/x86_64-linux-gnu/libz.so
fhog_object_detector_ex: /usr/lib/x86_64-linux-gnu/libjpeg.so
fhog_object_detector_ex: /home/xichao/miniconda3/lib/libmkl_rt.so
fhog_object_detector_ex: /usr/local/cuda/lib64/libcublas.so
fhog_object_detector_ex: /usr/local/cuda/lib64/libcudnn.so
fhog_object_detector_ex: /usr/local/cuda/lib64/libcurand.so
fhog_object_detector_ex: /usr/local/cuda/lib64/libcusolver.so
fhog_object_detector_ex: /usr/lib/x86_64-linux-gnu/libiomp5.so
fhog_object_detector_ex: /usr/local/lib/libsqlite3.so
fhog_object_detector_ex: CMakeFiles/fhog_object_detector_ex.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/xichao/dlib-19.16/examples/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable fhog_object_detector_ex"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/fhog_object_detector_ex.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/fhog_object_detector_ex.dir/build: fhog_object_detector_ex

.PHONY : CMakeFiles/fhog_object_detector_ex.dir/build

CMakeFiles/fhog_object_detector_ex.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/fhog_object_detector_ex.dir/cmake_clean.cmake
.PHONY : CMakeFiles/fhog_object_detector_ex.dir/clean

CMakeFiles/fhog_object_detector_ex.dir/depend:
	cd /home/xichao/dlib-19.16/examples/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/xichao/dlib-19.16/examples /home/xichao/dlib-19.16/examples /home/xichao/dlib-19.16/examples/build /home/xichao/dlib-19.16/examples/build /home/xichao/dlib-19.16/examples/build/CMakeFiles/fhog_object_detector_ex.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/fhog_object_detector_ex.dir/depend

