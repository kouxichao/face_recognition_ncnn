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
include CMakeFiles/fhog_ex.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/fhog_ex.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/fhog_ex.dir/flags.make

CMakeFiles/fhog_ex.dir/fhog_ex.cpp.o: CMakeFiles/fhog_ex.dir/flags.make
CMakeFiles/fhog_ex.dir/fhog_ex.cpp.o: ../fhog_ex.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xichao/dlib-19.16/examples/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/fhog_ex.dir/fhog_ex.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fhog_ex.dir/fhog_ex.cpp.o -c /home/xichao/dlib-19.16/examples/fhog_ex.cpp

CMakeFiles/fhog_ex.dir/fhog_ex.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fhog_ex.dir/fhog_ex.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xichao/dlib-19.16/examples/fhog_ex.cpp > CMakeFiles/fhog_ex.dir/fhog_ex.cpp.i

CMakeFiles/fhog_ex.dir/fhog_ex.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fhog_ex.dir/fhog_ex.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xichao/dlib-19.16/examples/fhog_ex.cpp -o CMakeFiles/fhog_ex.dir/fhog_ex.cpp.s

# Object files for target fhog_ex
fhog_ex_OBJECTS = \
"CMakeFiles/fhog_ex.dir/fhog_ex.cpp.o"

# External object files for target fhog_ex
fhog_ex_EXTERNAL_OBJECTS =

fhog_ex: CMakeFiles/fhog_ex.dir/fhog_ex.cpp.o
fhog_ex: CMakeFiles/fhog_ex.dir/build.make
fhog_ex: dlib_build/libdlib.a
fhog_ex: /usr/local/cuda/lib64/libcudart_static.a
fhog_ex: /usr/lib/x86_64-linux-gnu/librt.so
fhog_ex: /usr/lib/x86_64-linux-gnu/librt.so
fhog_ex: /usr/lib/x86_64-linux-gnu/libnsl.so
fhog_ex: /usr/lib/x86_64-linux-gnu/libSM.so
fhog_ex: /usr/lib/x86_64-linux-gnu/libICE.so
fhog_ex: /usr/lib/x86_64-linux-gnu/libX11.so
fhog_ex: /usr/lib/x86_64-linux-gnu/libXext.so
fhog_ex: /usr/lib/x86_64-linux-gnu/libpng.so
fhog_ex: /usr/lib/x86_64-linux-gnu/libz.so
fhog_ex: /usr/lib/x86_64-linux-gnu/libjpeg.so
fhog_ex: /home/xichao/miniconda3/lib/libmkl_rt.so
fhog_ex: /usr/local/cuda/lib64/libcublas.so
fhog_ex: /usr/local/cuda/lib64/libcudnn.so
fhog_ex: /usr/local/cuda/lib64/libcurand.so
fhog_ex: /usr/local/cuda/lib64/libcusolver.so
fhog_ex: /usr/lib/x86_64-linux-gnu/libiomp5.so
fhog_ex: /usr/local/lib/libsqlite3.so
fhog_ex: CMakeFiles/fhog_ex.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/xichao/dlib-19.16/examples/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable fhog_ex"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/fhog_ex.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/fhog_ex.dir/build: fhog_ex

.PHONY : CMakeFiles/fhog_ex.dir/build

CMakeFiles/fhog_ex.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/fhog_ex.dir/cmake_clean.cmake
.PHONY : CMakeFiles/fhog_ex.dir/clean

CMakeFiles/fhog_ex.dir/depend:
	cd /home/xichao/dlib-19.16/examples/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/xichao/dlib-19.16/examples /home/xichao/dlib-19.16/examples /home/xichao/dlib-19.16/examples/build /home/xichao/dlib-19.16/examples/build /home/xichao/dlib-19.16/examples/build/CMakeFiles/fhog_ex.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/fhog_ex.dir/depend

