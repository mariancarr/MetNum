# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/d/metodosnum

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/d/metodosnum/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/metodosnum.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/metodosnum.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/metodosnum.dir/flags.make

CMakeFiles/metodosnum.dir/Tp2/main.cpp.o: CMakeFiles/metodosnum.dir/flags.make
CMakeFiles/metodosnum.dir/Tp2/main.cpp.o: ../Tp2/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/d/metodosnum/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/metodosnum.dir/Tp2/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/metodosnum.dir/Tp2/main.cpp.o -c /mnt/d/metodosnum/Tp2/main.cpp

CMakeFiles/metodosnum.dir/Tp2/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/metodosnum.dir/Tp2/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/metodosnum/Tp2/main.cpp > CMakeFiles/metodosnum.dir/Tp2/main.cpp.i

CMakeFiles/metodosnum.dir/Tp2/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/metodosnum.dir/Tp2/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/metodosnum/Tp2/main.cpp -o CMakeFiles/metodosnum.dir/Tp2/main.cpp.s

# Object files for target metodosnum
metodosnum_OBJECTS = \
"CMakeFiles/metodosnum.dir/Tp2/main.cpp.o"

# External object files for target metodosnum
metodosnum_EXTERNAL_OBJECTS =

metodosnum: CMakeFiles/metodosnum.dir/Tp2/main.cpp.o
metodosnum: CMakeFiles/metodosnum.dir/build.make
metodosnum: CMakeFiles/metodosnum.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/d/metodosnum/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable metodosnum"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/metodosnum.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/metodosnum.dir/build: metodosnum

.PHONY : CMakeFiles/metodosnum.dir/build

CMakeFiles/metodosnum.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/metodosnum.dir/cmake_clean.cmake
.PHONY : CMakeFiles/metodosnum.dir/clean

CMakeFiles/metodosnum.dir/depend:
	cd /mnt/d/metodosnum/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/d/metodosnum /mnt/d/metodosnum /mnt/d/metodosnum/cmake-build-debug /mnt/d/metodosnum/cmake-build-debug /mnt/d/metodosnum/cmake-build-debug/CMakeFiles/metodosnum.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/metodosnum.dir/depend
