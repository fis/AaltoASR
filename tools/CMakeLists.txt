project(tools)

if(NOT CMAKE_SYSTEM_NAME) # not cross-compiling
  add_subdirectory( playseg )
  add_subdirectory( lattice_rescore )
endif(NOT CMAKE_SYSTEM_NAME)
add_subdirectory( morph_lattice )
