project(tools)

if(NOT CROSS_MINGW) # not cross-compiling
  add_subdirectory( playseg )
  add_subdirectory( lattice_rescore )
endif(NOT CROSS_MINGW)
add_subdirectory( morph_lattice )
