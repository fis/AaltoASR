cmake_minimum_required(VERSION 2.8)
PROJECT(AaltoASRdeps)

# CMake project to install all dependencies for a MinGW cross-compiler build

include(ExternalProject)

ExternalProject_Add(libogg
   PREFIX ${CMAKE_BINARY_DIR}/target
   URL http://downloads.xiph.org/releases/ogg/libogg-1.3.2.tar.gz
   URL_MD5 b72e1a1dbadff3248e4ed62a4177e937
   CONFIGURE_COMMAND <SOURCE_DIR>/configure --enable-static=yes --enable-shared=no --prefix=<INSTALL_DIR> --host=${CROSS_TARGET}
   BUILD_COMMAND make
   BUILD_IN_SOURCE 1
   INSTALL_COMMAND make install
)

ExternalProject_add(libvorbis
   DEPENDS libogg
   PREFIX ${CMAKE_BINARY_DIR}/target
   URL http://downloads.xiph.org/releases/vorbis/libvorbis-1.3.4.tar.gz
   URL_MD5 8851c593a52d1ef9c526d95174873852
   CONFIGURE_COMMAND <SOURCE_DIR>/configure --enable-static=yes --enable-shared=no --prefix=<INSTALL_DIR> --host=${CROSS_TARGET} CFLAGS=-I<INSTALL_DIR>/include LDFLAGS=-L<INSTALL_DIR>/lib
   BUILD_COMMAND make
   BUILD_IN_SOURCE 1
   INSTALL_COMMAND make install
)

ExternalProject_Add(flac
   DEPENDS libogg
   PREFIX ${CMAKE_BINARY_DIR}/target
   URL http://downloads.xiph.org/releases/flac/flac-1.2.1.tar.gz
   URL_MD5 153c8b15a54da428d1f0fadc756c22c7
   PATCH_COMMAND patch -p1 < ${CMAKE_CURRENT_LIST_DIR}/flac-1.2.1.patch
   CONFIGURE_COMMAND <SOURCE_DIR>/configure --enable-static=yes --enable-shared=no --prefix=<INSTALL_DIR> --host=${CROSS_TARGET} CFLAGS=-I<INSTALL_DIR>/include LDFLAGS=-L<INSTALL_DIR>/lib
   BUILD_COMMAND make
   BUILD_IN_SOURCE 1
   INSTALL_COMMAND make install
)

ExternalProject_Add(sndfile
   DEPENDS libogg libvorbis flac
   PREFIX ${CMAKE_BINARY_DIR}/target
   URL http://www.mega-nerd.com/libsndfile/files/libsndfile-1.0.25.tar.gz
   URL_MD5 e2b7bb637e01022c7d20f95f9c3990a2
   CONFIGURE_COMMAND <SOURCE_DIR>/configure --enable-static=yes --enable-shared=no --prefix=<INSTALL_DIR> --host=${CROSS_TARGET} CFLAGS=-I<INSTALL_DIR>/include LDFLAGS=-L<INSTALL_DIR>/lib
   BUILD_COMMAND make
   BUILD_IN_SOURCE 1
   INSTALL_COMMAND make install
)

#ExternalProject_Add(libmad
#   PREFIX ${CMAKE_BINARY_DIR}/target
#   URL ftp://ftp.mars.org/pub/mpeg/libmad-0.15.1b.tar.gz
#   URL_MD5 1be543bc30c56fb6bea1d7bf6a64e66c
#   CONFIGURE_COMMAND <SOURCE_DIR>/configure --enable-fpm=intel --enable-static=yes --enable-shared=no --prefix=<INSTALL_DIR> --host=${CROSS_TARGET} CFLAGS=-I<INSTALL_DIR>/include LDFLAGS=-L<INSTALL_DIR>/lib
#   BUILD_COMMAND make
#   BUILD_IN_SOURCE 1
#   INSTALL_COMMAND make install
#)

IF(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  SET(OPENBLAS_BITS 64)
ELSE(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  SET(OPENBLAS_BITS 32)
ENDIF(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
ExternalProject_Add(OpenBLAS
   PREFIX ${CMAKE_BINARY_DIR}/target
   GIT_REPOSITORY https://github.com/xianyi/OpenBLAS.git
   GIT_TAG v0.2.11
   CONFIGURE_COMMAND true
   BUILD_COMMAND make BINARY=${OPENBLAS_BITS} CC=${CMAKE_C_COMPILER} FC=${CMAKE_SYSTEM_PROCESSOR}-w64-mingw32-gfortran HOSTCC=gcc TARGET=CORE2
   BUILD_IN_SOURCE 1
   INSTALL_COMMAND make install PREFIX=${CMAKE_BINARY_DIR}/target
)

