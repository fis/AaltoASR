#  SNDFILE_INCLUDE_DIRS   - where to find sndfile.h
#  SNDFILE_LIBRARIES      - List of libraries when using sndfile
#  SNDFILE_FOUND          - True if sndfile found


FIND_PATH(SNDFILE_INCLUDE_DIR NAMES sndfile.h)
MARK_AS_ADVANCED(SNDFILE_INCLUDE_DIR)

# Look for the library.
FIND_LIBRARY(SNDFILE_LIBRARY NAMES sndfile)
MARK_AS_ADVANCED(SNDFILE_LIBRARY)

# If cross-compiling, also include the Ogg, FLAC and Vorbis libs
if(CMAKE_SYSTEM_NAME)
  FIND_LIBRARY(VORBIS_LIBRARY NAMES vorbis)
  FIND_LIBRARY(OGG_LIBRARY NAMES ogg)
  FIND_LIBRARY(FLAC_LIBRARY NAMES FLAC)
  MARK_AS_ADVANCED(VORBIS_LIBRARY OGG_LIBRARY FLAC_LIBRARY)
  SET(_extravars VORBIS_LIBRARY OGG_LIBRARY FLAC_LIBRARY)
  SET(_extralibs ${VORBIS_LIBRARY} ${OGG_LIBRARY} ${FLAC_LIBRARY})
else()
  SET(_extravars)
  SET(_extralibs)
endif(CMAKE_SYSTEM_NAME)

# handle the QUIETLY and REQUIRED arguments and set SNDFILE_FOUND to 
# TRUE if all listed variables are TRUE
include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(SNDFILE
                                  REQUIRED_VARS SNDFILE_LIBRARY ${_extravars}
                                  SNDFILE_INCLUDE_DIR )

IF(SNDFILE_FOUND)
  SET(SNDFILE_LIBRARIES ${SNDFILE_LIBRARY} ${_extralibs})
  SET(SNDFILE_INCLUDE_DIRS ${SNDFILE_INCLUDE_DIR})
ENDIF(SNDFILE_FOUND)
