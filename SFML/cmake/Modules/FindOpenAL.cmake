# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindOpenAL
----------

Finds Open Audio Library (OpenAL).

Projects using this module should use ``#include "al.h"`` to include the OpenAL
header file, **not** ``#include <AL/al.h>``.  The reason for this is that the
latter is not entirely portable.  Windows/Creative Labs does not by default put
their headers in ``AL/`` and macOS uses the convention ``<OpenAL/al.h>``.

Hints
^^^^^

Environment variable ``$OPENALDIR`` can be used to set the prefix of OpenAL
installation to be found.

By default on macOS, system framework is search first.  In other words,
OpenAL is searched in the following order:

1. System framework: ``/System/Library/Frameworks``, whose priority can be
   changed via setting the :variable:`CMAKE_FIND_FRAMEWORK` variable.
2. Environment variable ``$OPENALDIR``.
3. System paths.
4. User-compiled framework: ``~/Library/Frameworks``.
5. Manually compiled framework: ``/Library/Frameworks``.
6. Add-on package: ``/opt``.

IMPORTED Targets
^^^^^^^^^^^^^^^^

.. versionadded:: 3.25

This module defines the :prop_tgt:`IMPORTED` target:

``OpenAL::OpenAL``
  The OpenAL library, if found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``OPENAL_FOUND``
  If false, do not try to link to OpenAL
``OPENAL_INCLUDE_DIR``
  OpenAL include directory
``OPENAL_LIBRARY``
  Path to the OpenAL library
``OPENAL_VERSION_STRING``
  Human-readable string containing the version of OpenAL
#]=======================================================================]

# For Windows, Creative Labs seems to have added a registry key for their
# OpenAL 1.1 installer. I have added that key to the list of search paths,
# however, the key looks like it could be a little fragile depending on
# if they decide to change the 1.00.0000 number for bug fix releases.
# Also, they seem to have laid down groundwork for multiple library platforms
# which puts the library in an extra subdirectory. Currently there is only
# Win32 and I have hardcoded that here. This may need to be adjusted as
# platforms are introduced.
# The OpenAL 1.0 installer doesn't seem to have a useful key I can use.
# I do not know if the Nvidia OpenAL SDK has a registry key.

find_path(OPENAL_INCLUDE_DIR al.h
  HINTS
    ENV OPENALDIR
  PATHS
    ~/Library/Frameworks
    /Library/Frameworks
    /opt
    [HKEY_LOCAL_MACHINE\\SOFTWARE\\Creative\ Labs\\OpenAL\ 1.1\ Software\ Development\ Kit\\1.00.0000;InstallDir]
  PATH_SUFFIXES include/AL include/OpenAL include AL OpenAL
  )

find_library(OPENAL_LIBRARY
  NAMES OpenAL al openal OpenAL32
  HINTS
    ENV OPENALDIR
  PATHS
    ~/Library/Frameworks
    /Library/Frameworks
    /opt
    [HKEY_LOCAL_MACHINE\\SOFTWARE\\Creative\ Labs\\OpenAL\ 1.1\ Software\ Development\ Kit\\1.00.0000;InstallDir]
  PATH_SUFFIXES libx32 lib64 lib libs64 libs
  )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  OpenAL
  REQUIRED_VARS OPENAL_LIBRARY OPENAL_INCLUDE_DIR
  VERSION_VAR OPENAL_VERSION_STRING
  )

mark_as_advanced(OPENAL_LIBRARY OPENAL_INCLUDE_DIR)

if(OPENAL_INCLUDE_DIR AND OPENAL_LIBRARY)
  if(NOT TARGET OpenAL::OpenAL)
    if(EXISTS "${OPENAL_LIBRARY}")
      if(WIN32)
        get_filename_component(OPENAL_PATH ${OPENAL_LIBRARY} DIRECTORY)
        add_library(OpenAL::OpenAL SHARED IMPORTED)
        if (ARCH_x86)
          set(DLL_PATH "${OPENAL_PATH}/../../bin/x86/openal32.dll")
        elseif(ARCH_X64)
          set(DLL_PATH "${OPENAL_PATH}/../../bin/x64/openal32.dll")
        else()
          set(DLL_PATH "${OPENAL_PATH}/../../bin/ARM64/openal32.dll")
        endif()
        set_target_properties(OpenAL::OpenAL PROPERTIES
          IMPORTED_LOCATION "${DLL_PATH}"
          IMPORTED_IMPLIB "${OPENAL_LIBRARY}")
      elseif(APPLE)
          find_file(OPENAL_FULL_PATH OpenAL OpenAL.tbd PATHS ${OPENAL_LIBRARY} REQUIRED)
          add_library(OpenAL::OpenAL SHARED IMPORTED)
          set_target_properties(OpenAL::OpenAL PROPERTIES
            IMPORTED_LOCATION "${OPENAL_FULL_PATH}")
      else()
        add_library(OpenAL::OpenAL UNKNOWN IMPORTED)
        set_target_properties(OpenAL::OpenAL PROPERTIES
          IMPORTED_LOCATION "${OPENAL_LIBRARY}")
      endif()
    else()
      add_library(OpenAL::OpenAL INTERFACE IMPORTED)
      set_target_properties(OpenAL::OpenAL PROPERTIES
        IMPORTED_LIBNAME "${OPENAL_LIBRARY}")
    endif()
    set_target_properties(OpenAL::OpenAL PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${OPENAL_INCLUDE_DIR}")
  endif()
endif()