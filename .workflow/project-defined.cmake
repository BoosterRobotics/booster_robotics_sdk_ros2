macro(closure_import_prepare_project)
endmacro()


macro(closure_import_install_project)
    if (DEFINED ARGV0 AND NOT "${ARGV0}" STREQUAL "")
        cmake_language(CALL ${ARGV0})
    endif ()
endmacro()


macro(closure_link_target_exe i_exe_target)
endmacro()


macro(closure_link_target_lib i_lib_target)
endmacro()
