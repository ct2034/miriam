# If the pyx file is a C++ file, we should specify that here.
#set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/adamsmap.pyx PROPERTIES CYTHON_IS_CXX FALSE )
  
if(APPLE)
	include_directories(/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy/core/include/)
endif(APPLE)

# Multi-file cython modules do not appear to be working at the moment.
cython_add_module(adamsmap adamsmap.pyx)
cython_add_module(eval_disc eval_disc.pyx)

if(${adamsmap_ROS})
	set_target_properties(adamsmap PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION})
	set_target_properties(eval_disc PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION})
	install(TARGETS adamsmap eval_disc
	  ARCHIVE DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
	  LIBRARY DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
	)
endif(${adamsmap_ROS})