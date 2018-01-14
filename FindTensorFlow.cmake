include(FindPackageHandleStandardArgs)
unset(TENSORFLOW_FOUND)

execute_process(
    COMMAND python -c "import tensorflow as tf; print 'tf_includepath',tf.sysconfig.get_include()"
    OUTPUT_VARIABLE TF_INC
    ERROR_VARIABLE TF_INC
    RESULT_VARIABLE TF_INC_OK
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REGEX MATCH "tf_includepath (.*)" _ ${TF_INC})
set(TF_INC ${CMAKE_MATCH_1})


execute_process(
    COMMAND python -c "import tensorflow as tf; print 'tf_libpath',tf.sysconfig.get_lib()"
    OUTPUT_VARIABLE TF_LIB
    ERROR_VARIABLE TF_LIB
    RESULT_VARIABLE TF_LIB_OK
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REGEX MATCH "tf_libpath (.*)" _ ${TF_LIB})
set(TF_LIB ${CMAKE_MATCH_1})

message(STATUS ${TF_INC})
message(STATUS ${TF_LIB})
if (${TF_INC_OK} EQUAL 0 AND ${TF_LIB_OK} EQUAL 0)
    find_path(TensorFlow_INCLUDE_DIR
        NAMES tensorflow/core/framework/op.h
        PATHS ${TF_INC}
        NO_DEFAULT_PATH
    )
    find_library(TensorFlow_LIBRARY 
            NAMES tensorflow_framework
            PATHS ${TF_LIB}
            NO_DEFAULT_PATH
    )
endif (${TF_INC_OK} EQUAL 0 AND ${TF_LIB_OK} EQUAL 0)

# set TensorFlow_FOUND
find_package_handle_standard_args(TensorFlow DEFAULT_MSG TensorFlow_INCLUDE_DIR TensorFlow_LIBRARY)

# set external variables for usage in CMakeLists.txt
if(TENSORFLOW_FOUND)
    set(TensorFlow_LIBRARIES ${TensorFlow_LIBRARY})
    set(TensorFlow_INCLUDE_DIRS ${TensorFlow_INCLUDE_DIR} ${TensorFlow_INCLUDE_DIR}/external/nsync/public) #fix: https://github.com/sadeepj/crfasrnn_keras/issues/19
endif()

# hide locals from GUI
mark_as_advanced(TensorFlow_INCLUDE_DIR TensorFlow_LIBRARY)
