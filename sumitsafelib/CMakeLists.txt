# Define the sumitsafelib library
add_library(sumitsafelib SHARED
    src/sumitsafelib.cpp

)

# Include header directories, including paths from the root
target_include_directories(sumitsafelib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>  
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ggml/include>  
    $<INSTALL_INTERFACE:include>
)

# Link the whisper library
target_link_libraries(sumitsafelib PRIVATE whisper)

# Use C++17
set_target_properties(sumitsafelib PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

# Specify the output name for the .dylib
set_target_properties(sumitsafelib PROPERTIES
    OUTPUT_NAME "sumitsafe"
    VERSION 1.0
    SOVERSION 1
)

# Installation instructions
install(TARGETS sumitsafelib
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
