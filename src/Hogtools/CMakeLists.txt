cmake_minimum_required(VERSION 2.8)

set(FHOG_LIB_SRC gradientMex.cpp gradientMex.h sse.hpp wrappers.hpp)

add_library(features STATIC ${FHOG_LIB_SRC})
target_link_libraries(features ${OpenCV_LIBS})
set_target_properties(features PROPERTIES VERSION 1.0.0 SOVERSION 1)