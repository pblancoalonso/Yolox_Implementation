project(detector)

find_package(InferenceEngine REQUIRED)
find_package(OpenCV REQUIRED)

add_library(${PROJECT_NAME} source/Detector.cpp)

add_library(lib::detector ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} 
    PUBLIC 
        include/
        )

target_link_libraries(${PROJECT_NAME} PUBLIC
    ${InferenceEngine_LIBRARIES}
    ${OpenCV_LIBS})