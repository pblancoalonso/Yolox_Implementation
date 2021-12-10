project(config-parser)

add_library(${PROJECT_NAME} source/Parser.cpp)

add_library(lib::config-parser ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} 
    PRIVATE 
        include/
        )