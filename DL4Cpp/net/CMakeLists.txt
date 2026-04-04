add_executable(unet_graph graph.cpp)
target_link_libraries(unet_graph PRIVATE DL4Cpp)

add_custom_command(TARGET unet_graph POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/unet.pnnx.param
        $<TARGET_FILE_DIR:unet_graph>/unet.pnnx.param
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/unet.pnnx.bin
        $<TARGET_FILE_DIR:unet_graph>/unet.pnnx.bin
    COMMENT "Copying UNet model files to output directory"
)
