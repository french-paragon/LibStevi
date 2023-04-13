 add_subdirectory(io_img_and_disp)
 add_subdirectory(intrisic_image_decomposition)
 add_subdirectory(checkboard_detection)
 add_subdirectory(bg_fg_stereo)

 if (buildGui)
     add_subdirectory(stevimg_viewer)
 endif(buildGui)
