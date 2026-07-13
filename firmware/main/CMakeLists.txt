idf_component_register(
    SRCS "main.cc" "inference.cc" "model_data.cc"
    INCLUDE_DIRS "."
    REQUIRES esp-tflite-micro esp_driver_uart esp_driver_gpio esp_timer
)
