# edgeAI-MachineSense Phase 1 firmware: TFLite-Micro autoencoder inference over UART
# (replay mode -- no sensor required, see main/README or ../README.md).
cmake_minimum_required(VERSION 3.16)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(machinesense_replay)
