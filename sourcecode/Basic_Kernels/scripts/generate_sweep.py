#!/usr/bin/env python3
#*----------------------------------------------------------------------------*
#* Copyright (C) 2019-2020 ETH Zurich, Switzerland                            *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");            *
#* you may not use this file except in compliance with the License.           *
#* You may obtain a copy of the License at                                    *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                 *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software        *
#* distributed under the License is distributed on an "AS IS" BASIS,          *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
#* See the License for the specific language governing permissions and        *
#* limitations under the License.                                             *
#*                                                                            *
#* Authors:  Gianna Paulin                                                    *
#*----------------------------------------------------------------------------*

import sys
import csv
from collections import OrderedDict
import numpy as np
import os
import string

np.set_printoptions(threshold=sys.maxsize)

if __name__ == '__main__':

    ## Add Seed
    np.random.seed(54)

    ## get arguments
    model = (sys.argv[1])
    nr_cores = (sys.argv[2])
    what = (sys.argv[3])        # "timer" or "all"
    difficulty = (sys.argv[4])  # "baseline" or "opt"
    batch = (sys.argv[5])  # "batch nr" or "opt"

    # print("model ", model, " nr_cores ", nr_cores, " what ", what, " difficulty ", difficulty)

    # INT_MAX = np.power(2,15)
    # INT_MIN = -np.power(2,15)


    # print("WRITE CONFIG_PROFILING.H")

    ###
    ### WRITE CONFIG_PROFILING.H
    ###
    if not os.path.exists('../config_profiling.h'):
        os.mknod('../config_profiling.h')

    header_file = open("config_profiling.h", "w")


    s="                                                                                        \n \
/** Copyright (c) 2019-2020 ETH Zurich                                                         \n \
 * Licensed under the Apache License, Version 2.0 (the \"License\")                            \n \
 *  @file config_profiling.h                                                                   \n \
 *  @brief General configurations for the Model selection                                      \n \
 *                                                                                             \n \
 *  This file can be used to select between different models and the multi core / single core  \n \
 *  implementations.                                                                           \n \
 *                                                                                             \n \
 * @author Renzo Andri (andrire)                                                               \n \
 * @author Gianna Paulin (pauling)                                                             \n \
 */                                                                                            \n \
                                                                                               \n \
"
    header_file.write(s) 

    s="#define MODEL"+str(model)
    header_file.write(s) 

    s=" \
// #define MODEL0                                                                              \n \
// #define MODEL1                                                                              \n \
// #define MODEL2                                                                              \n \
// #define MODEL3                                                                              \n \
// #define MODEL5                                                                              \n \
// #define MODEL6                                                                              \n \
// #define MODEL7                                                                              \n \
// #define MODEL8                                                                              \n \
// #define MODEL9                                                                              \n \
// #define MODEL10                                                                             \n \
// #define MODEL11                                                                             \n \
                                                                                               \n \
// #define LSTM_ON 1                                                                           \n \
// #define SWEEP                                                                               \n \
                                                                                               \n \
                                                                                               \n \
#ifdef MODEL0                                                                                  \n \
#define LSTM_ON 1                                                                              \n \
#endif                                                                                         \n \
#ifdef MODEL1                                                                                  \n \
#define LSTM_ON 1                                                                              \n \
#endif                                                                                         \n \
"
    header_file.write(s) 


    if difficulty=="baseline":
        s="                                                                                    \n \
#define OUTPUTBUFFER 1                                                                         \n \
// #define FMINTILING                                                                          \n \
// #define FMOUTTILING                                                                         \n \
#define W_OFFSET 0                                                                             \n \
                                                                                               \n \
#define MULTICORE                                                                              \n \
"

    else:
        s="                                                                                    \n \
#define OUTPUTBUFFER 8                                                                         \n \
#define FMINTILING                                                                             \n \
#define FMOUTTILING                                                                            \n \
#define W_OFFSET 0                                                                             \n \
                                                                                               \n \
#define MULTICORE                                                                              \n \
"

    header_file.write(s) 

    s="#define NR_CORES " + str(nr_cores)
    header_file.write(s) 

    s="                                                                                        \n \
// #define NR_CORES 1                                                                          \n \
// #define NR_CORES 2                                                                          \n \
// #define NR_CORES 4                                                                          \n \
// #define NR_CORES 8                                                                          \n \
// #define NR_CORES 16                                                                         \n \
                                                                                               \n \
// #define ACT_OFFSET                                                                          \n \
"
    header_file.write(s) 


    # print("WRITE CONFIG_PROFILING.H - DONE")


    # print("WRITE CONFIG.H - DONE")


    ###
    ### WRITE CONFIG.H
    ###
    if not os.path.exists('../config.h'):
        os.mknod('../config.h')

    header_file = open("config.h", "w")



    s=" \
                                                                                                        \n\
/** Copyright (c) 2019 ETH Zurich, Integrated System Laboratory, Renzo Andri, Gianna Paulin             \n\
 *  @file config.h                                                                                      \n\
 *  @brief General configurations for the SW optimizations                                              \n\
 *                                                                                                      \n\
 *  This file can be used to select between different implementations for the extended RISC-Y core,     \n\
 *  have a look into the tzscale/Basic_Kernels/config.h for the corresponding configurations for the    \n\
 *  tzscale implementations.                                                                            \n\
 *                                                                                                      \n\
 * @author Renzo Andri (andrire)                                                                        \n\
 * @author Gianna Paulin (pauling)                                                                      \n\
 */                                                                                                     \n\
                                                                                                        \n\
#ifndef ASIP                                                                                            \n\
                                                                                                        \n\
/// Fixed-point implementation                                                                          \n\
#define FixedPt 1                                                                                       \n\
                                                                                                        \n\
/// Use SIMD instructions                                                                               \n\
#define SIMD                                                                                            \n\
                                                                                                        \n\
/// Prints for Debug active                                                                             \n\
// #define DEBUG                                                                                        \n\
// #define DEBUG_LSTM                                                                                   \n\
                                                                                                        \n\
/// Print out results                                                                                   \n\
// #define PRINTF_ACTIVE                                                                                \n\
                                                                                                        \n\
/// Output FM tiling                                                                                    \n\
                                                                                                        \n\
/// Use lw+incr+sdotp VLIW extension                                                                    \n\
"
    header_file.write(s) 

    if difficulty=="baseline":
        s="// #define VLIWEXT \n"
    else:
        s="#define VLIWEXT \n"

    header_file.write(s) 

    s="                                                                                                            \n\
/// Use manual loop unfolding                                                                           \n\
#define MANUALLOOPUNFOLDING                                                                             \n\
                                                                                                        \n\
/// Do activation on the fly                                                                            \n\
#define DOACTONTHEFLY                                                                                   \n\
/// On RISC-Y use TANH and sigmoid extension                                                            \n\
"
    header_file.write(s) 

    if difficulty=="baseline":
        s="// #define PULP_USETANHSIG \n"
    else:
        s="#define PULP_USETANHSIG \n"
    header_file.write(s) 

    s="                                                                                                            \n\
/// use DMA for data copying                                                                            \n\
#define DMA                                                                                             \n\
// #define TILING                                                                                       \n\
                                                                                                        \n\
#define LSTM_OPT                                                                                        \n\
#define LSTM_HIGH_OPT                                                                                   \n\
\n\
"
    header_file.write(s) 

    if int(batch) > 0:
        s = "#define BATCHING " + str(batch)
        header_file.write(s) 

    s="                                                                                                         \n\
#define TILING_HARD                                                                                     \n\
                                                                                                        \n\
#define PREFETCH_ICACHE                                                                                 \n\
                                                                                                        \n\
/// activate old rt                                                                                     \n\
// #define PROFILING                                                                                    \n\
                                                                                                        \n\
/// activate new runtime                                                                                \n\
#define PROFILING_NEW                                                                                   \n\
"
    header_file.write(s) 

    if what=="all":
        s="// #define TIMER \n"
    else:
        s="#define TIMER \n"
    header_file.write(s) 


    s="                                                                                                 \n\
/// profiling level: amdahls law seriell code                                                           \n\
// #define PROFILING_LINEAR_AMDAHL_SERIELL                                                              \n\
/// profiling level: amdahls law parallel code                                                          \n\
// #define PROFILING_LINEAR_AMDAHL_PARALLEL                                                             \n\
                                                                                                        \n\
/// profiling level: amdahls law seriell code                                                           \n\
// #define PROFILING_LSTM_AMDAHL_SERIELL                                                                \n\
/// profiling level: amdahls law parallel code                                                          \n\
// #define PROFILING_LSTM_AMDAHL_PARALLEL                                                               \n\
                                                                                                        \n\
// #define PROFILING_TILING                                                                             \n\
//#define PROFILING_EFFICIENT_TILING                                                                    \n\
/// define the profiling level                                                                          \n\
#define PROFILING_ALL                                                                                   \n\
// #define PROFILING_LINEAR                                                                             \n\
// #define PROFILING_LSTM                                                                               \n\
// #define PROFILING_TWOLINEAR                                                                          \n\
// #define PROFILING_TANH                                                                               \n\
// #define PROFILING_SIG                                                                                \n\
// #define PROFILING_COPY                                                                               \n\
// #define PROFILING_HADM                                                                               \n\
// #define PROFILING_ADDT                                                                               \n\
                                                                                                        \n\
#endif                                                                                                  \n\
                                                                                                        \n\
"
    header_file.write(s) 

    # print("DONE config generation")