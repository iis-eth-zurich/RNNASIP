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

    np.random.seed(54)

    if not os.path.exists('../sweep_config.h'):
        os.mknod('../sweep_config.h')

    header_file = open("sweep_config.h", "w")
    nr_inp = (sys.argv[1])
    nr_out = (sys.argv[2])
    nr_cores = (sys.argv[3])
    model = (sys.argv[4])
    print("nr_inp ", nr_inp, " nr_out ", nr_out, " nr_cores ", nr_cores, " model ", model)

    INT_MAX = np.power(2,15)
    INT_MIN = -np.power(2,15)


    if model=="lstm":

        ###
        ### WRITE SWEEP_CONFIG.H
        ###

        ## WRITE COPYRIGHT
        s = "// Copyright (c) 2019-2020 ETH Zurich\n\
// Licensed under the Apache License, Version 2.0 (the \"License\")\n"

        header_file.write(s) 
        s = "#ifndef BENCHMARK_HEADER_FILE \n#define BENCHMARK_HEADER_FILE\n\n"
        header_file.write(s) 

        ## WRITE INCLUDES
        s = "#include \"general.h\"\n"
        header_file.write(s)
        s = "#include \"config.h\"\n"
        header_file.write(s)
        s = "#include \"config_profiling.h\"\n\n"
        header_file.write(s)


        ## WRITE DIMENSIONS
        s = "#define N_INP " + str(nr_inp) + "\n"
        header_file.write(s) 
        s = "#define N_OUT " + str(nr_out) + "\n"
        header_file.write(s) 
        s= "\n"
        header_file.write(s) 


        ## WRITE INPUT
        X = np.random.randint(INT_MIN, INT_MAX, size=int(nr_inp))
        x_data = np.array2string(X, precision=0, separator=',')[1:-1]
        x_data =  x_data.replace('\n', '')
        s = "L2_DATA data_t m_In[" + str(nr_inp) + "] = {" + x_data + " }; \n"
        header_file.write(s) 


        ## WRITE C
        C = np.random.randint(INT_MIN, INT_MAX, size=int(nr_out))
        x_data = np.array2string(C, precision=0, separator=',')[1:-1]
        x_data =  x_data.replace('\n', '')
        s = "L2_DATA data_t m_lstm_h[" + str(nr_out) + "] = {" + x_data + " }; \n"
        header_file.write(s) 


        ## WRITE H
        H = np.random.randint(INT_MIN, INT_MAX, size=int(nr_out))
        x_data = np.array2string(H, precision=0, separator=',')[1:-1]
        x_data =  x_data.replace('\n', '')
        s = "L2_DATA data_t m_lstm_c[" + str(nr_out) + "] = {" + x_data + " }; \n"
        header_file.write(s) 

        s= "\n"
        header_file.write(s) 

        ## WRITE W_IH
        WIH = np.random.randint(INT_MIN, INT_MAX, size=(4*int(nr_out), int(nr_inp)))
        x_data = np.array2string(WIH, precision=0, separator=',')[1:-1]
        x_data = x_data.replace('[', '{')
        x_data =  x_data.replace(']', '}')
        x_data =  x_data.replace('\n', '')
        s = "L2_DATA data_t m_lstm_weight_ih[" + str(4*int(nr_out)) + "][" + str(nr_inp) + "] = {" + x_data + " }; \n"
        header_file.write(s) 


        ## WRITE W_HH
        WHH = np.random.randint(INT_MIN, INT_MAX, size=(4*int(nr_out), int(nr_out)))
        x_data = np.array2string(WHH, precision=0, separator=',')[1:-1]
        x_data = x_data.replace('[', '{')
        x_data =  x_data.replace(']', '}')
        x_data =  x_data.replace('\n', '')
        s = "L2_DATA data_t m_lstm_weight_hh[" + str(4*int(nr_out)) + "][" + str(nr_out) + "] = {" + x_data + " }; \n"
        header_file.write(s) 

        s= "\n"
        header_file.write(s) 

        ## WRITE B_IH
        BIH = np.random.randint(INT_MIN, INT_MAX, size=4*int(nr_out))
        x_data = np.array2string(BIH, precision=0, separator=',')[1:-1]
        x_data = x_data.replace('[', '{')
        x_data =  x_data.replace(']', '}')
        x_data =  x_data.replace('\n', '')
        s = "L2_DATA data_t m_lstm_bias_ih[" + str(4*int(nr_out)) + "] = {" + x_data + " }; \n"
        header_file.write(s) 

        ## WRITE B_HH
        BHH = np.random.randint(INT_MIN, INT_MAX, size=4*int(nr_out))
        x_data = np.array2string(BHH, precision=0, separator=',')[1:-1]
        x_data = x_data.replace('[', '{')
        x_data =  x_data.replace(']', '}')
        x_data =  x_data.replace('\n', '')
        s = "L2_DATA data_t m_lstm_bias_hh[" + str(4*int(nr_out)) + "] = {" + x_data + " }; \n"
        header_file.write(s) 

        s = "\n#endif \n"
        header_file.write(s) 

        header_file.close()


        ###
        ### WRITE CONFIG_PROFILING.H
        ###
        if not os.path.exists('../config_profiling.h'):
            os.mknod('../config_profiling.h')

        header_file = open("config_profiling.h", "w")


        s="                                                                                        \n \
/** Copyright (c) 2019 ETH Zurich, Integrated System Laboratory, Renzo Andri, Gianna Paulin    \n \
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
#define LSTM_ON 1                                                                              \n \
#define SWEEP                                                                                  \n \
                                                                                               \n \
                                                                                               \n \
#ifdef MODEL0                                                                                  \n \
#define LSTM_ON 1                                                                              \n \
#endif                                                                                         \n \
#ifdef MODEL1                                                                                  \n \
#define LSTM_ON 1                                                                              \n \
#endif                                                                                         \n \
                                                                                               \n \
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





    else:

        ###
        ### WRITE SWEEP_CONFIG.H
        ###

        ## WRITE COPYRIGHT
        s = "// Copyright (c) 2019 ETH Zurich, Integrated System Laboratory, Renzo Andri, Gianna Paulin\n"
        header_file.write(s) 
        s = "#ifndef BENCHMARK_HEADER_FILE \n#define BENCHMARK_HEADER_FILE\n\n"
        header_file.write(s) 

        ## WRITE INCLUDES
        s = "#include \"general.h\"\n"
        header_file.write(s)
        s = "#include \"config.h\"\n"
        header_file.write(s)
        s = "#include \"config_profiling.h\"\n\n"
        header_file.write(s)


        ## WRITE DIMENSIONS
        s = "#define N_INP " + str(nr_inp) + "\n"
        header_file.write(s) 
        s = "#define N_OUT " + str(nr_out) + "\n"
        header_file.write(s) 
        s= "\n"
        header_file.write(s) 


        ## WRITE INPUT
        X = np.random.randint(INT_MIN, INT_MAX, size=int(nr_inp))
        x_data = np.array2string(X, precision=0, separator=',')[1:-1]
        x_data =  x_data.replace('\n', '')
        s = "L2_DATA data_t m_In[" + str(nr_inp) + "] = {" + x_data + " }; \n"
        header_file.write(s) 

        s= "\n"
        header_file.write(s) 

        ## WRITE W_IH
        WIH = np.random.randint(INT_MIN, INT_MAX, size=(int(nr_out), int(nr_inp)))
        x_data = np.array2string(WIH, precision=0, separator=',')[1:-1]
        x_data = x_data.replace('[', '{')
        x_data =  x_data.replace(']', '}')
        x_data =  x_data.replace('\n', '')
        s = "L2_DATA data_t m_linear_Weights[" + str(int(nr_out)) + "][" + str(nr_inp) + "] = {" + x_data + " }; \n"
        header_file.write(s) 


        s= "\n"
        header_file.write(s) 

        ## WRITE B_IH
        BIH = np.random.randint(INT_MIN, INT_MAX, size=int(nr_out))
        x_data = np.array2string(BIH, precision=0, separator=',')[1:-1]
        x_data = x_data.replace('[', '{')
        x_data =  x_data.replace(']', '}')
        x_data =  x_data.replace('\n', '')
        s = "L2_DATA data_t m_linear_Bias[" + str(int(nr_out)) + "] = {" + x_data + " }; \n"
        header_file.write(s) 

        s = "\n#endif \n"
        header_file.write(s) 

        header_file.close()


        ###
        ### WRITE CONFIG_PROFILING.H
        ###
        if not os.path.exists('../config_profiling.h'):
            os.mknod('../config_profiling.h')

        header_file = open("config_profiling.h", "w")


        s="                                                                                        \n \
/** Copyright (c) 2019 ETH Zurich, Integrated System Laboratory, Renzo Andri, Gianna Paulin    \n \
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
//#define LSTM_ON 1                                                                              \n \
#define SWEEP                                                                                  \n \
                                                                                               \n \
                                                                                               \n \
#ifdef MODEL0                                                                                  \n \
#define LSTM_ON 1                                                                              \n \
#endif                                                                                         \n \
#ifdef MODEL1                                                                                  \n \
#define LSTM_ON 1                                                                              \n \
#endif                                                                                         \n \
                                                                                               \n \
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

