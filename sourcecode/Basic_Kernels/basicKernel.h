/** @file basicKernel.h
 *  @brief Header File for basic ML kernels (including FC, LSTM, Conv2D for the RNN ASIP
 * 
 *  Header of basicKernel.c C implemenentation implementing several levels of opitmizations for the RISC-Y extension and the tzscale extension
 * 
 * @author Renzo Andri (andrire)
 * @author Gianna Paulin (pauling)
 *
 *----------------------------------------------------------------------------*
 * Copyright (C) 2019-2020 ETH Zurich, Switzerland                            *
 * SPDX-License-Identifier: Apache-2.0                                        *
 *                                                                            *
 * Licensed under the Apache License, Version 2.0 (the "License");            *
 * you may not use this file except in compliance with the License.           *
 * You may obtain a copy of the License at                                    *
 *                                                                            *
 * http://www.apache.org/licenses/LICENSE-2.0                                 *
 *                                                                            *
 * Unless required by applicable law or agreed to in writing, software        *
 * distributed under the License is distributed on an "AS IS" BASIS,          *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
 * See the License for the specific language governing permissions and        *
 * limitations under the License.                                             *
 *----------------------------------------------------------------------------*
 */

#ifndef BASIC_KERNEL_HEADER_FILE
#define BASIC_KERNEL_HEADER_FILE

#include <config.h>
#include <config_profiling.h>
#ifndef ASIP
    #include "pulp.h"
    // #include "rt/rt_api.h"
    #include <math.h>
#endif

#include "general.h"

// addresses for enabling ICACHE prefetch for Marsellus
#define ICACHE_CTRL_UNIT 0x10201400
#define ICACHE_PREFETCH ICACHE_CTRL_UNIT + 0x18


// #include "general.h"
#ifdef SINGLECORE
    #include "basicKernel_sc.h"
#else
    #include "basicKernel_mc.h"
#endif

data_t generic_tanh(data_t value);
data_t generic_sig(data_t x);
extern double tanh(double value);



/** @brief requantizing/shifting result and run activation-on-the-fly
 *
 *  requantizing/shifting result and run activation-on-the-fly in case DOACTONTHEFLY is set
 *
 *  @param value Array of concecutive layers of the current neural network
 *  @param activationFunction activation function to be applied (ACT_NONE, ACT_TANH, ACT_SIG)
 *  @return Output quantized and activated result
 */
inline int shiftAndAct(int value, int activationFunction) {
    int temp;
#ifdef DOACTONTHEFLY
    temp = value>>(q_fraqP1); // TODO merging shifting and tanh/sigmoid instruction
    switch(activationFunction) {
        case ACT_NONE: return temp; break;
        case ACT_TANH: return generic_tanh(temp); break;
        case ACT_SIG:  return generic_sig(temp); break;
    }
#else
    return value>>(q_fraqP1);
#endif
}


data_t * NOINLINE inferNetwork(
    struct layer * network,
    int depth,
    data_t * __restrict__ inFeatures,
    data_t * __restrict__ buffer
);

#endif // BASIC_KERNEL_HEADER_FILE
