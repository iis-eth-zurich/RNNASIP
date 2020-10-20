/** @file config.h
 *  @brief General configurations for the SW optimizations
 *
 *  This file can be used to select between different implementations for the extended RISC-Y core,
 *  have a look into the tzscale/Basic_Kernels/config.h for the corresponding configurations for the
 *  tzscale implementations.
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

#ifndef ASIP

/// Fixed-point implementation
#define FixedPt 1

/// Use SIMD instructions
#define SIMD

/// Prints for Debug active
// #define DEBUG
// #define DEBUG_LSTM

/// Print out results
#define PRINTF_ACTIVE

/// Output FM tiling

/// Use lw+incr+sdotp VLIW extension
#define VLIWEXT

/// Use manual loop unfolding
#define MANUALLOOPUNFOLDING

/// Do activation on the fly
#define DOACTONTHEFLY
/// On RISC-Y use TANH and sigmoid extension
#define PULP_USETANHSIG

/// use DMA for data copying
#define DMA
// #define TILING

#define LSTM_OPT
#define LSTM_HIGH_OPT

// #define BATCHING 1
// #define TILING_HARD

#define PREFETCH_ICACHE

/// activate old rt
// #define PROFILING

/// activate new runtime
#define PROFILING_NEW
#define TIMER

/// profiling level: amdahls law seriell code
// #define PROFILING_LINEAR_AMDAHL_SERIELL
/// profiling level: amdahls law parallel code
// #define PROFILING_LINEAR_AMDAHL_PARALLEL

/// profiling level: amdahls law seriell code
// #define PROFILING_LSTM_AMDAHL_SERIELL
/// profiling level: amdahls law parallel code
// #define PROFILING_LSTM_AMDAHL_PARALLEL

// #define PROFILING_TILING
//#define PROFILING_EFFICIENT_TILING
/// define the profiling level
#define PROFILING_ALL
// #define PROFILING_LINEAR
// #define PROFILING_LSTM
// #define PROFILING_TWOLINEAR
// #define PROFILING_TANH
// #define PROFILING_SIG
// #define PROFILING_COPY
// #define PROFILING_HADM
// #define PROFILING_ADDT

#endif
