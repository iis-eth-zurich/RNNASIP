/** @file config_profiling.h
 *  @brief General configurations for the Model selection
 *
 *  This file can be used to select between different models and the multi core / single core
 *  implementations.
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

// #define MODEL0
 // #define MODEL1
 #define MODEL2
 // #define MODEL3
 // #define MODEL5
 // #define MODEL6
 // #define MODEL7
 // #define MODEL8
 // #define MODEL9
 // #define MODEL10
 // #define MODEL11

 // #define LSTM_ON 1
 // #define SWEEP


 #ifdef MODEL0
 #define LSTM_ON 1
 #endif
 #ifdef MODEL1
 #define LSTM_ON 1
 #endif

 #define OUTPUTBUFFER 8
 #define FMINTILING
 #define FMOUTTILING
 #define W_OFFSET 0

 // #define SINGLECORE
 #define MULTICORE
 // #define NR_CORES 16
 #define NR_CORES 1
 // #define NR_CORES 2
 // #define NR_CORES 4
 // #define NR_CORES 8
 // #define NR_CORES 16

 // #define ACT_OFFSET
