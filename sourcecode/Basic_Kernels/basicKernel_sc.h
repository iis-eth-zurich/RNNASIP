/** @file LinLayer_mc.h
 *  @brief Header File for basic ML kernels (including FC, LSTM, Conv2D for the RNN ASIP
 * 
 *  Header of LinLayer_sc.c C implemenentation implementing several levels of opitmizations for the RISC-Y extension and the tzscale extension
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

#ifndef BASICKERNEL_SC_HEADER_FILE
#define BASICKERNEL_SC_HEADER_FILE

#include <config.h>
#include <config_profiling.h>
#ifndef ASIP
    #include "pulp.h"
    // #include "rt/rt_api.h"
    #include <math.h>
#endif
#include "general.h"
#include "basicKernel.h"

#ifdef EFFICIENT_CORE_ASSIGNMENT

    void NOINLINE LinearLayer (
        // Layer Attributes
        int inFeaturesSize,
        int outFeaturesSize,
        int tile_size,
        short hasBias,
        // Layer Parameters
        data_t * __restrict__ weight,
        data_t * __restrict__ bias,
        // Input and Output Features
        data_t * __restrict__ inFeatures,
        data_t * __restrict__ outFeatures
    ); //property(functional);

#else

void NOINLINE LinearLayer (
    // Layer Attributes
    int inFeaturesSize,
    int outFeaturesSize,
    short hasBias,
    // Layer Parameters
    data_t * __restrict__ weight,
    data_t * __restrict__ bias,
    // Input and Output Features
    data_t * __restrict__ inFeatures,
    data_t * __restrict__ outFeatures
); //property(functional);

#endif


void NOINLINE TwoLinearLayersAccumulate (
    // Layer Attributes
    int inFeaturesSize1, int inFeaturesSize2, int outFeaturesSize, 
    int activationFunction,
    // Layer Parameters
    data_t * __restrict__ weight1,
    data_t * __restrict__ weight2,
    data_t * __restrict__ bias1,
    data_t * __restrict__ bias2,
    // Input and Output Features
    data_t * __restrict__ inFeatures1,
    data_t * __restrict__ inFeatures2,
    data_t * __restrict__ outFeatures);


void NOINLINE RNNLayer (
    // Layer Attributes
    int inFeaturesSize, int hiddenFeaturesSize,
    // Layer Parameters
    data_t * __restrict__ weight_ih_l,
    data_t * __restrict__ weight_hh_l,
    data_t * __restrict__ bias_ih_l,
    data_t * __restrict__ bias_hh_l,
    // Input and Output Features
    data_t * __restrict__ inFeatures,
    data_t * __restrict__ outFeatures, // out and hidden
    // Hidden Features
    data_t * __restrict__ hiddenFeatures);


void NOINLINE LSTMLayer (
    // Layer Attributes
    int inFeaturesSize, int hiddenFeaturesSize,
    // Layer Parameters
    data_t * __restrict__ weight_ih_l,
    data_t * __restrict__ weight_hh_l,
    data_t * __restrict__ bias_ih_l,
    data_t * __restrict__ bias_hh_l,
    // Input and Output Features
    data_t * __restrict__ inFeatures,
    // data_t * __restrict__ outFeatures, // out and hidden
    // Hidden Features
    data_t * __restrict__ lstm_h,
    data_t * __restrict__ lstm_c,
    // intermediate nodes
    data_t * __restrict__ lstm_h_out,
    data_t * __restrict__ lstm_f,
    data_t * __restrict__ lstm_i,
    data_t * __restrict__ lstm_g,
    data_t * __restrict__ lstm_o);


int NOINLINE Conv2dLayer (
    // Layer Attributes
    struct layer * _layer,
    int h_im,
    int w_im,
    data_t * __restrict__ inFeatures,
    data_t * __restrict__ outFeatures);

void NOINLINE SigLayer (
    // Layer Attributes
    int TensorSize,
    data_t * __restrict__ Features);

void NOINLINE AddTensor (
        // Layer Attributes
    int TensorSize,
        // Layer Parameters
    data_t * __restrict__ FeaturesA,
    data_t * __restrict__ FeaturesB);

// A*=B
void NOINLINE HadMulTensor (
        // Layer Attributes
    int TensorSize,
        // Layer Parameters
    data_t * __restrict__ FeaturesA,
    data_t * __restrict__ FeaturesB);
// A=B
void NOINLINE CopyTensor (
        // Layer Attributes
    int TensorSize,
        // Layer Parameters
    data_t * __restrict__ FeaturesA,
    data_t * __restrict__ FeaturesB);

void NOINLINE fillTensor (
  // Layer Attributes
    int TensorSize,
    data_t * __restrict__ Tensor,
    data_t fillValue);

void PrintTensor2D (
        // Layer Attributes
    int dim1, int dim2,
    data_t * __restrict__ dataArray
    );
void PrintTensor (
        // Layer Attributes
    int dim1,
    data_t * __restrict__ dataArray
    );

void error2D (
        // Layer Attributes
    int dim1, int dim2,
    data_t * __restrict__ dataArray,
    data_t * __restrict__ data2Array,
    data_t * __restrict__ error
    );

data_t PrintTensorDiff (
        // Layer Attributes
    int dim1,
    data_t * __restrict__ dataArray,
    data_t * __restrict__ data2Array
    );

data_t PrintTensorDiff2D (
        // Layer Attributes
    int dim1, int dim2,
    data_t * __restrict__ dataArray,
    data_t * __restrict__ data2Array
    );

void printFloat(data_t value);

struct network
{
    int test;
};

#endif // BASICKERNEL_SC_HEADER_FILE
