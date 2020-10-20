/** @file basicKernel_mc.c
 *  @brief Multi-core implementation of Basic ML kernels (including FC, LSTM, Conv2D for the RNN ASIP)
 *
 *  Basic ML kernels (including FC, LSTM, Conv2D for the RNN ASIP) implemented for multi-core
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


#include "basicKernel_mc.h"
#include <stdio.h>

/** \brief Length (in time) of RNN Sequence */
int rnn_seqSize=1; 
/** \brief Length (in time) of LSTM Sequence */
int lstm_seqSize=1;

/** \brief Piecewise Linear Approximation of tangent hyperbolic and sigmoid */
const int lut_numelements = 16;
const int lb_lut_numelements = 4;

#ifdef MULTICORE
/** \brief Piecewise Linear Approximation "m"-LUT of tanh */
__attribute__ ((section(".heapsram")))  short l1_lut_Tanh_m[16] = {4021, 3563, 2835, 2070, 1418, 929, 592, 370, 228, 140, 86, 52, 32, 19, 12, 7};
/** \brief Piecewise Linear Approximation "q"-LUT of tanh */
__attribute__ ((section(".heapsram")))  int   l1_lut_Tanh_q[16] = {17060, 512067, 2012407, 4361003, 7021506, 9510743, 11575189, 13158594, 14311861, 15123015, 15679911, 16055709, 16306104, 16471340, 16579558, 16650000};
/** \brief Piecewise Linear Approximation "m"-LUT of sigm */
__attribute__ ((section(".heapsram")))  short l1_lut_sig_m[16]  = {1019, 988, 930, 850, 758, 660, 563, 472, 391, 319, 258, 207, 165, 131, 104, 82};
/** \brief Piecewise Linear Approximation "q"-LUT of sigm */
__attribute__ ((section(".heapsram")))  int   l1_lut_sig_q[16]  = {8389671, 8423495, 8544906, 8789991, 9169470, 9670607, 10264318, 10914030, 11583389, 12241371, 12864661, 13437943, 13952921, 14406803, 14800713, 15138308};
#else

/** \brief Piecewise Linear Approximation "m"-LUT of tanh */
const short lut_Tanh_m[16] = {4021, 3563, 2835, 2070, 1418, 929, 592, 370, 228, 140, 86, 52, 32, 19, 12, 7};
/** \brief Piecewise Linear Approximation "q"-LUT of tanh */
const int lut_Tanh_q[16]   = {17060, 512067, 2012407, 4361003, 7021506, 9510743, 11575189, 13158594, 14311861, 15123015, 15679911, 16055709, 16306104, 16471340, 16579558, 16650000};
/** \brief Piecewise Linear Approximation "m"-LUT of sigm */
const short lut_sig_m[16]  = {1019, 988, 930, 850, 758, 660, 563, 472, 391, 319, 258, 207, 165, 131, 104, 82};
/** \brief Piecewise Linear Approximation "q"-LUT of sigm */
const int lut_sig_q[16]    = {8389671, 8423495, 8544906, 8789991, 9169470, 9670607, 10264318, 10914030, 11583389, 12241371, 12864661, 13437943, 13952921, 14406803, 14800713, 15138308};
#endif


//////////////////////////////////////////////////////////////////////////////////////////////
/** @brief Sigmoid Activation Function
 *
 *  @param value input varialbe
 *  @return sigmoid of the input variable
 */
inline data_t sig(data_t value) {
    data_t a = value;
    unsigned int lutsize = 16;
    unsigned int value1 = 4096;
    unsigned int value0p999 = 4095;
    int m;
    int q;
    int q_signed;
    unsigned short sign = a<0 ? 1 : 0;
    int tmp;
    int mac_result;
    int mac_result_signed;
    data_t abs_a;

    // get abs value
    if(sign==0x1) {
        abs_a = -a;
    } else {
        abs_a = (a);
    }

    tmp = abs_a>>(13-3);

    if(tmp>=lutsize) {
        return (sign) ? (data_t)0: (data_t)value1;
    }
    else {
#ifdef MULTICORE
        m = l1_lut_sig_m[tmp];
        q = l1_lut_sig_q[tmp];
#else
        m = lut_sig_m[tmp];
        q = lut_sig_q[tmp];
#endif
        mac_result        = (m*abs_a+q)>>12;
        mac_result_signed = (sign==1)? ~mac_result : mac_result;

        if(sign==1) {
            return (value0p999+(mac_result_signed)); // 1-(mx+q)=4096+(~mac_result+1)=4095+(~mac_result)
        }
        else {
            return mac_result_signed;
        }
    }
}


//////////////////////////////////////////////////////////////////////////////////////////////
/** @brief Tangent Hyperbolic
 *
 *  @param value input variable
 *  @return tangent hypberbolic of the input variable
 */
inline data_t Tanh(data_t value) {
    data_t x = value;
    unsigned int lutsize = 16;
    unsigned int value1 = 4096;
    unsigned int value0p999 = 4095;
    int m;
    int q;
    int q_signed;
    unsigned short sign = (x>>31) & 0x1;
    int id;
    int mac_result;
    int mac_result_signed;
    int abs_x;

    // get abs value
    if(sign==0x1) {
        abs_x = -x;
    } else {
        abs_x = x;
    }

    id = abs_x>>(13-3); // get index of LUT

    if(id>=lutsize) {
        return (sign==0x1)?(int)-value1: (int)value1;
    }
    else {
#ifdef MULTICORE
        m = l1_lut_Tanh_m[id];
        q = l1_lut_Tanh_q[id];
#else
        m = lut_Tanh_m[id];
        q = lut_Tanh_q[id];
#endif
        mac_result        = (m*abs_x+q)>>12;
        mac_result_signed = (sign==1)? ~mac_result : mac_result;
        return mac_result_signed;
    }
}




////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  _     _                         _                           ////////////////////////////////////////////////////////////////////////////////////////////////////
// | |   (_)_ __   ___  __ _ _ __  | |    __ _ _   _  ___ _ __  ////////////////////////////////////////////////////////////////////////////////////////////////////
// | |   | | '_ \ / _ \/ _` | '__| | |   / _` | | | |/ _ \ '__| ////////////////////////////////////////////////////////////////////////////////////////////////////
// | |___| | | | |  __/ (_| | |    | |__| (_| | |_| |  __/ |    ////////////////////////////////////////////////////////////////////////////////////////////////////
// |_____|_|_| |_|\___|\__,_|_|    |_____\__,_|\__, |\___|_|    ////////////////////////////////////////////////////////////////////////////////////////////////////
//                                             |___/            ////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

 //////////////////////////////////////////////////////////////////////////////////////////////   
 //        _____  _______ ______  _______ _______ _______      _    _        _____ _  _  _   // 
 //|      |     | |_____| |     \ |  |  | |_____| |             \  /  |        |   |  |  |   // 
 //|_____ |_____| |     | |_____/ |  |  | |     | |_____         \/   |_____ __|__ |__|__|   // 
 //                                                                                          //
 //////////////////////////////////////////////////////////////////////////////////////////////
#if defined FMOUTTILING && !defined(ASIP) && defined MANUALLOOPUNFOLDING && defined VLIWEXT // obv vliw
/** @brief Calculates a Fully-Connected (or Linear Layer) 
 *  
 *  Calculates a fully conntected Layer with the custom VLIW instructions for load and MAC
 *  Supports the following configurations:
 *  INPUTFMTILING false/true with input tile size 2
 *  OUTPUTFMTILING false/true with output tile sizes 1,2,4,8,10,12,14 (odd are currently not
 *                 supported as the SPR would need to be switched)
 *  FixedPt and SIMD and MANUALLOOPUNFOLDING only
 *
 *  @param inFeaturesSize Number of input neurons
 *  @param outFeaturesSize Number of output neurons
 *  @param hasBias FC with bias or not?
 *  @param weight Pointer to weights
 *  @param bias Pointer to bias
 *  @param inFeatures Input Feature Map
 *  @param outFeatures Output Feature Map
 */

#ifdef BATCHING


#ifdef EFFICIENT_CORE_ASSIGNMENT

void NOINLINE LinearLayer (
  // Layer Attributes
  int inFeaturesSize,
  int outFeaturesSize,
  int tile_size,
  short hasBias,
  int nr_inFeatures,
  // Layer Parameters
  data_t * __restrict__ weight,
  data_t * __restrict__ bias,
  // Input and Output Features
  data_t * __restrict__ inFeatures,
  data_t * __restrict__ outFeatures) //property(functional)
// {
#else

void NOINLINE LinearLayer (
  // Layer Attributes
  int inFeaturesSize,
  int outFeaturesSize,
  short hasBias,
  int nr_inFeatures,
  // Layer Parameters
  data_t * __restrict__ weight,
  data_t * __restrict__ bias,
  // Input and Output Features
  data_t * __restrict__ inFeatures,
  data_t * __restrict__ outFeatures) //property(functional)
#endif
{

  if(core_id==0)
  {
    PROFILING_LINEAR_START
    PROFILING_LINEAR_AMDAHL_SERIELL_START
  }

  // printf("### batching %d \n", nr_inFeatures);
  // printf("weight: ");PrintTensor(outFeaturesSize, bias);
  // printf("input: ");PrintTensor(inFeaturesSize, inFeatures);
  // printf("%d %x %x %x %x\n", core_id, &outFeatures, &weight, &bias, &inFeatures);

#ifdef MULTICORE

#ifndef LSTM_ON

#ifdef EFFICIENT_CORE_ASSIGNMENT // or TILING_HARD

  PROFILING_EFFICIENT_TILING_START

  int core_id = rt_core_id();
  int start   = tile_size * core_id;
  int stop    = MIN(start + tile_size, outFeaturesSize);
  int chunck_final = (stop-start);

  PROFILING_EFFICIENT_TILING_END

#else // TILING_HARD // or EFFICIENT_CORE_ASSIGNMENT

  PROFILING_TILING_START

  /* instructions to parallelize the workload:
  each core computes a balanced number of neurons */
  int core_id = rt_core_id();
  int n_cores = NR_CORES;
  int chunck = 1;
  /* handle the case when number of neurons
  is less than number of cores: chunck=1 */
  if(outFeaturesSize < n_cores)
  {
      n_cores = outFeaturesSize;
      // n_cores = 1;
      // if (core_id == 0)
      // {
      //   chunck = outFeaturesSize;
      // } else
      // {
      //   chunck = 0;
      // }
  }
  else
  {
      int Log2Core = __builtin_pulp_fl1(n_cores);
      chunck = (outFeaturesSize >> Log2Core) + ((outFeaturesSize & (n_cores-1))!=0);
  }
  /* start and stop neuron to be computed, for each core */
  int start = MIN(chunck * core_id,outFeaturesSize);
  int stop = MIN(start + chunck, outFeaturesSize);
  int chunck_final = (stop-start);

  PROFILING_TILING_END

#endif // TILING_HARD or EFFICIENT_CORE_ASSIGNMENT

#else // LSTM_ON

  /* instructions to parallelize the workload:
  each core computes a balanced number of neurons */
  int core_id = rt_core_id();
  int n_cores = NR_CORES;
  int chunck = 1;
  int chunkg_orig=1;

  int start_offset = 0;
  /* handle the case when number of neurons
  is less than number of cores: chunck=1 */
  if(outFeaturesSize <= n_cores)
  {
      // n_cores = outFeaturesSize;
      n_cores = 1;
      if (core_id == 0)
      {
        chunck = outFeaturesSize;
      } else
      {
        chunck = 0;
      }
  }
  else
  {
      int Log2Core = __builtin_pulp_fl1(n_cores);
      chunck = (outFeaturesSize >> Log2Core) + ((outFeaturesSize & (n_cores-1))!=0);
      chunkg_orig = chunck;
      // printf(" core_id %d a\n",core_id);
      if ((chunck % 2)!=0)
      {
        // printf(" core_id %d b\n",core_id);
        if ((core_id%2)==0)
        {
          // printf(" core_id %d +\n",core_id);
          chunkg_orig = chunck;
          chunck = chunck+1;
        }
        else
        {
          // printf(" core_id %d -\n",core_id);
          chunkg_orig = chunck;
          chunck = chunck-1;
          start_offset = 1;
        }
      }
  }
  /* start and stop neuron to be computed, for each core */
  int start = MIN((chunkg_orig) * core_id+start_offset,outFeaturesSize);
  int stop = MIN(start + chunck, outFeaturesSize);
  int chunck_final = (stop-start);

#endif // LSTM_ON

#ifdef DEBUG_LSTM
  // printf("bla in default w outputtiling, core_id: %d, start: %d, stop: %d , chunck_final: %d, outFeaturesSize: %d, NR_CORES: %d \n", core_id, start, stop, chunck_final, outFeaturesSize, NR_CORES);
#endif // DEBUG_LSTM
#endif

  int inFeaturesSizeP2 = inFeaturesSize/2;
  int inFeaturesSizeP2_p1 = inFeaturesSizeP2 + W_OFFSET/2;
  #ifdef FMINTILING
  int inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
  #else
  int inFeaturesSizeP4 = inFeaturesSizeP2;   // no input FM tiling
  #endif

// #if OUTPUTBUFFER > 8
//   int tileOptions[] = {OUTPUTBUFFER,8,4,2,1};
// #elif OUTPUTBUFFER > 4
  #if BATCHING == 1
  int tileOptions[] = {8,4,2,1};
  #elif BATCHING == 2
  int tileOptions[] = {4,2,1};
  // #elif BATCHING == 3
  // int tileOptions[] = {OUTPUTBUFFER,4,2,1};
  #elif BATCHING == 4
  int tileOptions[] = {2,1};
  #endif
// #else
//   int tileOptions[] = {OUTPUTBUFFER,2,1};
// #endif

#ifdef MULTICORE
  // data_t  * bias_ptr   = &bias[start];
  // v2s     * weight_ptr = &((v2s*)weight)[start*inFeaturesSizeP2_p1];
#if BATCHING > 0
  data_t  * bias_ptr0   = &bias[start];
  v2s     * weight_ptr0 = &((v2s*)weight)[start*inFeaturesSizeP2_p1];
  data_t  * outFeatures_ptr0 = &outFeatures[start];
  data_t  * inFeatures_ptr0 = &(inFeatures[0]);
#endif
#if BATCHING > 1
  data_t  * bias_ptr1   = &bias[start];
  v2s     * weight_ptr1 = &((v2s*)weight)[start*inFeaturesSizeP2_p1];
  data_t  * outFeatures_ptr1 = &outFeatures[1*outFeaturesSize + start];
  data_t  * inFeatures_ptr1 = &(inFeatures[1*inFeaturesSize]);
#endif
#if BATCHING > 3
  data_t  * bias_ptr2   = &bias[start];
  v2s     * weight_ptr2 = &((v2s*)weight)[start*inFeaturesSizeP2_p1];
  data_t  * outFeatures_ptr2 = &outFeatures[2*outFeaturesSize + start];
  data_t  * inFeatures_ptr2 = &(inFeatures)[2*inFeaturesSize];

  data_t  * bias_ptr3   = &bias[start];
  v2s     * weight_ptr3 = &((v2s*)weight)[start*inFeaturesSizeP2_p1];
  data_t  * outFeatures_ptr3 = &outFeatures[3*outFeaturesSize + start];
  data_t  * inFeatures_ptr3 = &(inFeatures)[3*inFeaturesSize];
#endif
  // printf("%d %x %x %x %x\n", core_id, &outFeatures_ptr, &weight_ptr, &bias_ptr, &inFeatures);

//   printf("input in 1 : ");
//   PrintTensor(100, inFeatures_ptr0);
// #if BATCHING > 1
//   printf("input in 2 : ");
//   PrintTensor(100, inFeatures_ptr1);
// #endif


  int outFeaturesPerTile = 1;
#else
  data_t  * bias_ptr   = bias;
  v2s     * weight_ptr = (v2s*)weight;
  data_t  * outFeatures_ptr = outFeatures;
  int outFeaturesPerTile = 1;
#endif

  // register definition for manual loop unfolding
  register_attribute int32_t   temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9, temp10, temp11, temp12, temp13, temp14;
  register_attribute uint32_t  addr0, addr1, addr2, addr3, addr4, addr5, addr6, addr7, addr8, addr9, addr10, addr11, addr12, addr13, addr14;
  register_attribute uint32_t  in_addr0, in_addr1, in_addr2, in_addr3;

  // register null
  register int x0 asm("x0");

#ifdef MULTICORE
  int outFeatureTiles;
  int outFeaturesSize_remain = chunck_final;
#else
  int outFeatureTiles;
  int outFeaturesSize_remain = outFeaturesSize;
#endif

  if(core_id==0)
  {
    PROFILING_LINEAR_AMDAHL_SERIELL_END
    PROFILING_LINEAR_AMDAHL_PARALLEL_START
  }
  // Tile with largest tileOption
  for(unsigned int i=0; i<sizeof(tileOptions)/sizeof(int); i++) {
    outFeaturesPerTile = tileOptions[i];
    outFeatureTiles = outFeaturesSize_remain/outFeaturesPerTile;

    if(outFeatureTiles == 0) continue;

    // Select Tile Size
    switch(outFeaturesPerTile) {
     #if OUTPUTBUFFER > 2
     case OUTPUTBUFFER:
     for (int o_tile=0; o_tile< outFeatureTiles; o_tile++)
     {
                      // printf("in case 8\n");

      // Manual loop unfolding
      // Inititalize accumulation registers with bias and shift accordingly

    #if BATCHING == 1
        temp0 = (int32_t)bias_ptr0[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
        temp1 = (int32_t)bias_ptr0[o_tile*outFeaturesPerTile+1]<<(q_fraqP1);
        temp2 = (int32_t)bias_ptr0[o_tile*outFeaturesPerTile+2]<<(q_fraqP1);
        temp3 = (int32_t)bias_ptr0[o_tile*outFeaturesPerTile+3]<<(q_fraqP1);
        temp4 = (int32_t)bias_ptr0[o_tile*outFeaturesPerTile+4]<<(q_fraqP1);
        temp5 = (int32_t)bias_ptr0[o_tile*outFeaturesPerTile+5]<<(q_fraqP1);
        // temp6 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+6]<<(q_fraqP1);
        // temp7 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+7]<<(q_fraqP1);

        temp13 = (int32_t)bias_ptr0[o_tile*outFeaturesPerTile+OUTPUTBUFFER-2]<<(q_fraqP1);
        temp14 = (int32_t)bias_ptr0[o_tile*outFeaturesPerTile+OUTPUTBUFFER-1]<<(q_fraqP1);
    #endif



    #if BATCHING == 1
        addr0  = (uint32_t) &((v2s*)weight_ptr0)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 0))];
        addr1  = (uint32_t) &((v2s*)weight_ptr0)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 1))];
        addr2  = (uint32_t) &((v2s*)weight_ptr0)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 2))];
        addr3  = (uint32_t) &((v2s*)weight_ptr0)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 3))];
        addr4  = (uint32_t) &((v2s*)weight_ptr0)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 4))];
        addr5  = (uint32_t) &((v2s*)weight_ptr0)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 5))];
        addr6  = (uint32_t) &((v2s*)weight_ptr0)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 6))];
        addr7  = (uint32_t) &((v2s*)weight_ptr0)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 7))];
    #endif

        asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload first weight
        asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload first weight

    #if BATCHING == 1
        in_addr0 = (uint32_t) (((v2s*)inFeatures_ptr0+0*inFeaturesSizeP4));
    #endif

        for(int i=0; i<inFeaturesSizeP4; i++) {



            // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
    #if BATCHING == 1
            v2s inF_temp_0;//  = ((v2s*)inFeatures)[2*i];
            v2s inF_temp2_0;// = ((v2s*)inFeatures)[2*i+1];
            asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp_0), "+r" (in_addr0));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
            asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2_0), "+r" (in_addr0)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
    #endif

        // printf("w in 1 : %d \n", addr0);
        // printf("temp0: %d \n", temp0>>(q_fraqP1));
        // printf("temp1: %d \n", temp1>>(q_fraqP1));
        // printf("temp2: %d \n", temp2>>(q_fraqP1));
        // printf("temp3: %d \n", temp3>>(q_fraqP1));

  //             asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr0));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
  // #ifdef FMINTILING
  //             asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr0)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
  // #endif
              // MANUAL loop unfolding
    #if BATCHING == 1
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp_0) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp_0) );
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr4) : "r" (inF_temp_0) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr5) : "r" (inF_temp_0) );
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp4),  "+r" (addr6) : "r" (inF_temp_0) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp5),  "+r" (addr7) : "r" (inF_temp_0) );
            // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp6),  "+r" (addr8) : "r" (inF_temp_0) );
            // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr9) : "r" (inF_temp_0) );

            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp13),  "+r" (addr0) : "r" (inF_temp_0) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp14),  "+r" (addr1) : "r" (inF_temp_0) );
    #endif

// do it twice for FMINTILING
  // #ifdef FMINTILING
        // printf("w in 1 : %d \n", addr0);
        // printf("temp0: %d \n", temp0>>(q_fraqP1));
        // printf("temp1: %d \n", temp1>>(q_fraqP1));
        // printf("temp2: %d \n", temp2>>(q_fraqP1));
        // printf("temp3: %d \n", temp3>>(q_fraqP1));

    #if BATCHING == 1
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp2_0) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp2_0) );
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr4) : "r" (inF_temp2_0) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr5) : "r" (inF_temp2_0) );
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp4),  "+r" (addr6) : "r" (inF_temp2_0) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp5),  "+r" (addr7) : "r" (inF_temp2_0) );
            // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp6),  "+r" (addr8) : "r" (inF_temp2_0) );
            // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr9) : "r" (inF_temp2_0) );
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp13),  "+r" (addr0) : "r" (inF_temp2_0) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp14),  "+r" (addr1) : "r" (inF_temp2_0) );
    #endif

            // }
            }
  #ifdef FMINTILING
          if(inFeaturesSizeP2%2==1) { // add contribution of left over input channel (input channels not multiple of 4)
              v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
              // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
              // asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
              // asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];

              // printf("blabsli\n");
      #if BATCHING == 1
              v2s inF_temp_0;//  = ((v2s*)inFeatures)[2*i];
              // v2s inF_temp2_0;// = ((v2s*)inFeatures)[2*i+1];
              asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp_0), "+r" (in_addr0));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
              // asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2_0), "+r" (in_addr0)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
      #endif


               // MANUAL loop unfolding
               // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {
               // __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+o_rel)) + i], temp[o_rel]);


      #if BATCHING == 1
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp_0) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp_0) );
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr4) : "r" (inF_temp_0) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr5) : "r" (inF_temp_0) );
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp4),  "+r" (addr6) : "r" (inF_temp_0) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp5),  "+r" (addr7) : "r" (inF_temp_0) );
              // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp6),  "+r" (addr8) : "r" (inF_temp_0) );
              // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr9) : "r" (inF_temp_0) );
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp13),  "+r" (addr0) : "r" (inF_temp_0) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp14),  "+r" (addr1) : "r" (inF_temp_0) );
      #endif

             }

  #endif

// Store the final results back to the memory
        // for(int o_rel =0;o_rel<outFeaturesPerTile;o_rel++) {

      #if BATCHING == 1
             outFeatures_ptr0[(o_tile*outFeaturesPerTile+0)] = temp0>>(q_fraqP1);
             outFeatures_ptr0[(o_tile*outFeaturesPerTile+1)] = temp1>>(q_fraqP1);
             outFeatures_ptr0[(o_tile*outFeaturesPerTile+2)] = temp2>>(q_fraqP1);
             outFeatures_ptr0[(o_tile*outFeaturesPerTile+3)] = temp3>>(q_fraqP1);
             outFeatures_ptr0[(o_tile*outFeaturesPerTile+4)] = temp4>>(q_fraqP1);
             outFeatures_ptr0[(o_tile*outFeaturesPerTile+5)] = temp5>>(q_fraqP1);
             // outFeatures_ptr0[(o_tile*outFeaturesPerTile+6)] = temp6>>(q_fraqP1);
             // outFeatures_ptr0[(o_tile*outFeaturesPerTile+7)] = temp7>>(q_fraqP1);
             outFeatures_ptr0[(o_tile*outFeaturesPerTile+6)] = temp13>>(q_fraqP1);
             outFeatures_ptr0[(o_tile*outFeaturesPerTile+7)] = temp14>>(q_fraqP1);
      // #elif BATCHING == 2
      //        outFeatures_ptr0[(o_tile*outFeaturesPerTile+0)] = temp0>>(q_fraqP1);
      //        outFeatures_ptr0[(o_tile*outFeaturesPerTile+1)] = temp1>>(q_fraqP1);
      //        outFeatures_ptr0[(o_tile*outFeaturesPerTile+2)] = temp2>>(q_fraqP1);
      //        outFeatures_ptr0[(o_tile*outFeaturesPerTile+3)] = temp3>>(q_fraqP1);

      //        outFeatures_ptr1[(o_tile*outFeaturesPerTile+0)] = temp4>>(q_fraqP1);
      //        outFeatures_ptr1[(o_tile*outFeaturesPerTile+1)] = temp5>>(q_fraqP1);
      //        outFeatures_ptr1[(o_tile*outFeaturesPerTile+2)] = temp6>>(q_fraqP1);
      //        outFeatures_ptr1[(o_tile*outFeaturesPerTile+3)] = temp7>>(q_fraqP1);
      // #elif BATCHING == 4
      //        outFeatures_ptr3[(o_tile*outFeaturesPerTile+0)] = temp0>>(q_fraqP1);
      //        outFeatures_ptr3[(o_tile*outFeaturesPerTile+1)] = temp1>>(q_fraqP1);
      //        outFeatures_ptr3[(o_tile*outFeaturesPerTile+0)] = temp2>>(q_fraqP1);
      //        outFeatures_ptr3[(o_tile*outFeaturesPerTile+1)] = temp3>>(q_fraqP1);
      //        outFeatures_ptr3[(o_tile*outFeaturesPerTile+0)] = temp4>>(q_fraqP1);
      //        outFeatures_ptr3[(o_tile*outFeaturesPerTile+1)] = temp5>>(q_fraqP1);
      //        outFeatures_ptr3[(o_tile*outFeaturesPerTile+0)] = temp6>>(q_fraqP1);
      //        outFeatures_ptr3[(o_tile*outFeaturesPerTile+1)] = temp7>>(q_fraqP1);
      #endif
        //      if(core_id==0){
        //   printf("Results bla in: ");
        // PrintTensor(4*200, outFeatures);
        //   }

                // printf("outFeatures[%i]=%i\n", o_rel, outFeatures[(o_tile*outFeaturesPerTile+o_rel)]);
        // }

           }
           break;
     #endif
     #if OUTPUTBUFFER > 8
           case 8:
           for (int o_tile=0; o_tile< outFeatureTiles; o_tile++)
           {

            temp0 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
            temp1 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+1]<<(q_fraqP1);
            temp2 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+2]<<(q_fraqP1);
            temp3 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+3]<<(q_fraqP1);
            temp4 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+4]<<(q_fraqP1);
            temp5 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+5]<<(q_fraqP1);
            temp6 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+6]<<(q_fraqP1);
            temp7 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+7]<<(q_fraqP1);

          // }
          // printf("wght=%i, addr before=%i (+%i)=%i\n", weight, addr0, inFeaturesSizeP2*outFeaturesPerTile, addr0+4*inFeaturesSizeP2*outFeaturesPerTile);
            addr0 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+0))];
            addr1 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+1))];
            addr2 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+2))];
            addr3 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+3))];
            addr4 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+4))];
            addr5 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+5))];
            addr6 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+6))];
            addr7 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+7))];


          asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload first weight
          asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload 2nd weight

          in_addr = (((v2s*)inFeatures));
          for(int i=0; i<inFeaturesSizeP4; i++) {
              v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
              v2s inF_temp2;// = ((v2s*)inFeatures)[2*i+1];

              // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
              asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
  #ifdef FMINTILING
              asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
  #endif
              // MANUAL loop unfolding
              // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {
              // __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+o_rel)) + i], temp[o_rel]);
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp) );
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr4) : "r" (inF_temp) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr5) : "r" (inF_temp) );
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp4),  "+r" (addr6) : "r" (inF_temp) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp5),  "+r" (addr7) : "r" (inF_temp) );
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr0) : "r" (inF_temp) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr1) : "r" (inF_temp) );
  #ifdef FMINTILING
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp2) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp2) );
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr4) : "r" (inF_temp2) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr5) : "r" (inF_temp2) );
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp4),  "+r" (addr6) : "r" (inF_temp2) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp5),  "+r" (addr7) : "r" (inF_temp2) );
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr0) : "r" (inF_temp2) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr1) : "r" (inF_temp2) );
  #endif
            // }
            }
  #ifdef FMINTILING
          if(inFeaturesSizeP2%2==1) { // add contribution of left over input channel (input channels not multiple of 4)

               v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
               asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
               asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp) );
               asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp) );
               asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr4) : "r" (inF_temp) );
               asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr5) : "r" (inF_temp) );
               asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp4),  "+r" (addr6) : "r" (inF_temp) );
               asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp5),  "+r" (addr7) : "r" (inF_temp) );
               asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr0) : "r" (inF_temp) );
               asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr1) : "r" (inF_temp) );

             }
  #endif
        // for(int o_rel =0;o_rel<outFeaturesPerTile;o_rel++) {
             outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = temp0>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+1)] = temp1>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+2)] = temp2>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+3)] = temp3>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+4)] = temp4>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+5)] = temp5>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+6)] = temp6>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+7)] = temp7>>(q_fraqP1);
        // }
           }
           break;
     #endif
     #if OUTPUTBUFFER > 4
           case 4:
           for (int o_tile=0; o_tile< outFeatureTiles; o_tile++)
           {
              // printf("in case 4\n");


          #if BATCHING > 0
              temp0 = (int32_t)bias_ptr0[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
              temp1 = (int32_t)bias_ptr0[o_tile*outFeaturesPerTile+1]<<(q_fraqP1);
              temp2 = (int32_t)bias_ptr0[o_tile*outFeaturesPerTile+2]<<(q_fraqP1);
              temp3 = (int32_t)bias_ptr0[o_tile*outFeaturesPerTile+3]<<(q_fraqP1);
          #endif
          #if BATCHING > 1
              temp4 = (int32_t)bias_ptr1[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
              temp5 = (int32_t)bias_ptr1[o_tile*outFeaturesPerTile+1]<<(q_fraqP1);
              temp6 = (int32_t)bias_ptr1[o_tile*outFeaturesPerTile+2]<<(q_fraqP1);
              temp7 = (int32_t)bias_ptr1[o_tile*outFeaturesPerTile+3]<<(q_fraqP1);
          #endif

          // }
          #if BATCHING > 0
            addr0  = (uint32_t) &((v2s*)weight_ptr0)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 0))];
            addr1  = (uint32_t) &((v2s*)weight_ptr0)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 1))];
            addr2  = (uint32_t) &((v2s*)weight_ptr0)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 2))];
            addr3  = (uint32_t) &((v2s*)weight_ptr0)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 3))];

          #endif
          #if BATCHING > 1
            addr4  = (uint32_t) &((v2s*)weight_ptr1)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 0))];
            addr5  = (uint32_t) &((v2s*)weight_ptr1)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 1))];
            addr6  = (uint32_t) &((v2s*)weight_ptr1)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 2))];
            addr7  = (uint32_t) &((v2s*)weight_ptr1)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 3))];
        #endif


    #if BATCHING > 0
          asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload first weight
          asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload first weight
    #endif

    #if BATCHING > 0
        in_addr0 = (uint32_t) &((v2s*)inFeatures_ptr0)[0]; //inFeatures+0*inFeaturesSize));
    #endif



        for(int i=0; i<inFeaturesSizeP4; i++) {

    #if BATCHING > 0
            v2s inF_temp_0;//  = ((v2s*)inFeatures)[2*i];
            v2s inF_temp2_0;// = ((v2s*)inFeatures)[2*i+1];
            asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp_0), "+r" (in_addr0));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
            asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2_0), "+r" (in_addr0)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
    #endif

        // printf("w in 1 : %d w in 2 : %d input in 2 : %d\n", addr0, addr4, inF_temp_0);
        // printf("temp0: %d \n", temp0>>(q_fraqP1));
        // printf("temp1: %d \n", temp1>>(q_fraqP1));
        // printf("temp2: %d \n", temp2>>(q_fraqP1));
        // printf("temp3: %d \n", temp3>>(q_fraqP1));

    #if BATCHING > 0
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp_0) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp_0) );
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr0) : "r" (inF_temp_0) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr1) : "r" (inF_temp_0) );
    #endif

        // printf("w in 1 : %d w in 2 : %d input in 2 : %d\n", addr0, addr4, inF_temp_0);
        // printf("temp0: %d \n", temp0>>(q_fraqP1));
        // printf("temp1: %d \n", temp1>>(q_fraqP1));
        // printf("temp2: %d \n", temp2>>(q_fraqP1));
        // printf("temp3: %d \n", temp3>>(q_fraqP1));

    #if BATCHING > 0
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp2_0) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp2_0) );
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr0) : "r" (inF_temp2_0) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr1) : "r" (inF_temp2_0) );
    #endif


            }

    #if BATCHING > 1
        in_addr1 = (uint32_t) &((v2s*)inFeatures_ptr1)[0]; //inFeatures+1*inFeaturesSize));
    #endif

    #if BATCHING > 1
          asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr4) : "r" (x0) ); // preload first weight
          asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr5) : "r" (x0) ); // preload first weight
    #endif

        for(int i=0; i<inFeaturesSizeP4; i++) {

    #if BATCHING > 1
            v2s inF_temp_1;//  = ((v2s*)inFeatures)[2*i];
            v2s inF_temp2_1;// = ((v2s*)inFeatures)[2*i+1];
            asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp_1), "+r" (in_addr1));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
            asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2_1), "+r" (in_addr1)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
    #endif

        // printf("w in 1 : %d w in 2 : %d input in 2 : %d\n", addr0, addr4, inF_temp_1);
        // printf("2 temp0: %d \n", temp4>>(q_fraqP1));
        // printf("2 temp1: %d \n", temp5>>(q_fraqP1));
        // printf("2 temp2: %d \n", temp6>>(q_fraqP1));
        // printf("2 temp3: %d \n", temp7>>(q_fraqP1));

    #if BATCHING > 1
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp4),  "+r" (addr6) : "r" (inF_temp_1) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp5),  "+r" (addr7) : "r" (inF_temp_1) );
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr4) : "r" (inF_temp_1) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr5) : "r" (inF_temp_1) );
    #endif

        // printf("w in 1 : %d w in 2 : %d input in 2 : %d\n", addr0, addr4, inF_temp_1);
        // printf("2 temp0: %d \n", temp4>>(q_fraqP1));
        // printf("2 temp1: %d \n", temp5>>(q_fraqP1));
        // printf("2 temp2: %d \n", temp6>>(q_fraqP1));
        // printf("2 temp3: %d \n", temp7>>(q_fraqP1));

    #if BATCHING > 1
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp4),  "+r" (addr6) : "r" (inF_temp2_1) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp5),  "+r" (addr7) : "r" (inF_temp2_1) );
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr4) : "r" (inF_temp2_1) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr5) : "r" (inF_temp2_1) );
    #endif
            }



  #ifdef FMINTILING
          if(inFeaturesSizeP2%2==1) { // add contribution of left over input channel (input channels not multiple of 4)

            // printf("in here0\n");
                  // printf("in here0, inaddr %d inaddr %d\n", in_addr0, in_addr1);

               // v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
               // asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
               // asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp) );
               // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp) );
               // asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr0) : "r" (inF_temp) );
               // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr1) : "r" (inF_temp) );

    // #if BATCHING > 0
    //       asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload first weight
    //       asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload first weight
    // #endif
    #if BATCHING > 0
            v2s inF_temp_0;//  = ((v2s*)inFeatures)[2*i];
            // v2s inF_temp2_0;// = ((v2s*)inFeatures)[2*i+1];
            asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp_0), "+r" (in_addr0));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
            // asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2_0), "+r" (in_addr0)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
    #endif
        // printf("w in 1 : %d w in 2 : %d input in 1 : %d i\n", addr0, addr4, inF_temp_0);
        // printf("temp0: %d \n", temp0>>(q_fraqP1));
        // printf("temp1: %d \n", temp1>>(q_fraqP1));
        // printf("temp2: %d \n", temp2>>(q_fraqP1));
        // printf("temp3: %d \n", temp3>>(q_fraqP1));
        // printf("2 temp0: %d \n", temp4>>(q_fraqP1));
        // printf("2 temp1: %d \n", temp5>>(q_fraqP1));
        // printf("2 temp2: %d \n", temp6>>(q_fraqP1));
        // printf("2 temp3: %d \n", temp7>>(q_fraqP1));

    #if BATCHING > 0
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp_0) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp_0) );
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr0) : "r" (inF_temp_0) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr1) : "r" (inF_temp_0) );
    #endif

        // printf("w in 1 : %d w in 2 : %d input in 1 : %d i\n", addr0, addr4, inF_temp_0);
        // printf("temp0: %d \n", temp0>>(q_fraqP1));
        // printf("temp1: %d \n", temp1>>(q_fraqP1));
        // printf("temp2: %d \n", temp2>>(q_fraqP1));
        // printf("temp3: %d \n", temp3>>(q_fraqP1));
        // printf("2 temp0: %d \n", temp4>>(q_fraqP1));
        // printf("2 temp1: %d \n", temp5>>(q_fraqP1));
        // printf("2 temp2: %d \n", temp6>>(q_fraqP1));
        // printf("2 temp3: %d \n", temp7>>(q_fraqP1));

          }

    if(inFeaturesSizeP2%2==1) { // add contribution of left over input channel (input channels not multiple of 4)

                  // printf("in here1, inaddr %d inaddr %d\n", in_addr0, in_addr1);

    // #if BATCHING > 1
    //       asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr4) : "r" (x0) ); // preload first weight
    //       asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr5) : "r" (x0) ); // preload first weight
    // #endif
    #if BATCHING > 1
            v2s inF_temp_1;//  = ((v2s*)inFeatures)[2*i];
            asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp_1), "+r" (in_addr1));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
    #endif
    #if BATCHING > 1
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp4),  "+r" (addr6) : "r" (inF_temp_1) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp5),  "+r" (addr7) : "r" (inF_temp_1) );
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr4) : "r" (inF_temp_1) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr5) : "r" (inF_temp_1) );
    #endif

      // printf("w in 1 : %d w in 2 :  %d input in 2 : %d\n", addr0, addr4, inF_temp_1);
        // printf("temp0: %d \n", temp0>>(q_fraqP1));
        // printf("temp1: %d \n", temp1>>(q_fraqP1));
        // printf("temp2: %d \n", temp2>>(q_fraqP1));
        // printf("temp3: %d \n", temp3>>(q_fraqP1));
        // printf("2 temp0: %d \n", temp4>>(q_fraqP1));
        // printf("2 temp1: %d \n", temp5>>(q_fraqP1));
        // printf("2 temp2: %d \n", temp6>>(q_fraqP1));
        // printf("2 temp3: %d \n", temp7>>(q_fraqP1));

             }
  #endif
        // for(int o_rel =0;o_rel<outFeaturesPerTile;o_rel++) {
             // printf("finish\n");
    #if BATCHING > 0
             outFeatures_ptr0[(o_tile*outFeaturesPerTile+0)] = temp0>>(q_fraqP1);
             outFeatures_ptr0[(o_tile*outFeaturesPerTile+1)] = temp1>>(q_fraqP1);
             outFeatures_ptr0[(o_tile*outFeaturesPerTile+2)] = temp2>>(q_fraqP1);
             outFeatures_ptr0[(o_tile*outFeaturesPerTile+3)] = temp3>>(q_fraqP1);
    #endif
    #if BATCHING > 1
             outFeatures_ptr1[(o_tile*outFeaturesPerTile+0)] = temp4>>(q_fraqP1);
             outFeatures_ptr1[(o_tile*outFeaturesPerTile+1)] = temp5>>(q_fraqP1);
             outFeatures_ptr1[(o_tile*outFeaturesPerTile+2)] = temp6>>(q_fraqP1);
             outFeatures_ptr1[(o_tile*outFeaturesPerTile+3)] = temp7>>(q_fraqP1);
      #endif

        //                   if(core_id==0){
        //   printf("Results bla in: ");
        // PrintTensor(2*500, outFeatures);
        //   }
        // printf("Results bla in: ");
        // PrintTensor(2*200, outFeatures);
             // outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = temp0>>(q_fraqP1);
             // outFeatures_ptr[(o_tile*outFeaturesPerTile+1)] = temp1>>(q_fraqP1);
             // outFeatures_ptr[(o_tile*outFeaturesPerTile+2)] = temp2>>(q_fraqP1);
             // outFeatures_ptr[(o_tile*outFeaturesPerTile+3)] = temp3>>(q_fraqP1);
               // printf("outFeatures[%i]=%i\n", o_rel, outFeatures[(o_tile*outFeaturesPerTile+o_rel)]);
        // }
           }
           break;


     #endif
           case 2:


        //     if(core_id==0){
        //   printf("Results bla in: ");
        // PrintTensor(4*200, outFeatures);
        //   }
           for (int o_tile=0; o_tile< outFeatureTiles; o_tile++)
           {
              // printf("in case 2\n");
          //   temp0 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
          //   temp1 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+1]<<(q_fraqP1);
          // // temp2 = (int32_t)bias[o_tile*outFeaturesPerTile+2]<<(q_fraqP1);
          // // temp3 = (int32_t)bias[o_tile*outFeaturesPerTile+3]<<(q_fraqP1);
          // // }
          //   addr0 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+0))];
          //   addr1 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+1))];

          #if BATCHING > 0
              temp0 = (int32_t)bias_ptr0[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
              temp1 = (int32_t)bias_ptr0[o_tile*outFeaturesPerTile+1]<<(q_fraqP1);
              // temp2 = (int32_t)bias_ptr0[o_tile*outFeaturesPerTile+2]<<(q_fraqP1);
              // temp3 = (int32_t)bias_ptr0[o_tile*outFeaturesPerTile+3]<<(q_fraqP1);
          #endif
          #if BATCHING > 1
              temp4 = (int32_t)bias_ptr1[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
              temp5 = (int32_t)bias_ptr1[o_tile*outFeaturesPerTile+1]<<(q_fraqP1);
              // temp6 = (int32_t)bias_ptr1[o_tile*outFeaturesPerTile+2]<<(q_fraqP1);
              // temp7 = (int32_t)bias_ptr1[o_tile*outFeaturesPerTile+3]<<(q_fraqP1);
          #endif
          #if BATCHING > 3
              temp6 = (int32_t)bias_ptr2[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
              temp7 = (int32_t)bias_ptr2[o_tile*outFeaturesPerTile+1]<<(q_fraqP1);

              temp8 = (int32_t)bias_ptr3[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
              temp9 = (int32_t)bias_ptr3[o_tile*outFeaturesPerTile+1]<<(q_fraqP1);
          #endif
          // }
          #if BATCHING > 0
              addr0  = (uint32_t) &((v2s*)weight_ptr0)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 0))];
              addr1  = (uint32_t) &((v2s*)weight_ptr0)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 1))];
              // addr2  = (uint32_t) &((v2s*)weight_ptr0)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 2))];
              // addr3  = (uint32_t) &((v2s*)weight_ptr0)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 3))];
          #endif


          #if BATCHING > 0
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload first weight
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload first weight

          // #if BATCHING > 0
          //     in_addr0 = (uint32_t) &((v2s*)inFeatures_ptr0)[0]; //inFeatures+0*inFeaturesSize));
          // #endif


          for(int i=0; i<inFeaturesSizeP2; i++) {

            // printf("index i %d", i);

            v2s inF_temp1 = ((v2s*)inFeatures_ptr0)[i];

            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr0) : "r" (inF_temp1) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr1) : "r" (inF_temp1) );

          }
          #endif


          #if BATCHING > 1
              addr4  = (uint32_t) &((v2s*)weight_ptr1)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 0))];
              addr5  = (uint32_t) &((v2s*)weight_ptr1)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 1))];
              // addr6  = (uint32_t) &((v2s*)weight_ptr1)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 2))];
              // addr7  = (uint32_t) &((v2s*)weight_ptr1)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 3))];
          #endif
      // #if BATCHING > 1
      //     in_addr1 = (uint32_t) &((v2s*)inFeatures_ptr1)[0]; //inFeatures+1*inFeaturesSize));
      // #endif

      #if BATCHING > 1
          asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr4) : "r" (x0) ); // preload first weight
          asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr5) : "r" (x0) ); // preload first weight


          for(int i=0; i<inFeaturesSizeP2; i++) {

            v2s inF_temp2 = ((v2s*)inFeatures_ptr1)[i];

            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp4),  "+r" (addr4) : "r" (inF_temp2) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp5),  "+r" (addr5) : "r" (inF_temp2) );

          }
      #endif



          #if BATCHING > 3
              addr6  = (uint32_t) &((v2s*)weight_ptr2)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 0))];
              addr7  = (uint32_t) &((v2s*)weight_ptr2)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 1))];
              addr8  = (uint32_t) &((v2s*)weight_ptr3)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 0))];
              addr9  = (uint32_t) &((v2s*)weight_ptr3)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 1))];
          #endif


      // #if BATCHING > 2
      //     in_addr2 = (uint32_t) &((v2s*)inFeatures_ptr2)[0]; //inFeatures+1*inFeaturesSize));
      // #endif

      #if BATCHING > 2
          asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr6) : "r" (x0) ); // preload first weight
          asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr7) : "r" (x0) ); // preload first weight


          for(int i=0; i<inFeaturesSizeP2; i++) {

            v2s inF_temp3 = ((v2s*)inFeatures_ptr2)[i];

            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr6) : "r" (inF_temp3) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr7) : "r" (inF_temp3) );

          }
      #endif

      // #if BATCHING > 3
      //     in_addr3 = (uint32_t) &((v2s*)inFeatures_ptr3)[0]; //inFeatures+1*inFeaturesSize));
      // #endif

      #if BATCHING > 3
          asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr8) : "r" (x0) ); // preload first weight
          asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr9) : "r" (x0) ); // preload first weight

          for(int i=0; i<inFeaturesSizeP2; i++) {

            v2s inF_temp4 = ((v2s*)inFeatures_ptr3)[i];

            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp8),  "+r" (addr8) : "r" (inF_temp4) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp9),  "+r" (addr9) : "r" (inF_temp4) );

          }
      #endif


      #if BATCHING > 0
             outFeatures_ptr0[(o_tile*outFeaturesPerTile+0)] = temp0>>(q_fraqP1);
             outFeatures_ptr0[(o_tile*outFeaturesPerTile+1)] = temp1>>(q_fraqP1);
      #endif
      #if BATCHING > 1
             outFeatures_ptr1[(o_tile*outFeaturesPerTile+0)] = temp4>>(q_fraqP1);
             outFeatures_ptr1[(o_tile*outFeaturesPerTile+1)] = temp5>>(q_fraqP1);
      #endif
      #if BATCHING > 3
             outFeatures_ptr2[(o_tile*outFeaturesPerTile+0)] = temp6>>(q_fraqP1);
             outFeatures_ptr2[(o_tile*outFeaturesPerTile+1)] = temp7>>(q_fraqP1);
             outFeatures_ptr3[(o_tile*outFeaturesPerTile+0)] = temp8>>(q_fraqP1);
             outFeatures_ptr3[(o_tile*outFeaturesPerTile+1)] = temp9>>(q_fraqP1);
      #endif
        //   if(core_id==0){
        //   printf("Results bla in: ");
        // PrintTensor(4*200, outFeatures);
        //   }

        }
        break;
        case 1:
        //           if(core_id==0){
        //   printf("Results bla in: ");
        // PrintTensor(4*200, outFeatures);
        //   }
        for (int o_tile=0; o_tile< outFeatureTiles; o_tile++)
        {
          // printf("in case 1\n");

      #if BATCHING > 0
          temp0 = (int32_t)bias_ptr0[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
      #endif
      #if BATCHING > 1
          temp1 = (int32_t)bias_ptr1[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
      #endif

          for(int i=0; i<inFeaturesSizeP2; i++) {
      #if BATCHING > 0
            v2s inF_temp1 = ((v2s*)inFeatures_ptr0)[i];
      #endif
      #if BATCHING > 1
            v2s inF_temp2 = ((v2s*)inFeatures_ptr1)[i];
      #endif
      #if BATCHING > 0
            temp0 = __SUMDOTP2(inF_temp1, ((v2s*)weight_ptr0)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile)) + i], temp0);
      #endif
      #if BATCHING > 1
            temp1 = __SUMDOTP2(inF_temp2, ((v2s*)weight_ptr1)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile)) + i], temp1);
      #endif

          }


      #if BATCHING > 3
          temp2 = (int32_t)bias_ptr2[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
          temp3 = (int32_t)bias_ptr3[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
      #endif
          for(int i=0; i<inFeaturesSizeP2; i++) {
      #if BATCHING > 3
            v2s inF_temp3 = ((v2s*)inFeatures_ptr2)[i];
            v2s inF_temp4 = ((v2s*)inFeatures_ptr3)[i];
      #endif
      #if BATCHING > 3
            temp2 = __SUMDOTP2(inF_temp3, ((v2s*)weight_ptr2)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile)) + i], temp2);
            temp3 = __SUMDOTP2(inF_temp4, ((v2s*)weight_ptr3)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile)) + i], temp3);
      #endif
          }
          // outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = temp0>>(q_fraqP1);
      #if BATCHING > 0
             outFeatures_ptr0[(o_tile*outFeaturesPerTile+0)] = temp0>>(q_fraqP1);
      #endif
      #if BATCHING > 1
             outFeatures_ptr1[(o_tile*outFeaturesPerTile+0)] = temp1>>(q_fraqP1);
      #endif
      #if BATCHING > 3
             outFeatures_ptr2[(o_tile*outFeaturesPerTile+0)] = temp2>>(q_fraqP1);
             outFeatures_ptr3[(o_tile*outFeaturesPerTile+0)] = temp3>>(q_fraqP1);
      #endif
        //   if(core_id==0){
        //   printf("Results bla in: ");
        // PrintTensor(4*200, outFeatures);
        //   }
        }
        break;


      }


      // printf("hallo %d %d %d \n", core_id, outFeaturesSize_remain, *outFeatures_ptr0);

  // move pointers for next iteration
#if BATCHING > 0
      outFeatures_ptr0         = &(outFeatures_ptr0[(outFeatureTiles*outFeaturesPerTile)]);
      bias_ptr0                = &(bias_ptr0[outFeatureTiles*outFeaturesPerTile]);
      weight_ptr0              = &((v2s*)weight_ptr0)[(inFeaturesSizeP2_p1*(outFeatureTiles*outFeaturesPerTile))];
#endif
#if BATCHING > 1
      outFeatures_ptr1         = &(outFeatures_ptr1[(outFeatureTiles*outFeaturesPerTile)]);
      bias_ptr1                = &(bias_ptr1[outFeatureTiles*outFeaturesPerTile]);
      weight_ptr1              = &((v2s*)weight_ptr1)[(inFeaturesSizeP2_p1*(outFeatureTiles*outFeaturesPerTile))];
#endif
#if BATCHING > 3
      outFeatures_ptr2         = &outFeatures_ptr2[(outFeatureTiles*outFeaturesPerTile)];
      outFeatures_ptr3         = &outFeatures_ptr3[(outFeatureTiles*outFeaturesPerTile)];
      bias_ptr2                = &bias_ptr2[outFeatureTiles*outFeaturesPerTile];
      bias_ptr3                = &bias_ptr3[outFeatureTiles*outFeaturesPerTile];
      weight_ptr2              = &((v2s*)weight_ptr2)[(inFeaturesSizeP2_p1*(outFeatureTiles*outFeaturesPerTile))];
      weight_ptr3              = &((v2s*)weight_ptr3)[(inFeaturesSizeP2_p1*(outFeatureTiles*outFeaturesPerTile))];

#endif
      outFeaturesSize_remain -= outFeatureTiles*outFeaturesPerTile;

      // printf("hallo %d %d %d \n", core_id, outFeaturesSize_remain, *outFeatures_ptr0);
      if (outFeaturesSize_remain==0) break;
    }

      // if(core_id==0){
      //     printf("Results bla in: ");
      //   PrintTensor(2*500, outFeatures);
      //     }

  if(core_id==0)
  {
    PROFILING_LINEAR_AMDAHL_PARALLEL_END
    PROFILING_LINEAR_END
  }

  }


#else // BATCHING


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
  data_t * __restrict__ outFeatures) //property(functional)
// {
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
  data_t * __restrict__ outFeatures) //property(functional)
#endif
{

#if defined(PROFILING_NEW) || defined(PROFILING)
  if(rt_core_id()==0)
  {
    PROFILING_LINEAR_START
    PROFILING_LINEAR_AMDAHL_SERIELL_START
  }
#endif

  // printf("weight: ");PrintTensor(outFeaturesSize, bias);
  // printf("input: ");PrintTensor(inFeaturesSize, inFeatures);
  // printf("%d %x %x %x %x\n", core_id, &outFeatures, &weight, &bias, &inFeatures);

#ifdef MULTICORE

#ifndef LSTM_ON

#ifdef EFFICIENT_CORE_ASSIGNMENT // or TILING_HARD

#if defined(PROFILING_NEW) || defined(PROFILING)
  if(rt_core_id()==0)
  {
    PROFILING_EFFICIENT_TILING_START
  }
#endif

  int core_id = rt_core_id();
  int start   = tile_size * core_id;
  int stop    = MIN(start + tile_size, outFeaturesSize);
  int chunck_final = (stop-start);

#if defined(PROFILING_NEW) || defined(PROFILING)
  if(rt_core_id()==0)
  {
  PROFILING_EFFICIENT_TILING_END
  }
#endif

#else // TILING_HARD // or EFFICIENT_CORE_ASSIGNMENT

#if defined(PROFILING_NEW) || defined(PROFILING)
  if(rt_core_id()==0)
  {
    PROFILING_TILING_START
  }
#endif
  /* instructions to parallelize the workload:
  each core computes a balanced number of neurons */
  int core_id = rt_core_id();
  int n_cores = NR_CORES;
  int chunck = 1;
  /* handle the case when number of neurons
  is less than number of cores: chunck=1 */
  if(outFeaturesSize < n_cores)
  {
      n_cores = outFeaturesSize;
      // n_cores = 1;
      // if (core_id == 0)
      // {
      //   chunck = outFeaturesSize;
      // } else
      // {
      //   chunck = 0;
      // }
  }
  else
  {
      int Log2Core = __builtin_pulp_fl1(n_cores);
      chunck = (outFeaturesSize >> Log2Core) + ((outFeaturesSize & (n_cores-1))!=0);
  }
  /* start and stop neuron to be computed, for each core */
  int start = MIN(chunck * core_id,outFeaturesSize);
  int stop = MIN(start + chunck, outFeaturesSize);
  int chunck_final = (stop-start);

#if defined(PROFILING_NEW) || defined(PROFILING)
  if(rt_core_id()==0)
  {
    PROFILING_TILING_END
  }
#endif

#endif // TILING_HARD or EFFICIENT_CORE_ASSIGNMENT

#else // LSTM_ON

  /* instructions to parallelize the workload:
  each core computes a balanced number of neurons */
  int core_id = rt_core_id();
  int n_cores = NR_CORES;
  int chunck = 1;
  int chunkg_orig=1;

  int start_offset = 0;
  /* handle the case when number of neurons
  is less than number of cores: chunck=1 */
  if(outFeaturesSize <= n_cores)
  {
      // n_cores = outFeaturesSize;
      n_cores = 1;
      if (core_id == 0)
      {
        chunck = outFeaturesSize;
      } else
      {
        chunck = 0;
      }
  }
  else
  {
      int Log2Core = __builtin_pulp_fl1(n_cores);
      chunck = (outFeaturesSize >> Log2Core) + ((outFeaturesSize & (n_cores-1))!=0);
      chunkg_orig = chunck;
      // printf(" core_id %d a\n",core_id);
      if ((chunck % 2)!=0)
      {
        // printf(" core_id %d b\n",core_id);
        if ((core_id%2)==0)
        {
          // printf(" core_id %d +\n",core_id);
          chunkg_orig = chunck;
          chunck = chunck+1;
        }
        else
        {
          // printf(" core_id %d -\n",core_id);
          chunkg_orig = chunck;
          chunck = chunck-1;
          start_offset = 1;
        }
      }
  }
  /* start and stop neuron to be computed, for each core */
  int start = MIN((chunkg_orig) * core_id+start_offset,outFeaturesSize);
  int stop = MIN(start + chunck, outFeaturesSize);
  int chunck_final = (stop-start);

#endif // LSTM_ON

#ifdef DEBUG_LSTM
  // printf("bla in default w outputtiling, core_id: %d, start: %d, stop: %d , chunk: %d, outFeaturesSize: %d, NR_CORES: %d \n", core_id, start, stop, chunck, outFeaturesSize, n_cores);
#endif // DEBUG_LSTM
#endif

  int inFeaturesSizeP2 = inFeaturesSize/2;
  int inFeaturesSizeP2_p1 = inFeaturesSizeP2 + W_OFFSET/2;
  #ifdef FMINTILING
  int inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
  #else
  int inFeaturesSizeP4 = inFeaturesSizeP2;   // no input FM tiling
  #endif

  #if OUTPUTBUFFER > 8
  int tileOptions[] = {OUTPUTBUFFER,8,4,2,1};
  #elif OUTPUTBUFFER > 4
  int tileOptions[] = {OUTPUTBUFFER,4,2,1};
  #else
  int tileOptions[] = {OUTPUTBUFFER,2,1};
  #endif

#ifdef MULTICORE
  data_t  * bias_ptr   = &bias[start];
  v2s     * weight_ptr = &((v2s*)weight)[start*inFeaturesSizeP2_p1];
  data_t  * outFeatures_ptr = &outFeatures[start];
  // printf("%d %x %x %x %x\n", core_id, &outFeatures_ptr, &weight_ptr, &bias_ptr, &inFeatures);
  int outFeaturesPerTile = 1;
#else
  data_t  * bias_ptr   = bias;
  v2s     * weight_ptr = (v2s*)weight;
  data_t  * outFeatures_ptr = outFeatures;
  int outFeaturesPerTile = 1;
#endif

  // register definition for manual loop unfolding
  register_attribute int32_t   temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9, temp10, temp11, temp12, temp13, temp14;
  register_attribute uint32_t  addr0, addr1, addr2, addr3, addr4, addr5, addr6, addr7, addr8, addr9, addr10, addr11, addr12, addr13, addr14;
  register_attribute uint32_t  in_addr;

  // register null
  register int x0 asm("x0");

#ifdef MULTICORE
  int outFeatureTiles;
  int outFeaturesSize_remain = chunck_final;
#else
  int outFeatureTiles;
  int outFeaturesSize_remain = outFeaturesSize;
#endif

#if defined(PROFILING_NEW) || defined(PROFILING)
  if(core_id==0)
  {
    PROFILING_LINEAR_AMDAHL_SERIELL_END
    PROFILING_LINEAR_AMDAHL_PARALLEL_START
  }
#endif
  // Tile with largest tileOption
  for(unsigned int i=0; i<sizeof(tileOptions)/sizeof(int); i++) {
    outFeaturesPerTile = tileOptions[i];
    outFeatureTiles = outFeaturesSize_remain/outFeaturesPerTile;

    if(outFeatureTiles == 0) continue;

    // Select Tile Size
    switch(outFeaturesPerTile) {
     #if OUTPUTBUFFER > 2
     case OUTPUTBUFFER:
     for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
     {
      // Manual loop unfolding
      // Inititalize accumulation registers with bias and shift accordingly
          #if OUTPUTBUFFER > 2
      temp0 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 3
      temp1 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+1]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 4
      temp2 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+2]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 5
      temp3 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+3]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 6
      temp4 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+4]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 7
      temp5 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+5]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 8
      temp6 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+6]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 9
      temp7 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+7]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 10
      temp8 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+8]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 11
      temp9 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+9]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 12
      temp10 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+10]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 13
      temp11 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+11]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 14
      temp12 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+12]<<(q_fraqP1);
          #endif
      temp13 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+OUTPUTBUFFER-2]<<(q_fraqP1);
      temp14 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+OUTPUTBUFFER-1]<<(q_fraqP1);

      addr0  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 0))];
      addr1  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 1))];
      addr2  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 2))];
      addr3  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 3))];
      addr4  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 4))];
      addr5  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 5))];
      addr6  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 6))];
      addr7  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 7))];
      addr8  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 8))];
      addr9  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 9))];
      addr10 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+10))];
      addr11 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+11))];
      addr12 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+12))];
      addr13 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+OUTPUTBUFFER-2))];
      addr14 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+OUTPUTBUFFER-1))];

          asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload first weight
          asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload first weight

          in_addr = (uint32_t) (((v2s*)inFeatures));
          for(int i=0; i<inFeaturesSizeP4; i++) {
              v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
              v2s inF_temp2;// = ((v2s*)inFeatures)[2*i+1];


              // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
              asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
  #ifdef FMINTILING
              asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
  #endif
              // MANUAL loop unfolding
               #if OUTPUTBUFFER > 2
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 3
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 4
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr4) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 5
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr5) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 6
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp4),  "+r" (addr6) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 7
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp5),  "+r" (addr7) : "r" (inF_temp) ); 
               #endif
               #if OUTPUTBUFFER > 9
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr8) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 9
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr9) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 10
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp8),  "+r" (addr10) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 11
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp9),  "+r" (addr11) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 12
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp10),  "+r" (addr12) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 13
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp11),  "+r" (addr13) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 14
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp12),  "+r" (addr14) : "r" (inF_temp) );
               #endif
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp13),  "+r" (addr0) : "r" (inF_temp) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp14),  "+r" (addr1) : "r" (inF_temp) );
// do it twice for FMINTILING
  #ifdef FMINTILING
               #if OUTPUTBUFFER > 2
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp2) );
               #endif
               #if OUTPUTBUFFER > 3
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp2) );
               #endif
               #if OUTPUTBUFFER > 4
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr4) : "r" (inF_temp2) );
               #endif
               #if OUTPUTBUFFER > 5
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr5) : "r" (inF_temp2) );
               #endif
               #if OUTPUTBUFFER > 6
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp4),  "+r" (addr6) : "r" (inF_temp2) );
               #endif
               #if OUTPUTBUFFER > 7
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp5),  "+r" (addr7) : "r" (inF_temp2) ); 
               #endif
               #if OUTPUTBUFFER > 9
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr8) : "r" (inF_temp2) );
               #endif
               #if OUTPUTBUFFER > 9
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr9) : "r" (inF_temp2) );
               #endif
               #if OUTPUTBUFFER > 10
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp8),  "+r" (addr10) : "r" (inF_temp2) );
               #endif
               #if OUTPUTBUFFER > 11
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp9),  "+r" (addr11) : "r" (inF_temp2) );
               #endif
               #if OUTPUTBUFFER > 12
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp10),  "+r" (addr12) : "r" (inF_temp2) );
               #endif
               #if OUTPUTBUFFER > 13
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp11),  "+r" (addr13) : "r" (inF_temp2) );
               #endif
               #if OUTPUTBUFFER > 14
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp12),  "+r" (addr14) : "r" (inF_temp2) );
               #endif
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp13),  "+r" (addr0) : "r" (inF_temp2) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp14),  "+r" (addr1) : "r" (inF_temp2) );
  #endif
            // }
            }
  #ifdef FMINTILING
          if(inFeaturesSizeP2%2==1) { // add contribution of left over input channel (input channels not multiple of 4)
               v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
               // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
               asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
               // asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
               
               // MANUAL loop unfolding
               // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
               // __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+o_rel)) + i], temp[o_rel]);
               #if OUTPUTBUFFER > 2
               asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 3
               asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 4
               asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr4) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 5
               asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr5) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 6
               asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp4),  "+r" (addr6) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 7
               asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp5),  "+r" (addr7) : "r" (inF_temp) ); 
               #endif
               #if OUTPUTBUFFER > 9
               asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr8) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 9
               asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr9) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 10
               asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp8),  "+r" (addr10) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 11
               asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp9),  "+r" (addr11) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 12
               asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp10),  "+r" (addr12) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 13
               asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp11),  "+r" (addr13) : "r" (inF_temp) );
               #endif
               #if OUTPUTBUFFER > 14
               asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp12),  "+r" (addr14) : "r" (inF_temp) );
               #endif
               asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp13),  "+r" (addr0) : "r" (inF_temp) );
               asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp14),  "+r" (addr1) : "r" (inF_temp) );
             }

  #endif

// Store the final results back to the memory
        // for(int o_rel =0;o_rel<outFeaturesPerTile;o_rel++) {
                #if OUTPUTBUFFER > 2
             outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = temp0>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 3
             outFeatures_ptr[(o_tile*outFeaturesPerTile+1)] = temp1>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 4
             outFeatures_ptr[(o_tile*outFeaturesPerTile+2)] = temp2>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 5
             outFeatures_ptr[(o_tile*outFeaturesPerTile+3)] = temp3>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 6
             outFeatures_ptr[(o_tile*outFeaturesPerTile+4)] = temp4>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 7
             outFeatures_ptr[(o_tile*outFeaturesPerTile+5)] = temp5>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 8
             outFeatures_ptr[(o_tile*outFeaturesPerTile+6)] = temp6>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 9
             outFeatures_ptr[(o_tile*outFeaturesPerTile+7)] = temp7>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 10
             outFeatures_ptr[(o_tile*outFeaturesPerTile+8)] = temp8>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 11
             outFeatures_ptr[(o_tile*outFeaturesPerTile+9)] = temp9>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 12
             outFeatures_ptr[(o_tile*outFeaturesPerTile+10)] = temp10>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 13
             outFeatures_ptr[(o_tile*outFeaturesPerTile+11)] = temp11>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 14
             outFeatures_ptr[(o_tile*outFeaturesPerTile+12)] = temp12>>(q_fraqP1);
                #endif
             outFeatures_ptr[(o_tile*outFeaturesPerTile+OUTPUTBUFFER-2)] = temp13>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+OUTPUTBUFFER-1)] = temp14>>(q_fraqP1);
                // printf("outFeatures[%i]=%i\n", o_rel, outFeatures[(o_tile*outFeaturesPerTile+o_rel)]);
        // }

           }
           break;
     #endif
     #if OUTPUTBUFFER > 8
           case 8:
           for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
           {

            temp0 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
            temp1 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+1]<<(q_fraqP1);
            temp2 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+2]<<(q_fraqP1);
            temp3 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+3]<<(q_fraqP1);
            temp4 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+4]<<(q_fraqP1);
            temp5 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+5]<<(q_fraqP1);
            temp6 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+6]<<(q_fraqP1);
            temp7 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+7]<<(q_fraqP1);

          // }
          // printf("wght=%i, addr before=%i (+%i)=%i\n", weight, addr0, inFeaturesSizeP2*outFeaturesPerTile, addr0+4*inFeaturesSizeP2*outFeaturesPerTile);
            addr0 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+0))];
            addr1 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+1))];
            addr2 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+2))];
            addr3 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+3))];
            addr4 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+4))];
            addr5 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+5))];
            addr6 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+6))];
            addr7 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+7))];


          asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload first weight
          asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload 2nd weight

          in_addr = (((v2s*)inFeatures));
          for(int i=0; i<inFeaturesSizeP4; i++) {
              v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
              v2s inF_temp2;// = ((v2s*)inFeatures)[2*i+1];

              // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
              asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
  #ifdef FMINTILING
              asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
  #endif
              // MANUAL loop unfolding
              // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
              // __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+o_rel)) + i], temp[o_rel]);
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp) );
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr4) : "r" (inF_temp) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr5) : "r" (inF_temp) );
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp4),  "+r" (addr6) : "r" (inF_temp) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp5),  "+r" (addr7) : "r" (inF_temp) ); 
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr0) : "r" (inF_temp) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr1) : "r" (inF_temp) );
  #ifdef FMINTILING
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp2) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp2) );
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr4) : "r" (inF_temp2) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr5) : "r" (inF_temp2) );
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp4),  "+r" (addr6) : "r" (inF_temp2) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp5),  "+r" (addr7) : "r" (inF_temp2) ); 
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr0) : "r" (inF_temp2) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr1) : "r" (inF_temp2) );
  #endif
            // }
            }
  #ifdef FMINTILING
          if(inFeaturesSizeP2%2==1) { // add contribution of left over input channel (input channels not multiple of 4)

               v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
               asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
               asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp) );
               asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp) );
               asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr4) : "r" (inF_temp) );
               asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr5) : "r" (inF_temp) );
               asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp4),  "+r" (addr6) : "r" (inF_temp) );
               asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp5),  "+r" (addr7) : "r" (inF_temp) ); 
               asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr0) : "r" (inF_temp) );
               asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr1) : "r" (inF_temp) );

             }
  #endif
        // for(int o_rel =0;o_rel<outFeaturesPerTile;o_rel++) {
             outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = temp0>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+1)] = temp1>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+2)] = temp2>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+3)] = temp3>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+4)] = temp4>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+5)] = temp5>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+6)] = temp6>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+7)] = temp7>>(q_fraqP1);
        // }
           }
           break;
     #endif
     #if OUTPUTBUFFER > 4
           case 4:
           for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
           {

            temp0 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
            temp1 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+1]<<(q_fraqP1);
            temp2 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+2]<<(q_fraqP1);
            temp3 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+3]<<(q_fraqP1);

          // }
            addr0 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+0))];
            addr1 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+1))];
            addr2 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+2))];
            addr3 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+3))];


          asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload first weight
          asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload first weight

          in_addr = (uint32_t)(((v2s*)inFeatures));
          for(int i=0; i<inFeaturesSizeP4; i++) {
              v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
              v2s inF_temp2;// = ((v2s*)inFeatures)[2*i+1];


              // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
              asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
  #ifdef FMINTILING
              asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
  #endif
              // MANUAL loop unfolding
              // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
              // __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+o_rel)) + i], temp[o_rel]);
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp) );
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr0) : "r" (inF_temp) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr1) : "r" (inF_temp) );
  #ifdef FMINTILING
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp2) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp2) );
              asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr0) : "r" (inF_temp2) );
              asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr1) : "r" (inF_temp2) );
  #endif
            // }
            }
  #ifdef FMINTILING
          if(inFeaturesSizeP2%2==1) { // add contribution of left over input channel (input channels not multiple of 4)

               v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
               asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
               asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp) );
               asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp) );
               asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr0) : "r" (inF_temp) );
               asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr1) : "r" (inF_temp) );

             }
  #endif
        // for(int o_rel =0;o_rel<outFeaturesPerTile;o_rel++) {
             outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = temp0>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+1)] = temp1>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+2)] = temp2>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+3)] = temp3>>(q_fraqP1);
               // printf("outFeatures[%i]=%i\n", o_rel, outFeatures[(o_tile*outFeaturesPerTile+o_rel)]);
        // }
           }
           break;
     #endif
           case 2:
           for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
           {

            temp0 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
            temp1 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+1]<<(q_fraqP1);
          // temp2 = (int32_t)bias[o_tile*outFeaturesPerTile+2]<<(q_fraqP1);
          // temp3 = (int32_t)bias[o_tile*outFeaturesPerTile+3]<<(q_fraqP1);
          // }
            addr0 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+0))];
            addr1 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+1))];
          // addr2 = &((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+2))];
          // addr3 = &((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+3))];
          asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload no compute
          asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload no compute
          for(int i=0; i<inFeaturesSizeP2; i++) {
            v2s inF_temp = ((v2s*)inFeatures)[i];
            
            // int o_rel;
            // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr0) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr1) : "r" (inF_temp) );
               // asm volatile("pl.sdotsp.h %0, %1, %2" : "+r" (temp2) : "r" (addr3), "r" (inF_temp) );
               // asm volatile("pl.sdotsp.h %0, %1, %2" : "+r" (temp3) : "r" (addr0), "r" (inF_temp) );
            // }
          }
        // for(int o_rel =0;o_rel<outFeaturesPerTile;o_rel++) {
          outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = temp0>>(q_fraqP1);
          outFeatures_ptr[(o_tile*outFeaturesPerTile+1)] = temp1>>(q_fraqP1);
                // outFeatures[(o_tile*outFeaturesPerTile+2)] = temp2>>(q_fraqP1);
                // outFeatures[(o_tile*outFeaturesPerTile+3)] = temp3>>(q_fraqP1);
                // printf("outFeatures[%i]=%i\n", o_rel, outFeatures[(o_tile*outFeaturesPerTile+o_rel)]);
        // }

        }
        break;
        case 1:
        for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
        {
          temp0 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
          for(int i=0; i<inFeaturesSizeP2; i++) {
            v2s inF_temp = ((v2s*)inFeatures)[i];
            temp0 = __SUMDOTP2(inF_temp, ((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile)) + i], temp0);
          }
          outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = temp0>>(q_fraqP1);
        }
        break;
      }

  // move pointers for next iteration
      bias_ptr                = &bias_ptr[outFeatureTiles*outFeaturesPerTile];
      weight_ptr              = &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(outFeatureTiles*outFeaturesPerTile))];
      outFeatures_ptr         = &outFeatures_ptr[(outFeatureTiles*outFeaturesPerTile)];
      outFeaturesSize_remain -= outFeatureTiles*outFeaturesPerTile;
      // printf("hallo %d %d\n", core_id, outFeaturesSize_remain);
      if (outFeaturesSize_remain==0) break;
    }
#if defined(PROFILING_NEW) || defined(PROFILING)
  if(core_id==0)
    {
      PROFILING_LINEAR_AMDAHL_PARALLEL_END
      PROFILING_LINEAR_END
    }
#endif
  }

#endif // BATCHING

// new implementation for output FM tiling and input FM tiling without VLIW!
#elif defined FMOUTTILING && !defined(ASIP) && defined MANUALLOOPUNFOLDING && !defined VLIWEXT 
/** @brief Calculates a Fully-Connected (or Linear Layer) 
 *  
 *  Calculates a fully conntected Layer with the custom VLIW instructions for load and MAC
 *  Supports the following configurations:
 *  INPUTFMTILING false/true with input tile size 2
 *  OUTPUTFMTILING false/true with output tile sizes 1,2,4,8,10,12,14 (odd are currently not
 *                 supported as the SPR would need to be switched)
 *  FixedPt and SIMD and MANUALLOOPUNFOLDING only
 *
 *  @param inFeaturesSize Number of input neurons
 *  @param outFeaturesSize Number of output neurons
 *  @param hasBias FC with bias or not?
 *  @param weight Pointer to weights
 *  @param bias Pointer to bias
 *  @param inFeatures Input Feature Map
 *  @param outFeatures Output Feature Map
 */
void NOINLINE LinearLayer (
        // Layer Attributes
  int inFeaturesSize, int outFeaturesSize,
  short hasBias,
        // Layer Parameters
  data_t * __restrict__ weight,
  data_t * __restrict__ bias,
        // Input and Output Features
  data_t * __restrict__ inFeatures,
        data_t * __restrict__ outFeatures) //property(functional)
{
  PROFILING_LINEAR_START

#ifdef MULTICORE
  /* instructions to parallelize the workload:
  each core computes a balanced number of neurons */
  int core_id = rt_core_id();
  int n_cores = NR_CORES;
  int chunck = 1;
  /* handle the case when number of neurons
  is less than number of cores: chunck=1 */
  if(outFeaturesSize < n_cores)
  {
      n_cores = 1;
      if (core_id == 0)
      {
        chunck = outFeaturesSize;
      } else
      {
        chunck = 0;
      }
  }
  else
  {
      int Log2Core = __builtin_pulp_fl1(n_cores);
      chunck = (outFeaturesSize >> Log2Core) + ((outFeaturesSize & (n_cores-1))!=0);
  }
  /* start and stop neuron to be computed, for each core */
  int start = MIN(chunck * core_id,outFeaturesSize);
  int stop = MIN(start + chunck, outFeaturesSize);
  int chunck_final = (stop-start);

#ifdef DEBUG_LSTM
  printf("in default w outputtiling, core_id: %d, start: %d, stop: %d , chunk: %d, outFeaturesSize: %d, NR_CORES: %d \n", core_id, start, stop, chunck, outFeaturesSize, n_cores);
#endif // DEBUG_LSTM
#endif

  int inFeaturesSizeP2 = inFeaturesSize/2;
  int inFeaturesSizeP2_p1 = inFeaturesSizeP2 + W_OFFSET/2;
  #ifdef FMINTILING
  int inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
  #else
  int inFeaturesSizeP4 = inFeaturesSizeP2;   // no input FM tiling
  #endif

  #if OUTPUTBUFFER > 8
  int tileOptions[] = {OUTPUTBUFFER,8,4,2,1};
  #elif OUTPUTBUFFER > 4
  int tileOptions[] = {OUTPUTBUFFER,4,2,1};
  #else
  int tileOptions[] = {OUTPUTBUFFER,2,1};
  #endif


#ifdef MULTICORE
  data_t  * bias_ptr   = &bias[start];
  v2s     * weight_ptr = &((v2s*)weight)[start*inFeaturesSizeP2_p1];
  data_t  * outFeatures_ptr = &outFeatures[start];
  int outFeaturesPerTile = 1;
#else
  data_t  * bias_ptr   = bias;
  v2s     * weight_ptr = (v2s*)weight;
  data_t  * outFeatures_ptr = outFeatures;
  int outFeaturesPerTile = 1;
#endif

  // register definition for manual loop unfolding
  register_attribute int32_t   temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9, temp10, temp11, temp12, temp13, temp14;
  register_attribute uint32_t  addr0, addr1, addr2, addr3, addr4, addr5, addr6, addr7, addr8, addr9, addr10, addr11, addr12, addr13, addr14;
  register_attribute uint32_t  in_addr;

  // register null
  register int x0 asm("x0");

#ifdef MULTICORE
  int outFeatureTiles;
  int outFeaturesSize_remain = chunck_final;
#else
  int outFeatureTiles;
  int outFeaturesSize_remain = outFeaturesSize;
#endif

  // Tile with largest tileOption
  for(unsigned int i=0; i<sizeof(tileOptions)/sizeof(int); i++) {
    outFeaturesPerTile = tileOptions[i];
    outFeatureTiles = outFeaturesSize_remain/outFeaturesPerTile;
    
    if(outFeatureTiles == 0) continue;

    // Select Tile Size
    switch(outFeaturesPerTile) {
     #if OUTPUTBUFFER > 2
     case OUTPUTBUFFER:
     for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
     {
      // Manual loop unfolding
      // Inititalize accumulation registers with bias and shift accordingly
          #if OUTPUTBUFFER > 2
      temp0 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 3
      temp1 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+1]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 4
      temp2 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+2]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 5
      temp3 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+3]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 6
      temp4 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+4]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 7
      temp5 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+5]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 8
      temp6 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+6]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 9
      temp7 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+7]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 10
      temp8 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+8]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 11
      temp9 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+9]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 12
      temp10 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+10]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 13
      temp11 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+11]<<(q_fraqP1);
          #endif
          #if OUTPUTBUFFER > 14
      temp12 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+12]<<(q_fraqP1);
          #endif
      temp13 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+OUTPUTBUFFER-2]<<(q_fraqP1);
      temp14 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+OUTPUTBUFFER-1]<<(q_fraqP1);

      addr0  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 0))];
      addr1  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 1))];
      addr2  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 2))];
      addr3  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 3))];
      addr4  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 4))];
      addr5  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 5))];
      addr6  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 6))];
      addr7  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 7))];
      addr8  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 8))];
      addr9  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+ 9))];
      addr10 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+10))];
      addr11 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+11))];
      addr12 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+12))];
      addr13 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+OUTPUTBUFFER-2))];
      addr14 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+OUTPUTBUFFER-1))];

          // asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload first weight
          // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload first weight

          in_addr = (uint32_t)(((v2s*)inFeatures));
          for(int i=0; i<inFeaturesSizeP4; i++) {
              v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
              v2s inF_temp2;// = ((v2s*)inFeatures)[2*i+1];


              // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
              asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
  #ifdef FMINTILING
              asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
  #endif
              // MANUAL loop unfolding
               #if OUTPUTBUFFER > 2
              SDOTP_GENERIC(temp0, ((v2s*)addr0)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 3
              SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 4
              SDOTP_GENERIC(temp2, ((v2s*)addr2)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 5
              SDOTP_GENERIC(temp3, ((v2s*)addr3)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 6
              SDOTP_GENERIC(temp4, ((v2s*)addr4)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 7
              SDOTP_GENERIC(temp5, ((v2s*)addr5)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 9
              SDOTP_GENERIC(temp6, ((v2s*)addr6)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 9
              SDOTP_GENERIC(temp7, ((v2s*)addr7)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 10
              SDOTP_GENERIC(temp8, ((v2s*)addr8)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 11
              SDOTP_GENERIC(temp9, ((v2s*)addr9)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 12
              SDOTP_GENERIC(temp10, ((v2s*)addr10)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 13
              SDOTP_GENERIC(temp11, ((v2s*)addr11)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 14
              SDOTP_GENERIC(temp12, ((v2s*)addr12)[i], inF_temp);
               #endif
              SDOTP_GENERIC(temp13, ((v2s*)addr13)[i],inF_temp);
              SDOTP_GENERIC(temp14, ((v2s*)addr14)[i],inF_temp);
// do it twice for FMINTILING
  #ifdef FMINTILING
                #if OUTPUTBUFFER > 2
              SDOTP_GENERIC(temp0, ((v2s*)addr0)[i], inF_temp2);
               #endif
               #if OUTPUTBUFFER > 3
              SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp2);
               #endif
               #if OUTPUTBUFFER > 4
              SDOTP_GENERIC(temp2, ((v2s*)addr2)[i], inF_temp2);
               #endif
               #if OUTPUTBUFFER > 5
              SDOTP_GENERIC(temp3, ((v2s*)addr3)[i], inF_temp2);
               #endif
               #if OUTPUTBUFFER > 6
              SDOTP_GENERIC(temp4, ((v2s*)addr4)[i], inF_temp2);
               #endif
               #if OUTPUTBUFFER > 7
              SDOTP_GENERIC(temp5, ((v2s*)addr5)[i], inF_temp2);
               #endif
               #if OUTPUTBUFFER > 9
              SDOTP_GENERIC(temp6, ((v2s*)addr6)[i], inF_temp2);
               #endif
               #if OUTPUTBUFFER > 9
              SDOTP_GENERIC(temp7, ((v2s*)addr7)[i], inF_temp2);
               #endif
               #if OUTPUTBUFFER > 10
              SDOTP_GENERIC(temp8, ((v2s*)addr8)[i], inF_temp2);
               #endif
               #if OUTPUTBUFFER > 11
              SDOTP_GENERIC(temp9, ((v2s*)addr9)[i], inF_temp2);
               #endif
               #if OUTPUTBUFFER > 12
              SDOTP_GENERIC(temp10, ((v2s*)addr10)[i], inF_temp2);
               #endif
               #if OUTPUTBUFFER > 13
              SDOTP_GENERIC(temp11, ((v2s*)addr11)[i], inF_temp2);
               #endif
               #if OUTPUTBUFFER > 14
              SDOTP_GENERIC(temp12, ((v2s*)addr12)[i], inF_temp2);
               #endif
              SDOTP_GENERIC(temp13, ((v2s*)addr13)[i],inF_temp2);
              SDOTP_GENERIC(temp14, ((v2s*)addr14)[i],inF_temp2);
  #endif
            // }
            }
  #ifdef FMINTILING
          if(inFeaturesSizeP2%2==1) { // add contribution of left over input channel (input channels not multiple of 4)
               v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
               // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
               asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
               // asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
               
               // MANUAL loop unfolding
               // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
               // __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+o_rel)) + i], temp[o_rel]);
             #if OUTPUTBUFFER > 2
              SDOTP_GENERIC(temp0, ((v2s*)addr0)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 3
              SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 4
              SDOTP_GENERIC(temp2, ((v2s*)addr2)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 5
              SDOTP_GENERIC(temp3, ((v2s*)addr3)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 6
              SDOTP_GENERIC(temp4, ((v2s*)addr4)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 7
              SDOTP_GENERIC(temp5, ((v2s*)addr5)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 9
              SDOTP_GENERIC(temp6, ((v2s*)addr6)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 9
              SDOTP_GENERIC(temp7, ((v2s*)addr7)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 10
              SDOTP_GENERIC(temp8, ((v2s*)addr8)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 11
              SDOTP_GENERIC(temp9, ((v2s*)addr9)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 12
              SDOTP_GENERIC(temp10, ((v2s*)addr10)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 13
              SDOTP_GENERIC(temp11, ((v2s*)addr11)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 14
              SDOTP_GENERIC(temp12, ((v2s*)addr12)[i], inF_temp);
               #endif
              SDOTP_GENERIC(temp13, ((v2s*)addr13)[i],inF_temp);
              SDOTP_GENERIC(temp14, ((v2s*)addr14)[i],inF_temp);
             }

  #endif

// Store the final results back to the memory
        // for(int o_rel =0;o_rel<outFeaturesPerTile;o_rel++) {
                #if OUTPUTBUFFER > 2
             outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = temp0>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 3
             outFeatures_ptr[(o_tile*outFeaturesPerTile+1)] = temp1>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 4
             outFeatures_ptr[(o_tile*outFeaturesPerTile+2)] = temp2>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 5
             outFeatures_ptr[(o_tile*outFeaturesPerTile+3)] = temp3>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 6
             outFeatures_ptr[(o_tile*outFeaturesPerTile+4)] = temp4>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 7
             outFeatures_ptr[(o_tile*outFeaturesPerTile+5)] = temp5>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 8
             outFeatures_ptr[(o_tile*outFeaturesPerTile+6)] = temp6>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 9
             outFeatures_ptr[(o_tile*outFeaturesPerTile+7)] = temp7>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 10
             outFeatures_ptr[(o_tile*outFeaturesPerTile+8)] = temp8>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 11
             outFeatures_ptr[(o_tile*outFeaturesPerTile+9)] = temp9>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 12
             outFeatures_ptr[(o_tile*outFeaturesPerTile+10)] = temp10>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 13
             outFeatures_ptr[(o_tile*outFeaturesPerTile+11)] = temp11>>(q_fraqP1);
                #endif
                #if OUTPUTBUFFER > 14
             outFeatures_ptr[(o_tile*outFeaturesPerTile+12)] = temp12>>(q_fraqP1);
                #endif
             outFeatures_ptr[(o_tile*outFeaturesPerTile+OUTPUTBUFFER-2)] = temp13>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+OUTPUTBUFFER-1)] = temp14>>(q_fraqP1);
                // printf("outFeatures[%i]=%i\n", o_rel, outFeatures[(o_tile*outFeaturesPerTile+o_rel)]);
        // }

           }
           break;
     #endif
     #if OUTPUTBUFFER > 8
           case 8:
           for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
           {

            temp0 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
            temp1 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+1]<<(q_fraqP1);
            temp2 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+2]<<(q_fraqP1);
            temp3 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+3]<<(q_fraqP1);
            temp4 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+4]<<(q_fraqP1);
            temp5 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+5]<<(q_fraqP1);
            temp6 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+6]<<(q_fraqP1);
            temp7 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+7]<<(q_fraqP1);

          // }
          // printf("wght=%i, addr before=%i (+%i)=%i\n", weight, addr0, inFeaturesSizeP2*outFeaturesPerTile, addr0+4*inFeaturesSizeP2*outFeaturesPerTile);
            addr0 = &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+0))];
            addr1 = &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+1))];
            addr2 = &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+2))];
            addr3 = &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+3))];
            addr4 = &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+4))];
            addr5 = &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+5))];
            addr6 = &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+6))];
            addr7 = &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+7))];


          // asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload first weight
          // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload 2nd weight

          in_addr = (((v2s*)inFeatures));
          for(int i=0; i<inFeaturesSizeP4; i++) {
              v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
              v2s inF_temp2;// = ((v2s*)inFeatures)[2*i+1];

              // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
              asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
  #ifdef FMINTILING
              asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
  #endif
              // MANUAL loop unfolding
              // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
              // __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+o_rel)) + i], temp[o_rel]);
              SDOTP_GENERIC(temp0, ((v2s *)addr0)[i], inF_temp) ;
              SDOTP_GENERIC(temp1, ((v2s *)addr1)[i], inF_temp) ;
              SDOTP_GENERIC(temp2, ((v2s *)addr2)[i], inF_temp) ;
              SDOTP_GENERIC(temp3, ((v2s *)addr3)[i], inF_temp) ;
              SDOTP_GENERIC(temp4, ((v2s *)addr4)[i], inF_temp) ;
              SDOTP_GENERIC(temp5, ((v2s *)addr5)[i], inF_temp) ;
              SDOTP_GENERIC(temp6, ((v2s *)addr6)[i], inF_temp) ;
              SDOTP_GENERIC(temp7, ((v2s *)addr7)[i], inF_temp) ;
  #ifdef FMINTILING
              SDOTP_GENERIC(temp0, ((v2s *)addr0)[i], inF_temp2);
              SDOTP_GENERIC(temp1, ((v2s *)addr1)[i], inF_temp2);
              SDOTP_GENERIC(temp2, ((v2s *)addr2)[i], inF_temp2);
              SDOTP_GENERIC(temp3, ((v2s *)addr3)[i], inF_temp2);
              SDOTP_GENERIC(temp4, ((v2s *)addr4)[i], inF_temp2);
              SDOTP_GENERIC(temp5, ((v2s *)addr5)[i], inF_temp2);
              SDOTP_GENERIC(temp6, ((v2s *)addr6)[i], inF_temp2);
              SDOTP_GENERIC(temp7, ((v2s *)addr7)[i], inF_temp2);
  #endif
            // }
            }
  #ifdef FMINTILING
          if(inFeaturesSizeP2%2==1) { // add contribution of left over input channel (input channels not multiple of 4)
               v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
               asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
              SDOTP_GENERIC(temp0, ((v2s *)addr0)[i], inF_temp) ;
              SDOTP_GENERIC(temp1, ((v2s *)addr1)[i], inF_temp) ;
              SDOTP_GENERIC(temp2, ((v2s *)addr2)[i], inF_temp) ;
              SDOTP_GENERIC(temp3, ((v2s *)addr3)[i], inF_temp) ;
              SDOTP_GENERIC(temp4, ((v2s *)addr4)[i], inF_temp) ;
              SDOTP_GENERIC(temp5, ((v2s *)addr5)[i], inF_temp) ;
              SDOTP_GENERIC(temp6, ((v2s *)addr6)[i], inF_temp) ;
              SDOTP_GENERIC(temp7, ((v2s *)addr7)[i], inF_temp) ;
  #endif
        // for(int o_rel =0;o_rel<outFeaturesPerTile;o_rel++) {
             outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = temp0>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+1)] = temp1>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+2)] = temp2>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+3)] = temp3>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+4)] = temp4>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+5)] = temp5>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+6)] = temp6>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+7)] = temp7>>(q_fraqP1);
        // }
           }
           break;
     #endif
     #if OUTPUTBUFFER > 4
           case 4:
           for (int o_tile=0; o_tile< outFeatureTiles; o_tile++)
           {

            temp0 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
            temp1 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+1]<<(q_fraqP1);
            temp2 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+2]<<(q_fraqP1);
            temp3 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+3]<<(q_fraqP1);

          // }
            addr0 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+0))];
            addr1 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+1))];
            addr2 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+2))];
            addr3 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+3))];


          // asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload first weight
          // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload first weight

          in_addr = (uint32_t) (((v2s*)inFeatures));
          for(int i=0; i<inFeaturesSizeP4; i++) {
              v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
              v2s inF_temp2;// = ((v2s*)inFeatures)[2*i+1];


              // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
              asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
  #ifdef FMINTILING
              asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
  #endif
              // MANUAL loop unfolding
              // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
              // __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+o_rel)) + i], temp[o_rel]);
              SDOTP_GENERIC(temp0, ((v2s *)addr0)[i], inF_temp) ;
              SDOTP_GENERIC(temp1, ((v2s *)addr1)[i], inF_temp) ;
              SDOTP_GENERIC(temp2, ((v2s *)addr2)[i], inF_temp) ;
              SDOTP_GENERIC(temp3, ((v2s *)addr3)[i], inF_temp) ;
  #ifdef FMINTILING
              SDOTP_GENERIC(temp0, ((v2s *)addr0)[i], inF_temp2);
              SDOTP_GENERIC(temp1, ((v2s *)addr1)[i], inF_temp2);
              SDOTP_GENERIC(temp2, ((v2s *)addr2)[i], inF_temp2);
              SDOTP_GENERIC(temp3, ((v2s *)addr3)[i], inF_temp2);
  #endif
            // }
            }
  #ifdef FMINTILING
          if(inFeaturesSizeP2%2==1) { // add contribution of left over input channel (input channels not multiple of 4)
               v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
               asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
              SDOTP_GENERIC(temp0, ((v2s *)addr0)[i], inF_temp) ;
              SDOTP_GENERIC(temp1, ((v2s *)addr1)[i], inF_temp) ;
              SDOTP_GENERIC(temp2, ((v2s *)addr2)[i], inF_temp) ;
              SDOTP_GENERIC(temp3, ((v2s *)addr3)[i], inF_temp) ;

             }
  #endif
        // for(int o_rel =0;o_rel<outFeaturesPerTile;o_rel++) {
             outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = temp0>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+1)] = temp1>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+2)] = temp2>>(q_fraqP1);
             outFeatures_ptr[(o_tile*outFeaturesPerTile+3)] = temp3>>(q_fraqP1);
               // printf("outFeatures[%i]=%i\n", o_rel, outFeatures[(o_tile*outFeaturesPerTile+o_rel)]);
        // }
           }
           break;
     #endif
           case 2:
           for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
           {

            temp0 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
            temp1 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+1]<<(q_fraqP1);
          // temp2 = (int32_t)bias[o_tile*outFeaturesPerTile+2]<<(q_fraqP1);
          // temp3 = (int32_t)bias[o_tile*outFeaturesPerTile+3]<<(q_fraqP1);
          // }
            addr0 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+0))];
            addr1 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+1))];
          // addr2 = &((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+2))];
          // addr3 = &((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+3))];
          // asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload no compute
          // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload no compute
          for(int i=0; i<inFeaturesSizeP2; i++) {
            v2s inF_temp = ((v2s*)inFeatures)[i];
            
            // int o_rel;
            // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
              SDOTP_GENERIC(temp0, ((v2s*)addr0)[i], inF_temp) ;
              SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp) ;
               // asm volatile("pl.sdotsp.h %0, %1, %2" : "+r" (temp2) : "r" (addr3), "r" (inF_temp) );
               // asm volatile("pl.sdotsp.h %0, %1, %2" : "+r" (temp3) : "r" (addr0), "r" (inF_temp) );
            // }
          }
        // for(int o_rel =0;o_rel<outFeaturesPerTile;o_rel++) {
          outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = temp0>>(q_fraqP1);
          outFeatures_ptr[(o_tile*outFeaturesPerTile+1)] = temp1>>(q_fraqP1);
                // outFeatures[(o_tile*outFeaturesPerTile+2)] = temp2>>(q_fraqP1);
                // outFeatures[(o_tile*outFeaturesPerTile+3)] = temp3>>(q_fraqP1);
                // printf("outFeatures[%i]=%i\n", o_rel, outFeatures[(o_tile*outFeaturesPerTile+o_rel)]);
        // }

        }
        break;
        case 1:
// #ifdef MULTICORE
//         for (int o_tile=start; o_tile<stop; o_tile++) 
// #else
        for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
// #endif
        {
          temp0 = (int32_t)bias_ptr[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
          for(int i=0; i<inFeaturesSizeP2; i++) {
            v2s inF_temp = ((v2s*)inFeatures)[i];
            SDOTP_GENERIC(temp0, ((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile)) + i], inF_temp) ;
          }
          outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = temp0>>(q_fraqP1);
        }
        break;
      }

  // move pointers for next iteration
      bias_ptr                = &bias_ptr[outFeatureTiles*outFeaturesPerTile];
      weight_ptr              = &((v2s*)weight_ptr)[(inFeaturesSizeP2_p1*(outFeatureTiles*outFeaturesPerTile))];
      outFeatures_ptr         = &outFeatures_ptr[(outFeatureTiles*outFeaturesPerTile)];
      outFeaturesSize_remain -= outFeatureTiles*outFeaturesPerTile;
      if (outFeaturesSize_remain==0) break;
    }


    PROFILING_LINEAR_END
  }
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// _______ _______ _____  _____       _  _  _ _____ _______ _     _      _______ _______       _____  _     _ _______      _______ _____        _____ __   _  ///
// |_____| |______   |   |_____]      |  |  |   |      |    |_____|      |______ |  |  |      |     | |     |    |            |      |   |        |   | \  |  ///
// |     | ______| __|__ |            |__|__| __|__    |    |     |      |       |  |  |      |_____| |_____|    |            |    __|__ |_____ __|__ |  \_|  ///
//                                                                                                                                                            ///
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#elif defined ASIP && defined FMOUTTILING // LinearLayer Implementation for the ASIP tool and 
/** @brief Calculates a Fully-Connected (or Linear Layer) 
 *  
 *  Calculates a fully conntected Layer with for the ASIP designer (extended tzscale)
 *  Supports the following configurations:
 *  INPUTFMTILING false (no input FM tiling support as not needed)
 *  OUTPUTFMTILING false/true with output tile sizes 1,2,4,8,10
 *  MANUALLOOPUNFOLDING false/true
 *  FixedPt and SIMD and  only
 *
 *  @param inFeaturesSize Number of input neurons
 *  @param outFeaturesSize Number of output neurons
 *  @param hasBias FC with bias or not?
 *  @param weight Pointer to weights
 *  @param bias Pointer to bias
 *  @param inFeatures Input Feature Map
 *  @param outFeatures Output Feature Map
 */
  void NOINLINE LinearLayer (
        // Layer Attributes
    int inFeaturesSize, int outFeaturesSize,
    short hasBias,
        // Layer Parameters
    data_t * __restrict__ weight,
    data_t * __restrict__ bias,
        // Input and Output Features
    data_t * __restrict__ inFeatures,
        data_t * __restrict__ outFeatures) //property(functional)
  {
    PROFILING_LINEAR_START

    int inFeaturesSizeP2 = inFeaturesSize/2;

//  _  __    _                                                               _   
// / | \ \  | | ___   ___  _ __     _____   _____ _ __    ___     ___  _   _| |_ 
// | |  | | | |/ _ \ / _ \| '_ \   / _ \ \ / / _ \ '__|  / __|   / _ \| | | | __|
// | |  | | | | (_) | (_) | |_) | | (_) \ V /  __/ |    | (__   | (_) | |_| | |_ 
// |_|  | | |_|\___/ \___/| .__/   \___/ \_/ \___|_|     \___|__ \___/ \__,_|\__|
//     /_/                |_|                                |__|              

//----------------------------------------------------------------
// 1a) Tile Output Channels
//----------------------------------------------------------------

    int tileOptions[] = {10,8,4,2, 1};
    int outFeaturesPerTile = 1;
    for(unsigned int i=0; i<sizeof(tileOptions)/sizeof(int); i++) {
     if(outFeaturesSize % tileOptions[i] == 0) {
      outFeaturesPerTile = tileOptions[i];
      break;
    }
  }

  int outFeatureTiles = outFeaturesSize/outFeaturesPerTile;


  switch(outFeaturesPerTile) {
   case 10:
   for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) {
    int32_t register_attribute ra;int32_t register_attribute rb;int32_t register_attribute rc;int32_t register_attribute rd;
    int32_t register_attribute ra2;int32_t register_attribute rb2;int32_t register_attribute rc2;int32_t register_attribute rd2;
    int32_t register_attribute rc3;int32_t register_attribute rd3;
    ra = (int32_t)bias[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
    rb = (int32_t)bias[o_tile*outFeaturesPerTile+1]<<(q_fraqP1);
    rc = (int32_t)bias[o_tile*outFeaturesPerTile+2]<<(q_fraqP1);
    rd = (int32_t)bias[o_tile*outFeaturesPerTile+3]<<(q_fraqP1);
    ra2 = (int32_t)bias[o_tile*outFeaturesPerTile+4]<<(q_fraqP1);
    rb2 = (int32_t)bias[o_tile*outFeaturesPerTile+5]<<(q_fraqP1);
    rc2 = (int32_t)bias[o_tile*outFeaturesPerTile+6]<<(q_fraqP1);
    rd2 = (int32_t)bias[o_tile*outFeaturesPerTile+7]<<(q_fraqP1);
    rc3 = (int32_t)bias[o_tile*outFeaturesPerTile+8]<<(q_fraqP1);
    rd3 = (int32_t)bias[o_tile*outFeaturesPerTile+9]<<(q_fraqP1);


    for(int i=0; i<inFeaturesSizeP2; i++) { 
     v2s inF_temp = ((v2s*)inFeatures)[i];                                                          

               ra=ext_dotp_(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+0)) + i], ra); // lwinc x23, 0(x20)
               rb = rb + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+1)) + i]);      // lwinc x25, 0(x11); sdotp x7, x23, x25 
               rc = rc + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+2)) + i]);      // lwinc x24, 0(x11); sdotp x7, x23, x24 
               rd = rd + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+3)) + i]);      // lwinc xA, 0(xB); sdotp xC, x23, xB
               ra2 = ra2 + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+4)) + i]);    // lwinc xA, 0(xB); sdotp xC, x23, xB
               rb2 = rb2 + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+5)) + i]);    // lwinc xA, 0(xB); sdotp xC, x23, xB
               rc2 = rc2 + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+6)) + i]);    // lwinc xA, 0(xB); sdotp xC, x23, xB
               rd2 = rd2 + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+7)) + i]);    // lwinc xA, 0(xB); sdotp xC, x23, xB
               rc3 = rc3 + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+8)) + i]);    // lwinc xA, 0(xB); sdotp xC, x23, xB
               rd3 = rd3 + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+9)) + i]);    // lwinc xA, 0(xB); sdotp xC, x23, xB
             }
             outFeatures[(o_tile*outFeaturesPerTile+0)] = ra>>(q_fraqP1);
             outFeatures[(o_tile*outFeaturesPerTile+1)] = rb>>(q_fraqP1);
             outFeatures[(o_tile*outFeaturesPerTile+2)] = rc>>(q_fraqP1);
             outFeatures[(o_tile*outFeaturesPerTile+3)] = rd>>(q_fraqP1);
             outFeatures[(o_tile*outFeaturesPerTile+4)] = ra2>>(q_fraqP1);
             outFeatures[(o_tile*outFeaturesPerTile+5)] = rb2>>(q_fraqP1);
             outFeatures[(o_tile*outFeaturesPerTile+6)] = rc2>>(q_fraqP1);
             outFeatures[(o_tile*outFeaturesPerTile+7)] = rd2>>(q_fraqP1);
             outFeatures[(o_tile*outFeaturesPerTile+8)] = rc3>>(q_fraqP1);
             outFeatures[(o_tile*outFeaturesPerTile+9)] = rd3>>(q_fraqP1);


   } // for (int o_tile=0; o_tile< outFeatureTiles; o_tile++)
   break;

   case 8:
   for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) {
    int32_t register_attribute ra;int32_t register_attribute rb;int32_t register_attribute rc;int32_t register_attribute rd;
    int32_t register_attribute ra2;int32_t register_attribute rb2;int32_t register_attribute rc2;int32_t register_attribute rd2;
    ra = (int32_t)bias[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
    rb = (int32_t)bias[o_tile*outFeaturesPerTile+1]<<(q_fraqP1);
    rc = (int32_t)bias[o_tile*outFeaturesPerTile+2]<<(q_fraqP1);
    rd = (int32_t)bias[o_tile*outFeaturesPerTile+3]<<(q_fraqP1);
    ra2 = (int32_t)bias[o_tile*outFeaturesPerTile+4]<<(q_fraqP1);
    rb2 = (int32_t)bias[o_tile*outFeaturesPerTile+5]<<(q_fraqP1);
    rc2 = (int32_t)bias[o_tile*outFeaturesPerTile+6]<<(q_fraqP1);
    rd2 = (int32_t)bias[o_tile*outFeaturesPerTile+7]<<(q_fraqP1);
    for(int i=0; i<inFeaturesSizeP2; i++) { 
     v2s inF_temp = ((v2s*)inFeatures)[i];
     ra = ra + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+0)) + i]);
     rb = rb + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+1)) + i]);
     rc = rc + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+2)) + i]);
     rd = rd + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+3)) + i]);
     ra2 = ra2 + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+4)) + i]);
     rb2 = rb2 + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+5)) + i]);
     rc2 = rc2 + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+6)) + i]);
     rd2 = rd2 + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+7)) + i]);
      } // for(int i=0; i<inFeaturesSizeP2; i++)
      outFeatures[(o_tile*outFeaturesPerTile+0)] = ra>>(q_fraqP1);
      outFeatures[(o_tile*outFeaturesPerTile+1)] = rb>>(q_fraqP1);
      outFeatures[(o_tile*outFeaturesPerTile+2)] = rc>>(q_fraqP1);
      outFeatures[(o_tile*outFeaturesPerTile+3)] = rd>>(q_fraqP1);
      outFeatures[(o_tile*outFeaturesPerTile+4)] = ra2>>(q_fraqP1);
      outFeatures[(o_tile*outFeaturesPerTile+5)] = rb2>>(q_fraqP1);
      outFeatures[(o_tile*outFeaturesPerTile+6)] = rc2>>(q_fraqP1);
      outFeatures[(o_tile*outFeaturesPerTile+7)] = rd2>>(q_fraqP1);
   } // for (int o_tile=0; o_tile< outFeatureTiles; o_tile++)
   break;
   case 4:
   for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) {
    int32_t register_attribute ra;int32_t register_attribute rb;int32_t register_attribute rc;int32_t register_attribute rd;
    ra = (int32_t)bias[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
    rb = (int32_t)bias[o_tile*outFeaturesPerTile+1]<<(q_fraqP1);
    rc = (int32_t)bias[o_tile*outFeaturesPerTile+2]<<(q_fraqP1);
    rd = (int32_t)bias[o_tile*outFeaturesPerTile+3]<<(q_fraqP1);
    for(int i=0; i<inFeaturesSizeP2; i++) { 
     v2s inF_temp = ((v2s*)inFeatures)[i];
     ra = ra + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+0)) + i]);
     rb = rb + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+1)) + i]);
     rc = rc + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+2)) + i]);
     rd = rd + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+3)) + i]);
      } // for(int i=0; i<inFeaturesSizeP2; i++)
      outFeatures[(o_tile*outFeaturesPerTile+0)] = ra>>(q_fraqP1);
      outFeatures[(o_tile*outFeaturesPerTile+1)] = rb>>(q_fraqP1);
      outFeatures[(o_tile*outFeaturesPerTile+2)] = rc>>(q_fraqP1);
      outFeatures[(o_tile*outFeaturesPerTile+3)] = rd>>(q_fraqP1);
   } // for (int o_tile=0; o_tile< outFeatureTiles; o_tile++)
   break;
 case 2: // HOWTO duplicate and comment out not needed lines
 for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) {
      int32_t register_attribute ra;int32_t register_attribute rb;//int32_t register_attribute rc;int32_t register_attribute rd;
      ra = (int32_t)bias[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
      rb = (int32_t)bias[o_tile*outFeaturesPerTile+1]<<(q_fraqP1);
      // rc = (int32_t)bias[o_tile*outFeaturesPerTile+2]<<(q_fraqP1);
      // rd = (int32_t)bias[o_tile*outFeaturesPerTile+3]<<(q_fraqP1);
      for(int i=0; i<inFeaturesSizeP2; i++) { 
       v2s inF_temp = ((v2s*)inFeatures)[i];
       ra = ra + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+0)) + i]);
       rb = rb + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+1)) + i]);
               // rc = rc + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+2)) + i]);
               // rd = rd + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+3)) + i]);
      } // }
       // for(int i=0; i<inFeaturesSizeP2; i++)
      outFeatures[(o_tile*outFeaturesPerTile+0)] = ra>>(q_fraqP1);
      outFeatures[(o_tile*outFeaturesPerTile+1)] = rb>>(q_fraqP1);
      // outFeatures[(o_tile*outFeaturesPerTile+2)] = rc>>(q_fraqP1);
      // outFeatures[(o_tile*outFeaturesPerTile+3)] = rd>>(q_fraqP1);
   } // for (int o_tile=0; o_tile< outFeatureTiles; o_tile++)
   case 1: // HOWTO duplicate and comment out not needed lines
   for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) {
    int32_t register_attribute ra;
    ra = (int32_t)bias[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);

    for(int i=0; i<inFeaturesSizeP2; i++) chess_loop_range(1,) { 

     v2s inF_temp = ((v2s*)inFeatures)[i];
     ra = ra + (inF_temp * ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+0)) + i]);
   } 
   outFeatures[(o_tile*outFeaturesPerTile+0)] = ra>>(q_fraqP1);
   } // for (int o_tile=0; o_tile< outFeatureTiles; o_tile++)
   break;

}// switch outFeaturesPerTile


}

 ///////////////////////////////////////////////////////////////////////////////////////////////
 // ______  _______ _______ _______ _     _        _______      _____ _______  _____          //
 // |     \ |______ |______ |_____| |     | |         |           |   |  |  | |_____] |       //
 // |_____/ |______ |       |     | |_____| |_____    |         __|__ |  |  | |       |_____  //
 //                                                                                           //
 ///////////////////////////////////////////////////////////////////////////////////////////////
#else // any other case








// GIANNA: FROM HERE COMMENTED OUT

/** @brief Calculates a Fully-Connected (or Linear Layer) 
 *  
 *  Calculates a fully conntected Layer (standard implementation)
 *  Supports the following configurations:
 *  => (PULP-RISCY && !VLIWEXT || ASIP && !FMOUTTILING) && !FMINTILING
 *  => FMOUTINILING false (no input FM tiling support as not needed)
 *  => FMOUTTILING false/true with output tile sizes 1,2,4,8,10
 *  => FixedPt, SIMD || FLOAT
 *  => MANUALLOOPUNFOLDING not supported for non-SIMD
 *
 *  @param inFeaturesSize Number of input neurons
 *  @param outFeaturesSize Number of output neurons
 *  @param hasBias FC with bias or not?
 *  @param weight Pointer to weights
 *  @param bias Pointer to bias
 *  @param inFeatures Input Feature Map
 *  @param outFeatures Output Feature Map
 */
void NOINLINE LinearLayer ( 
  // Layer Attributes
  int inFeaturesSize, int outFeaturesSize,
  short hasBias,
  // Layer Parameters
  data_t * __restrict__ weight,
  data_t * __restrict__ bias,
  // Input and Output Features
  data_t * __restrict__ inFeatures,
  data_t * __restrict__ outFeatures) //property(functional)
{

    PROFILING_LINEAR_START

#ifdef MULTICORE
    /* instructions to parallelize the workload:
    each core computes a balanced number of neurons */
    int core_id = rt_core_id();
    int n_cores = NR_CORES;
    int chunck = 1;
    /* handle the case when number of neurons
    is less than number of cores: chunck=1 */
    if(outFeaturesSize < n_cores)
    {
      n_cores = outFeaturesSize;
      // n_cores = 1;
      // if (core_id == 0)
      // {
      //   chunck = outFeaturesSize;
      // } else
      // {
      //   chunck = 0;
      // }
    }
    else
    {
        int Log2Core = __builtin_pulp_fl1(n_cores);
        chunck = (outFeaturesSize >> Log2Core) + ((outFeaturesSize & (n_cores-1))!=0);
    }
    /* start and stop neuron to be computed, for each core */
    int start = MIN(chunck * core_id,outFeaturesSize);
    int stop = MIN(start + chunck, outFeaturesSize);
    int a = ((stop-start)>>1);

#ifdef DEBUG_LSTM
    printf("in default, core_id: %d, start: %d, stop: %d , chunk: %d, outFeaturesSize: %d, NR_CORES: %d \n", core_id, start, stop, chunck, outFeaturesSize, n_cores);
#endif // DEBUG_LSTM
#endif

    // TODO: currently it is not supporte to halve odd number of feature map size with SIMD
    // either fill it up with 0 or set the weight to 0
    // If this needs to be supported, round down the iterations and add the very last contribution separately.
    int inFeaturesSizeP2    = (inFeaturesSize)/2;
    int inFeaturesSizeP2_p1 = (inFeaturesSize)/2 + W_OFFSET/2;


// ===============================================================
//  _  __    _                                                               _   
// / | \ \  | | ___   ___  _ __     _____   _____ _ __    ___     ___  _   _| |_ 
// | |  | | | |/ _ \ / _ \| '_ \   / _ \ \ / / _ \ '__|  / __|   / _ \| | | | __|
// | |  | | | | (_) | (_) | |_) | | (_) \ V /  __/ |    | (__   | (_) | |_| | |_ 
// |_|  | | |_|\___/ \___/| .__/   \___/ \_/ \___|_|     \___|__ \___/ \__,_|\__|
//     /_/                |_|                                |__|              
//----------------------------------------------------------------
// 1a) Tile Output Channels
//----------------------------------------------------------------
#ifdef FMOUTTILING

// #ifdef DEBUG_LSTM
//     printf("FMOUTTILING \n");
// #endif // DEBUG_LSTM

    const int outFeaturesPerTile = Min(outFeaturesSize, OUTPUTBUFFER); // Find maximum possible feature tile size
    int outFeatureTiles = (int)(outFeaturesSize-1)/outFeaturesPerTile+1; // output channels per tile (round it up)

    // printf("outFeaturesSize=%i, outFeaturesPerTile=%i, outFeatureTiles=%i", outFeaturesSize, outFeaturesPerTile, outFeatureTiles);
    for (int o_tile=0; o_tile<outFeatureTiles; o_tile++) {

//----------------------------------------------------------------
// 1b) Do not tile Output Channels
//----------------------------------------------------------------
#else // not FMOUTTILING

// #ifdef DEBUG_LSTM
//     printf("no FMOUTTILING \n");
// #endif // DEBUG_LSTM

  const int outFeaturesPerTile = 1;  
  const int o_rel = 0;
  int o_tile =0;

#ifdef MULTICORE
    for (int o=start; o<stop; o++) 
  {
    o_tile = o;
#else
    for (int o=0; o< outFeaturesSize; o++) 
    {
        o_tile = o;
#endif

#endif // FMOUTTILING


// ===============================================================
//  ____   __    _       _ _     _                       
// |___ \  \ \  (_)_ __ (_) |_  | |_ ___ _ __ ___  _ __  
//   __) |  | | | | '_ \| | __| | __/ _ \ '_ ` _ \| '_ \ 
//  / __/   | | | | | | | | |_  | ||  __/ | | | | | |_) |
// |_____|  | | |_|_| |_|_|\__|  \__\___|_| |_| |_| .__/ 
//         /_/                                    |_|   
//----------------------------------------------------------------
// 2c) Fixed-Pt, not ASIP
//----------------------------------------------------------------
#ifdef FixedPt

// #ifdef DEBUG_LSTM
//     printf("FixedPt \n");
// #endif // DEBUG_LSTM

// manual loop unfolding to full utilize the registers
# ifdef MANUALLOOPUNFOLDING

// #ifdef DEBUG_LSTM
//     printf("MANUALLOOPUNFOLDING \n");
// #endif // DEBUG_LSTM

      register int32_t ra, rb, rc, rd, re, rf, rg, rh, ri, rj, rk, rl, rm, rn, ro;

      #if OUTPUTBUFFER>0
          ra = (int32_t)bias[o_tile*outFeaturesPerTile+0]<<(q_fraqP1);
      #endif   
      #if OUTPUTBUFFER>1
          rb = (int32_t)bias[o_tile*outFeaturesPerTile+1]<<(q_fraqP1);
      #endif
      #if OUTPUTBUFFER>2
          rc = (int32_t)bias[o_tile*outFeaturesPerTile+2]<<(q_fraqP1);
      #endif
      #if OUTPUTBUFFER>3
          rd = (int32_t)bias[o_tile*outFeaturesPerTile+3]<<(q_fraqP1);
      #endif
      #if OUTPUTBUFFER>4
          re = (int32_t)bias[o_tile*outFeaturesPerTile+4]<<(q_fraqP1);
      #endif
      #if OUTPUTBUFFER>5
          rf = (int32_t)bias[o_tile*outFeaturesPerTile+5]<<(q_fraqP1);
      #endif
      #if OUTPUTBUFFER>6
          rg = (int32_t)bias[o_tile*outFeaturesPerTile+6]<<(q_fraqP1);
      #endif
      #if OUTPUTBUFFER>7
          rh = (int32_t)bias[o_tile*outFeaturesPerTile+7]<<(q_fraqP1);
      #endif
      #if OUTPUTBUFFER>8
          ri = (int32_t)bias[o_tile*outFeaturesPerTile+8]<<(q_fraqP1);
      #endif
      #if OUTPUTBUFFER>9
          rj = (int32_t)bias[o_tile*outFeaturesPerTile+9]<<(q_fraqP1);
      #endif
      #if OUTPUTBUFFER>10
          rk = (int32_t)bias[o_tile*outFeaturesPerTile+10]<<(q_fraqP1);
      #endif
      #if OUTPUTBUFFER>11
          rl = (int32_t)bias[o_tile*outFeaturesPerTile+11]<<(q_fraqP1);
      #endif
      #if OUTPUTBUFFER>12
          rm = (int32_t)bias[o_tile*outFeaturesPerTile+12]<<(q_fraqP1);
      #endif
      #if OUTPUTBUFFER>13
          rn = (int32_t)bias[o_tile*outFeaturesPerTile+13]<<(q_fraqP1);
      #endif
      #if OUTPUTBUFFER>14
          ro = (int32_t)bias[o_tile*outFeaturesPerTile+14]<<(q_fraqP1);
      #endif

# else // NO MANUALLOOPUNFOLDING

// #ifdef DEBUG_LSTM
//     printf("no MANUALLOOPUNFOLDING \n");
// #endif // DEBUG_LSTM

      int32_t  temp[OUTPUTBUFFER];

      for(int o_rel =0; o_rel<outFeaturesPerTile; o_rel++) {
          temp[o_rel] = (int32_t)bias[o_tile*outFeaturesPerTile+o_rel]<<(q_fraqP1);
      }

# endif // MANUALLOOPUNFOLDING


//----------------------------------------------------------------
// 2d) Floating point
//----------------------------------------------------------------     
#else // not FixedPt

// #ifdef DEBUG_LSTM
//     printf("no FixedPt \n");
// #endif // DEBUG_LSTM

        data_t temp[OUTPUTBUFFER];
        temp = bias[o];

#endif // not FixedPt


// ===============================================================   
//  _____  __    _                                                    _       
// |___ /  \ \  | | ___   ___  _ __     _____   _____ _ __    ___    (_)_ __  
//   |_ \   | | | |/ _ \ / _ \| '_ \   / _ \ \ / / _ \ '__|  / __|   | | '_ \ 
//  ___) |  | | | | (_) | (_) | |_) | | (_) \ V /  __/ |    | (__    | | | | |
// |____/   | | |_|\___/ \___/| .__/   \___/ \_/ \___|_|     \___|___|_|_| |_|
//         /_/                |_|                               |_____|       
//----------------------------------------------------------------
// 3a) Iterate over N/2 pair (SIMD) of input channels where N=inFeaturesSize
//----------------------------------------------------------------
#ifdef SIMD

// #ifdef DEBUG_LSTM
//     printf("SIMD \n");
// #endif // DEBUG_LSTM

        for(int i=0; i<inFeaturesSizeP2; i++) {
            v2s inF_temp = ((v2s*)inFeatures)[i];

//----------------------------------------------------------------
// 3b) Iterate over (inFeaturesSize) inputChannels
//----------------------------------------------------------------
#else // no SIMD

// #ifdef DEBUG_LSTM
//     printf("no SIMD \n");
// #endif // DEBUG_LSTM

        for(int i=0; i<inFeaturesSize; i++) {
#endif


// ===============================================================         
//  _  _    __    _                                                               _     _   _ _      
// | || |   \ \  | | ___   ___  _ __     _____   _____ _ __    ___     ___  _   _| |_  | |_(_) | ___ 
// | || |_   | | | |/ _ \ / _ \| '_ \   / _ \ \ / / _ \ '__|  / __|   / _ \| | | | __| | __| | |/ _ \
// |__   _|  | | | | (_) | (_) | |_) | | (_) \ V /  __/ |    | (__   | (_) | |_| | |_  | |_| | |  __/
//    |_|    | | |_|\___/ \___/| .__/   \___/ \_/ \___|_|     \___|___\___/ \__,_|\__|  \__|_|_|\___|
//          /_/                |_|                               |_____|                             
//----------------------------------------------------------------
// 4a) Loop over tile for PULP
//----------------------------------------------------------------
#if defined FMOUTTILING && !defined MANUALLOOPUNFOLDING

            // NO MANUAL LOOP UNROLL
            for (int o_rel=0; o_rel < outFeaturesPerTile; o_rel++) { // chess_loop_count(4) chess_flatten_loop //chess_loop_range(1,4)
            // NO MANUAL LOOP UNROLL

#endif // defined FMOUTTILING && !defined MANUALLOOPUNFOLDING


// =============================================================== 
//  ____   __    ___                         __  __    _    ____ 
// | ___|  \ \  |_ _|_ __  _ __   ___ _ __  |  \/  |  / \  / ___|
// |___ \   | |  | || '_ \| '_ \ / _ \ '__| | |\/| | / _ \| |    
//  ___) |  | |  | || | | | | | |  __/ |    | |  | |/ ___ \ |___ 
// |____/   | | |___|_| |_|_| |_|\___|_|    |_|  |_/_/   \_\____|
//         /_/                                                   
//----------------------------------------------------------------
// 5b) SIMD on PULP with intriniscs
//----------------------------------------------------------------
#ifdef FixedPt
#ifdef SIMD

#   if !defined MANUALLOOPUNFOLDING

            temp[o_rel] = __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+o_rel)) + i], temp[o_rel]);

#   else   

        #if OUTPUTBUFFER>0
            ra = __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+0)) + i], ra);
        #endif
        #if OUTPUTBUFFER>1
            rb = __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+1)) + i], rb);
        #endif
        #if OUTPUTBUFFER>2
            rc = __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+2)) + i], rc);
        #endif
        #if OUTPUTBUFFER>3
            rd = __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+3)) + i], rd);
        #endif
        #if OUTPUTBUFFER>4
            re = __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+4)) + i], re);
        #endif
        #if OUTPUTBUFFER>5
            rf = __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+5)) + i], rf);
        #endif
        #if OUTPUTBUFFER>6
            rg = __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+6)) + i], rg);
        #endif
        #if OUTPUTBUFFER>7
            rh = __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+7)) + i], rh);
        #endif
        #if OUTPUTBUFFER>8
            ri = __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+8)) + i], ri);
        #endif
        #if OUTPUTBUFFER>9
            rj = __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+9)) + i], rj);
        #endif
        #if OUTPUTBUFFER>10
            rk = __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+10)) + i], rk);
        #endif
        #if OUTPUTBUFFER>11
            rl = __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+11)) + i], rl);
        #endif
        #if OUTPUTBUFFER>12
            rm = __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+12)) + i], rm);
        #endif
        #if OUTPUTBUFFER>13
            rn = __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+13)) + i], rn);
        #endif
        #if OUTPUTBUFFER>14
            ro = __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2_p1*(o_tile*outFeaturesPerTile+14)) + i], ro);
        #endif

    #endif // !defined MANUALLOOPUNFOLDING

//----------------------------------------------------------------
// 5d) Fixed-Point, not SIMD
//----------------------------------------------------------------
#else // no SIMD
    temp[o_rel] += (int32_t)((inFeatures[i]*weight[inFeaturesSize*(o_tile*outFeaturesPerTile+o_rel) + i]));

    #if defined MANUALLOOPUNFOLDING
        printf("WARN: for simplicity currently no MANUAL loop unfolding for non-SIMD fixed-point");
    #endif
#endif // SIMD

//----------------------------------------------------------------
// 5e) Floating Point
//----------------------------------------------------------------
#else // no FixedPt

    temp[o_rel] += inFeatures[i]*weight[inFeaturesSize*(o_tile*outFeaturesPerTile+o_rel) + i];
    
    #if defined MANUALLOOPUNFOLDING
        printf("WARN: for simplicity currently no MANUAL loop unfolding for float");
    #endif
#endif // FixedPt

    }

#if defined FMOUTTILING && !defined MANUALLOOPUNFOLDING
    }
#endif // defined FMOUTTILING && !defined MANUALLOOPUNFOLDING

// NO MANUAL LOOP UNFOLD
#if !defined MANUALLOOPUNFOLDING || !defined FixedPt

    for(int o_rel =0;o_rel<outFeaturesPerTile;o_rel++) {
        #ifdef FixedPt
            outFeatures[(o_tile*outFeaturesPerTile+o_rel)] = temp[o_rel]>>(q_fraqP1);
        #else // no FixedPt
            outFeatures[(o_tile*outFeaturesPerTile+o_rel)] = temp[o_rel];
        #endif // FixedPt
    }

#else // no (!defined MANUALLOOPUNFOLDING || !defined FixedPt)

    #if OUTPUTBUFFER>0
        outFeatures[(o_tile*outFeaturesPerTile+0)] = ra>>(q_fraqP1);
    #endif
    #if OUTPUTBUFFER>1
        outFeatures[(o_tile*outFeaturesPerTile+1)] = rb>>(q_fraqP1);
    #endif
    #if OUTPUTBUFFER>2
        outFeatures[(o_tile*outFeaturesPerTile+2)] = rc>>(q_fraqP1);
    #endif
    #if OUTPUTBUFFER>3
        outFeatures[(o_tile*outFeaturesPerTile+3)] = rd>>(q_fraqP1);
    #endif
    #if OUTPUTBUFFER>4
        outFeatures[(o_tile*outFeaturesPerTile+4)] = re>>(q_fraqP1);
    #endif
    #if OUTPUTBUFFER>5
        outFeatures[(o_tile*outFeaturesPerTile+5)] = rf>>(q_fraqP1);
    #endif
    #if OUTPUTBUFFER>6
        outFeatures[(o_tile*outFeaturesPerTile+6)] = rg>>(q_fraqP1);
    #endif
    #if OUTPUTBUFFER>7
        outFeatures[(o_tile*outFeaturesPerTile+7)] = rh>>(q_fraqP1);
    #endif
    #if OUTPUTBUFFER>8
        outFeatures[(o_tile*outFeaturesPerTile+8)] = ri>>(q_fraqP1);
    #endif
    #if OUTPUTBUFFER>9
        outFeatures[(o_tile*outFeaturesPerTile+9)] = rj>>(q_fraqP1);
    #endif
    #if OUTPUTBUFFER>10
        outFeatures[(o_tile*outFeaturesPerTile+10)] = rk>>(q_fraqP1);
    #endif
    #if OUTPUTBUFFER>11
        outFeatures[(o_tile*outFeaturesPerTile+11)] = rl>>(q_fraqP1);
    #endif
    #if OUTPUTBUFFER>12
        outFeatures[(o_tile*outFeaturesPerTile+12)] = rm>>(q_fraqP1);
    #endif
    #if OUTPUTBUFFER>13
        outFeatures[(o_tile*outFeaturesPerTile+13)] = rn>>(q_fraqP1);
    #endif
    #if OUTPUTBUFFER>14
        outFeatures[(o_tile*outFeaturesPerTile+14)] = ro>>(q_fraqP1);
    #endif

#endif // !defined MANUALLOOPUNFOLDING || !defined FixedPt

    }

    PROFILING_LINEAR_END
}


#endif









/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
/////   ____                 ____     _ _                           /////////////////////////
/////  / ___|___  _ ____   _|___ \ __| | |    __ _ _   _  ___ _ __  /////////////////////////
///// | |   / _ \| '_ \ \ / / __) / _` | |   / _` | | | |/ _ \ '__| /////////////////////////
///// | |__| (_) | | | \ V / / __/ (_| | |__| (_| | |_| |  __/ |    /////////////////////////
/////  \____\___/|_| |_|\_/ |_____\__,_|_____\__,_|\__, |\___|_|    /////////////////////////
/////                                              |___/            /////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
//         _____  _______ ______  _______ _______ _______      _    _        _____ _  _  _ //
// |      |     | |_____| |     \ |  |  | |_____| |             \  /  |        |   |  |  | //
// |_____ |_____| |     | |_____/ |  |  | |     | |_____         \/   |_____ __|__ |__|__| //
//                                                                                         //
/////////////////////////////////////////////////////////////////////////////////////////////                                                                                      
#if defined(VLIWEXT) // RISCY implementation with the lw-sdopt-VLIW
/** @brief Calculates a 2D Convolution Layer PULP+VLIW+(SIMD)
 *  input channels need to be multiple of 4 or 2 (with/without FMINTILING)
 *  Supporte configurations:
 *  > VLIWEXT 
 *  > SIMD only
 *  > FMIN and FMOUTILING
 *  > MANUALLOOPUNFOLDING true
 *
 *  @param _layer Layer Properties
 *  @param h_im Image Height
 *  @param w_im Image Width
 *  @param inFeatures Input Feature Map
 *  @param outFeatures Output Feature Map
 */
int NOINLINE Conv2dLayer (
  struct layer * _layer,
  int h_im,
  int w_im,
  data_t * __restrict__ inFeatures,
  data_t * __restrict__ outFeatures) {

   #if OUTPUTBUFFER > 8
 int tileOptions[] = {OUTPUTBUFFER,8,4,2,1};
   #elif OUTPUTBUFFER > 4
 int tileOptions[] = {OUTPUTBUFFER,4,2,1};
   #elif OUTPUTBUFFER > 2
 int tileOptions[] = {OUTPUTBUFFER,2,1};
   #elif OUTPUTBUFFER > 1
 int tileOptions[] = {2,1};
   #else
 int tileOptions[] = {1};
   #endif
 int outFeaturesPerTile = 1;
 int h_im_out = h_im;
 int w_im_out = w_im;
 int h_ker_half = (int)(_layer->attributes[LAY_CONV_KER]/2);
   int w_ker_half = h_ker_half; // TODO: symmetric kernel only
   data_t * bias_ptr   = _layer->parameters[CONV_BIAS];
   v2s* param_simd = (v2s*) _layer->parameters[CONV_WGHT];
   data_t  * outFeatures_ptr = outFeatures;

   unsigned int output_channel_offset = _layer->attributes[LAY_CONV_KER]*_layer->attributes[LAY_CONV_KER]*_layer->attributes[LAY_CONV_IN]/2;
   unsigned int kernel_H_offset = _layer->attributes[LAY_CONV_KER]*_layer->attributes[LAY_CONV_IN]/2;
   unsigned int kernel_W_offset = _layer->attributes[LAY_CONV_IN]/2;

   int c_in_max;
   c_in_max = _layer->attributes[LAY_CONV_IN];
   #ifdef FMINTILING
   c_in_max = _layer->attributes[LAY_CONV_IN]/4;
   #else
   c_in_max = _layer->attributes[LAY_CONV_IN]/2;
   #endif
   register int x0 asm("x0");
   register int32_t temp, temp1, temp2, temp3, temp4, temp5, temp6, temp7;
   register uint32_t  addr0, addr1, addr2, addr3, addr4, addr5, addr6, addr7;
   // const int outFeaturesPerTile = 2;
   unsigned int param_kh_base = 0;
   int outFeatureTiles;
   int outFeaturesSize_remain = _layer->attributes[LAY_CONV_OUT];

  // Tile with largest tileOption
   for(unsigned int i=0; i<sizeof(tileOptions)/sizeof(int); i++) {
    outFeaturesPerTile = tileOptions[i];
    outFeatureTiles = outFeaturesSize_remain/outFeaturesPerTile;
    
    if(outFeatureTiles == 0) continue;
    switch(outFeaturesPerTile) {
     #if OUTPUTBUFFER > 2
     case OUTPUTBUFFER:
     for(int c_out = 0; c_out < outFeatureTiles; c_out++) {
       for(int h_out =0; h_out < h_im_out; h_out++) 
       {
         for(int w_out=0;w_out<w_im_out; w_out++)
         {

           int kh_slide_start = Max(-h_out, -h_ker_half);                  // Handle borders
           int kh_slide_stop = Min(h_im_out-1-h_out,h_ker_half);           // Handle borders

            #if OUTPUTBUFFER > 2
           temp = bias_ptr[outFeaturesPerTile*c_out] << q_fraqP1;
            #endif
            #if OUTPUTBUFFER > 3
           temp1 = bias_ptr[outFeaturesPerTile*c_out+1] << q_fraqP1;
            #endif
            #if OUTPUTBUFFER > 4
           temp2 = bias_ptr[outFeaturesPerTile*c_out+2] << q_fraqP1;
            #endif
            #if OUTPUTBUFFER > 5
           temp3 = bias_ptr[outFeaturesPerTile*c_out+3] << q_fraqP1;
            #endif
            #if OUTPUTBUFFER > 6
           temp4 = bias_ptr[outFeaturesPerTile*c_out+4] << q_fraqP1;
            #endif
            #if OUTPUTBUFFER > 7
           temp5 = bias_ptr[outFeaturesPerTile*c_out+5] << q_fraqP1;
            #endif
           temp6 = bias_ptr[outFeaturesPerTile*c_out+OUTPUTBUFFER-2]<<(q_fraqP1);
           temp7 = bias_ptr[outFeaturesPerTile*c_out+OUTPUTBUFFER-1]<<(q_fraqP1);

           unsigned int param_kw_base = param_kh_base \
           + (kh_slide_start+h_ker_half) * kernel_H_offset;
           unsigned int feat_kw_base = (h_out+kh_slide_start)*w_im_out*_layer->attributes[LAY_CONV_IN]/2;
           for(int kh=kh_slide_start; kh <= kh_slide_stop;kh++)
           {
              int kw_slide_start = Max(-w_out, -w_ker_half);           // Handle borders
              int kw_slide_stop = Min(w_im_out-1-w_out,w_ker_half);           // Handle borders
              
              int param_id_base = param_kw_base \
                                            +(kw_slide_start+w_ker_half)*kernel_W_offset; // filter tap
                                            int param_id_base1 = param_id_base+output_channel_offset;
                                            int feat_id_base  = feat_kw_base \
                                            +(w_out+kw_slide_start)* _layer->attributes[LAY_CONV_IN]/2;
                                            for(int kw=kw_slide_start; kw <= kw_slide_stop;kw++)
                                            {
                                             addr0  = (uint32_t) &((v2s*)param_simd)[param_id_base];
                                             addr1  = (uint32_t) &((v2s*)param_simd)[param_id_base+1*output_channel_offset];
                                             addr2  = (uint32_t) &((v2s*)param_simd)[param_id_base+2*output_channel_offset];
                                             addr3  = (uint32_t) &((v2s*)param_simd)[param_id_base+3*output_channel_offset];
                                             addr4  = (uint32_t) &((v2s*)param_simd)[param_id_base+4*output_channel_offset];
                                             addr5  = (uint32_t) &((v2s*)param_simd)[param_id_base+5*output_channel_offset];
                                             addr6  = (uint32_t) &((v2s*)param_simd)[param_id_base+output_channel_offset*(OUTPUTBUFFER-2)];
                                             addr7  = (uint32_t) &((v2s*)param_simd)[param_id_base+output_channel_offset*(OUTPUTBUFFER-1)];

                 asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload first weight
                 asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload first weight
                 v2s* in_addr = &((v2s*)inFeatures)[feat_id_base];
                 for(int i=0; i <  c_in_max;i++) // i=c_in
                 {
                  // loop unfolding
                  // temp  = __SUMDOTP2(((v2s*)inFeatures)[feat_id_base + i], \
                  //  param_simd[param_id_base + i], temp);
                  // temp1 = __SUMDOTP2(((v2s*)inFeatures)[feat_id_base + i], \
                  //  param_simd[param_id_base1 + i], temp1);
                   // v2s inF_temp = ((v2s*)inFeatures)[feat_id_base + i];
                   // temp  = __SUMDOTP2(inF_temp, \
                   // param_simd[param_id_base + i], temp);
                   // asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp),  "+r" (addr0) : "r" (inF_temp) ); 
                   // temp1 = __SUMDOTP2(inF_temp, \
                   // param_simd[param_id_base1 + i], temp1);
                   // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr1) : "r" (inF_temp) ); 
                   // v2s inF_temp = ((v2s*)inFeatures)[feat_id_base + i];
                 v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
                 v2s inF_temp2;// = ((v2s*)inFeatures)[2*i+1];


                 // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
                 asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
                 #ifdef FMINTILING
                    asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
                 #endif
                 #if OUTPUTBUFFER > 2
                    asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp),  "+r" (addr2) : "r" (inF_temp) );
                 #endif
                 #if OUTPUTBUFFER > 3
                    asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp) );
                 #endif
                 #if OUTPUTBUFFER > 4
                    asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr4) : "r" (inF_temp) );
                 #endif
                 #if OUTPUTBUFFER > 5
                    asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr5) : "r" (inF_temp) );
                 #endif
                 #if OUTPUTBUFFER > 6
                    asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp4),  "+r" (addr6) : "r" (inF_temp) );
                 #endif
                 #if OUTPUTBUFFER > 7
                    asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp5),  "+r" (addr7) : "r" (inF_temp) ); 
                 #endif

                    asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr0) : "r" (inF_temp) );
                    asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr1) : "r" (inF_temp) );
// do it twice for FMINTILING
#ifdef FMINTILING
                 #if OUTPUTBUFFER > 2
                    asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp),  "+r" (addr2) : "r" (inF_temp2) );
                 #endif
                 #if OUTPUTBUFFER > 3
                    asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp2) );
                 #endif
                 #if OUTPUTBUFFER > 4
                    asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr4) : "r" (inF_temp2) );
                 #endif
                 #if OUTPUTBUFFER > 5
                    asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr5) : "r" (inF_temp2) );
                 #endif
                 #if OUTPUTBUFFER > 6
                    asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp4),  "+r" (addr6) : "r" (inF_temp2) );
                 #endif
                 #if OUTPUTBUFFER > 7
                    asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp5),  "+r" (addr7) : "r" (inF_temp2) ); 
                 #endif

                    asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr0) : "r" (inF_temp2) );
                    asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr1) : "r" (inF_temp2) );


#endif
                  }
              param_id_base  += kernel_W_offset; // filter tap
              param_id_base1 += kernel_W_offset; // filter tap
              feat_id_base   +=  _layer->attributes[LAY_CONV_IN]/2;
            }

                  #if OUTPUTBUFFER > 2
            outFeatures_ptr[(outFeaturesPerTile*c_out+0)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp) >> q_fraqP1);
                  #endif
                  #if OUTPUTBUFFER > 3
            outFeatures_ptr[(outFeaturesPerTile*c_out+1)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp1) >> q_fraqP1);
                  #endif
                  #if OUTPUTBUFFER > 4
            outFeatures_ptr[(outFeaturesPerTile*c_out+2)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp2) >> q_fraqP1);
                  #endif
                  #if OUTPUTBUFFER > 5
            outFeatures_ptr[(outFeaturesPerTile*c_out+3)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp3) >> q_fraqP1);
                  #endif
                  #if OUTPUTBUFFER > 6
            outFeatures_ptr[(outFeaturesPerTile*c_out+4)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp4) >> q_fraqP1);
                  #endif
                  #if OUTPUTBUFFER > 7
            outFeatures_ptr[(outFeaturesPerTile*c_out+5)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp5) >> q_fraqP1);
                  #endif

            outFeatures_ptr[(outFeaturesPerTile*c_out+OUTPUTBUFFER-2)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp6) >> q_fraqP1);
            outFeatures_ptr[(outFeaturesPerTile*c_out+OUTPUTBUFFER-1)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp7) >> q_fraqP1);
            param_kw_base += kernel_H_offset;
            feat_kw_base  += w_im_out*_layer->attributes[LAY_CONV_IN]/2;
          }
        }
      }
      param_kh_base += outFeaturesPerTile * output_channel_offset;
    }  
    break;
     #endif
     #if OUTPUTBUFFER > 4
    case 4:
    for(int c_out = 0; c_out < outFeatureTiles; c_out++) {
     for(int h_out =0; h_out < h_im_out; h_out++) 
     {
       for(int w_out=0;w_out<w_im_out; w_out++)
       {

           int kh_slide_start = Max(-h_out, -h_ker_half);                  // Handle borders
           int kh_slide_stop = Min(h_im_out-1-h_out,h_ker_half);           // Handle borders
           temp = bias_ptr[outFeaturesPerTile*c_out] << q_fraqP1;
           temp1 =  bias_ptr[(outFeaturesPerTile*c_out+1)] << q_fraqP1;
           temp = bias_ptr[outFeaturesPerTile*c_out] << q_fraqP1;
           temp1 = bias_ptr[outFeaturesPerTile*c_out+1] << q_fraqP1;
           temp6 = bias_ptr[outFeaturesPerTile*c_out+2]<<(q_fraqP1);
           temp7 = bias_ptr[outFeaturesPerTile*c_out+3]<<(q_fraqP1);

           unsigned int param_kw_base = param_kh_base \
           + (kh_slide_start+h_ker_half) * kernel_H_offset;
           unsigned int feat_kw_base = (h_out+kh_slide_start)*w_im_out*_layer->attributes[LAY_CONV_IN]/2;
           for(int kh=kh_slide_start; kh <= kh_slide_stop;kh++)
           {
              int kw_slide_start = Max(-w_out, -w_ker_half);           // Handle borders
              int kw_slide_stop = Min(w_im_out-1-w_out,w_ker_half);           // Handle borders
              
              int param_id_base = param_kw_base \
                                            +(kw_slide_start+w_ker_half)*kernel_W_offset; // filter tap
                                            int param_id_base1 = param_id_base+output_channel_offset;
                                            int feat_id_base  = feat_kw_base \
                                            +(w_out+kw_slide_start)* _layer->attributes[LAY_CONV_IN]/2;
                                            for(int kw=kw_slide_start; kw <= kw_slide_stop;kw++)
                                            {
                                             addr0  = (uint32_t) &((v2s*)param_simd)[param_id_base];
                                             addr1  = (uint32_t) &((v2s*)param_simd)[param_id_base+1*output_channel_offset];
                                             addr6  = (uint32_t) &((v2s*)param_simd)[param_id_base+output_channel_offset*2];
                                             addr7  = (uint32_t) &((v2s*)param_simd)[param_id_base+output_channel_offset*3];

                 asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload first weight
                 asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload first weight
                 v2s* in_addr = &((v2s*)inFeatures)[feat_id_base];
                 v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
                 v2s inF_temp2;// = ((v2s*)inFeatures)[2*i+1];


                 for(int i=0; i <  c_in_max;i++) // i=c_in
                 {
                 // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
                 asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
                 #ifdef FMINTILING
                    asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
                 #endif
                  // loop unfolding
                  // temp  = __SUMDOTP2(((v2s*)inFeatures)[feat_id_base + i], \
                  //  param_simd[param_id_base + i], temp);
                  // temp1 = __SUMDOTP2(((v2s*)inFeatures)[feat_id_base + i], \
                  //  param_simd[param_id_base1 + i], temp1);
                   // v2s inF_temp = ((v2s*)inFeatures)[feat_id_base + i];
                   // temp  = __SUMDOTP2(inF_temp, \
                   // param_simd[param_id_base + i], temp);
                   // asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp),  "+r" (addr0) : "r" (inF_temp) ); 
                   // temp1 = __SUMDOTP2(inF_temp, \
                   // param_simd[param_id_base1 + i], temp1);
                   // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr1) : "r" (inF_temp) ); 
                   // v2s inF_temp = ((v2s*)inFeatures)[feat_id_base + i];

                    asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp),   "+r" (addr2) : "r" (inF_temp) );
                    asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp) );
                    asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr0) : "r" (inF_temp) );
                    asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr1) : "r" (inF_temp) );
                    #ifdef FMINTILING
                    asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp),   "+r" (addr2) : "r" (inF_temp2) );
                    asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp2) );
                    asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr0) : "r" (inF_temp2) );
                    asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr1) : "r" (inF_temp2) );
                    #endif

                  }
              param_id_base  += kernel_W_offset; // filter tap
              param_id_base1 += kernel_W_offset; // filter tap
              feat_id_base   +=  _layer->attributes[LAY_CONV_IN]/2;
            }

            outFeatures_ptr[(outFeaturesPerTile*c_out+0)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp) >> q_fraqP1);
            outFeatures_ptr[(outFeaturesPerTile*c_out+1)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp1) >> q_fraqP1);
            outFeatures_ptr[(outFeaturesPerTile*c_out+2)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp6) >> q_fraqP1);
            outFeatures_ptr[(outFeaturesPerTile*c_out+3)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp7) >> q_fraqP1);

            param_kw_base += kernel_H_offset;
            feat_kw_base  += w_im_out*_layer->attributes[LAY_CONV_IN]/2;
          }
        }
      }
      param_kh_base += outFeaturesPerTile * output_channel_offset;
    }  
    break;
#endif
    case 2:
    for(int c_out = 0; c_out < outFeatureTiles; c_out++) {
     for(int h_out =0; h_out < h_im_out; h_out++) 
     {
       for(int w_out=0;w_out<w_im_out; w_out++)
       {

           int kh_slide_start = Max(-h_out, -h_ker_half);                  // Handle borders
           int kh_slide_stop = Min(h_im_out-1-h_out,h_ker_half);           // Handle borders
           
           temp6 = bias_ptr[outFeaturesPerTile*c_out+0]<<(q_fraqP1);
           temp7 = bias_ptr[outFeaturesPerTile*c_out+1]<<(q_fraqP1);

           unsigned int param_kw_base = param_kh_base \
           + (kh_slide_start+h_ker_half) * kernel_H_offset;
           unsigned int feat_kw_base = (h_out+kh_slide_start)*w_im_out*_layer->attributes[LAY_CONV_IN]/2;
           for(int kh=kh_slide_start; kh <= kh_slide_stop;kh++)
           {
              int kw_slide_start = Max(-w_out, -w_ker_half);           // Handle borders
              int kw_slide_stop = Min(w_im_out-1-w_out,w_ker_half);           // Handle borders
              
              int param_id_base = param_kw_base \
                                            +(kw_slide_start+w_ker_half)*kernel_W_offset; // filter tap
                                            int param_id_base1 = param_id_base+output_channel_offset;
                                            int feat_id_base  = feat_kw_base \
                                            +(w_out+kw_slide_start)* _layer->attributes[LAY_CONV_IN]/2;
                                            for(int kw=kw_slide_start; kw <= kw_slide_stop;kw++)
                                            {

                                             addr6  = (uint32_t) &((v2s*)param_simd)[param_id_base+output_channel_offset*0];
                                             addr7  = (uint32_t) &((v2s*)param_simd)[param_id_base+output_channel_offset*1];

                 asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr6) : "r" (x0) ); // preload first weight
                 asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr7) : "r" (x0) ); // preload first weight

                 v2s* in_addr = &((v2s*)inFeatures)[feat_id_base];
                 v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
                 v2s inF_temp2;// = ((v2s*)inFeatures)[2*i+1];


                 for(int i=0; i <  c_in_max;i++) // i=c_in
                 {
                 // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
                 asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
                 #ifdef FMINTILING
                    asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
                 #endif
                  // loop unfolding
                  // temp  = __SUMDOTP2(((v2s*)inFeatures)[feat_id_base + i], \
                  //  param_simd[param_id_base + i], temp);
                  // temp1 = __SUMDOTP2(((v2s*)inFeatures)[feat_id_base + i], \
                  //  param_simd[param_id_base1 + i], temp1);
                   // v2s inF_temp = ((v2s*)inFeatures)[feat_id_base + i];
                   // temp  = __SUMDOTP2(inF_temp, \
                   // param_simd[param_id_base + i], temp);
                   // asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp),  "+r" (addr0) : "r" (inF_temp) ); 
                   // temp1 = __SUMDOTP2(inF_temp, \
                   // param_simd[param_id_base1 + i], temp1);
                   // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr1) : "r" (inF_temp) ); 
                   // v2s inF_temp = ((v2s*)inFeatures)[feat_id_base + i];


                    asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr6) : "r" (inF_temp) );
                    asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr7) : "r" (inF_temp) );
                    #ifdef FMINTILING
                    asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr6) : "r" (inF_temp2) );
                    asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr7) : "r" (inF_temp2) );
                    #endif

                  }
              param_id_base  += kernel_W_offset; // filter tap
              param_id_base1 += kernel_W_offset; // filter tap
              feat_id_base   +=  _layer->attributes[LAY_CONV_IN]/2;
            }

            outFeatures_ptr[(outFeaturesPerTile*c_out+0)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp6) >> q_fraqP1);
            outFeatures_ptr[(outFeaturesPerTile*c_out+1)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp7) >> q_fraqP1);
            param_kw_base += kernel_H_offset;
            feat_kw_base  += w_im_out*_layer->attributes[LAY_CONV_IN]/2;
          }
        }
      }
      param_kh_base += outFeaturesPerTile * output_channel_offset;
    }  
    break;
    case 1:
    for(int c_out = 0; c_out < outFeatureTiles; c_out++) {
     for(int h_out =0; h_out < h_im_out; h_out++) 
     {
       for(int w_out=0;w_out<w_im_out; w_out++)
       {

           int kh_slide_start = Max(-h_out, -h_ker_half);                  // Handle borders
           int kh_slide_stop = Min(h_im_out-1-h_out,h_ker_half);           // Handle borders
           
           temp6 = bias_ptr[outFeaturesPerTile*c_out+0]<<(q_fraqP1);
          // temp7 = bias_ptr[outFeaturesPerTile*c_out+1]<<(q_fraqP1);

           unsigned int param_kw_base = param_kh_base \
           + (kh_slide_start+h_ker_half) * kernel_H_offset;
           unsigned int feat_kw_base = (h_out+kh_slide_start)*w_im_out*_layer->attributes[LAY_CONV_IN]/2;
           for(int kh=kh_slide_start; kh <= kh_slide_stop;kh++)
           {
              int kw_slide_start = Max(-w_out, -w_ker_half);           // Handle borders
              int kw_slide_stop = Min(w_im_out-1-w_out,w_ker_half);           // Handle borders
              
              int param_id_base = param_kw_base \
                                            +(kw_slide_start+w_ker_half)*kernel_W_offset; // filter tap
                                            int param_id_base1 = param_id_base+output_channel_offset;
                                            int feat_id_base  = feat_kw_base \
                                            +(w_out+kw_slide_start)* _layer->attributes[LAY_CONV_IN]/2;
                                            for(int kw=kw_slide_start; kw <= kw_slide_stop;kw++)
                                            {

                 // addr6  = &((v2s*)param_simd)[param_id_base+output_channel_offset*0];
                #ifdef FMINTILING
                for(int i=0; i <  2*c_in_max;i++) // i=c_in                              
                #else
                for(int i=0; i <  c_in_max;i++) // i=c_in                             
                #endif
                 
                 {
                  temp6  = __SUMDOTP2(((v2s*)inFeatures)[feat_id_base + i], \
                   param_simd[param_id_base + i], temp6);
                }
              param_id_base  += kernel_W_offset; // filter tap
              param_id_base1 += kernel_W_offset; // filter tap
              feat_id_base   +=  _layer->attributes[LAY_CONV_IN]/2;
            }

            outFeatures_ptr[(outFeaturesPerTile*c_out+0)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp6) >> q_fraqP1);
            param_kw_base += kernel_H_offset;
            feat_kw_base  += w_im_out*_layer->attributes[LAY_CONV_IN]/2;
          }
        }
      }
      param_kh_base += outFeaturesPerTile * output_channel_offset;
    }  
    break;
  }


      // move pointers for next iteration
  bias_ptr                = &bias_ptr[outFeaturesPerTile*outFeatureTiles];
      // param_simd              = &((v2s*)param_simd)[output_channel_offset*(outFeatureTiles*outFeaturesPerTile)];
  outFeatures_ptr         = &outFeatures_ptr[outFeaturesPerTile*outFeatureTiles*h_im_out*w_im_out];
  outFeaturesSize_remain -= outFeatureTiles*outFeaturesPerTile;

  if (outFeaturesSize_remain==0) break;
}
return 0;
}
#elif defined(FMOUTTILING) // RISCY implementation with the lw-sdopt-VLIW
/** @brief Calculates a 2D Convolution Layer PULP+VLIW+(SIMD)
 *  input channels need to be multiple of 4 or 2 (with/without FMINTILING)
 *  Supporte configurations:
 *  > VLIWEXT 
 *  > SIMD only
 *  > FMIN and FMOUTILING
 *  > MANUALLOOPUNFOLDING true
 *
 *  @param _layer Layer Properties
 *  @param h_im Image Height
 *  @param w_im Image Width
 *  @param inFeatures Input Feature Map
 *  @param outFeatures Output Feature Map
 */
int NOINLINE Conv2dLayer (
  struct layer * _layer,
  int h_im,
  int w_im,
  data_t * __restrict__ inFeatures,
  data_t * __restrict__ outFeatures) {

   #if OUTPUTBUFFER > 8
 int tileOptions[] = {OUTPUTBUFFER,8,4,2,1};
   #elif OUTPUTBUFFER > 4
 int tileOptions[] = {OUTPUTBUFFER,4,2,1};
   #elif OUTPUTBUFFER > 2
 int tileOptions[] = {OUTPUTBUFFER,2,1};
   #elif OUTPUTBUFFER > 1
 int tileOptions[] = {2,1};
   #else
 int tileOptions[] = {1};
   #endif
 int outFeaturesPerTile = 1;
 int h_im_out = h_im;
 int w_im_out = w_im;
 int h_ker_half = (int)(_layer->attributes[LAY_CONV_KER]/2);
   int w_ker_half = h_ker_half; // TODO: symmetric kernel only
   data_t * bias_ptr   = _layer->parameters[CONV_BIAS];
   v2s* param_simd = (v2s*) _layer->parameters[CONV_WGHT];
   data_t  * outFeatures_ptr = outFeatures;

   unsigned int output_channel_offset = _layer->attributes[LAY_CONV_KER]*_layer->attributes[LAY_CONV_KER]*_layer->attributes[LAY_CONV_IN]/2;
   unsigned int kernel_H_offset = _layer->attributes[LAY_CONV_KER]*_layer->attributes[LAY_CONV_IN]/2;
   unsigned int kernel_W_offset = _layer->attributes[LAY_CONV_IN]/2;

   int c_in_max;
   c_in_max = _layer->attributes[LAY_CONV_IN];
   // #ifdef FMINTILING
   // c_in_max = _layer->attributes[LAY_CONV_IN]/4;
   // #else
   c_in_max = _layer->attributes[LAY_CONV_IN]/2;
   // #endif
   
   register_attribute int32_t temp, temp1, temp2, temp3, temp4, temp5, temp6, temp7;
   register_attribute uint32_t  addr0, addr1, addr2, addr3, addr4, addr5, addr6, addr7;
   // const int outFeaturesPerTile = 2;
   unsigned int param_kh_base = 0;
   int outFeatureTiles;
   int outFeaturesSize_remain = _layer->attributes[LAY_CONV_OUT];

  // Tile with largest tileOption
   for(unsigned int i=0; i<sizeof(tileOptions)/sizeof(int); i++) {
    outFeaturesPerTile = tileOptions[i];
    outFeatureTiles = outFeaturesSize_remain/outFeaturesPerTile;
    
    if(outFeatureTiles == 0) continue;
    switch(outFeaturesPerTile) {
     #if OUTPUTBUFFER > 2
     case OUTPUTBUFFER:
     for(int c_out = 0; c_out < outFeatureTiles; c_out++) {
       for(int h_out =0; h_out < h_im_out; h_out++) 
       {
         for(int w_out=0;w_out<w_im_out; w_out++)
         {

           int kh_slide_start = Max(-h_out, -h_ker_half);                  // Handle borders
           int kh_slide_stop = Min(h_im_out-1-h_out,h_ker_half);           // Handle borders

            #if OUTPUTBUFFER > 2
           temp = bias_ptr[outFeaturesPerTile*c_out] << q_fraqP1;
            #endif
            #if OUTPUTBUFFER > 3
           temp1 = bias_ptr[outFeaturesPerTile*c_out+1] << q_fraqP1;
            #endif
            #if OUTPUTBUFFER > 4
           temp2 = bias_ptr[outFeaturesPerTile*c_out+2] << q_fraqP1;
            #endif
            #if OUTPUTBUFFER > 5
           temp3 = bias_ptr[outFeaturesPerTile*c_out+3] << q_fraqP1;
            #endif
            #if OUTPUTBUFFER > 6
           temp4 = bias_ptr[outFeaturesPerTile*c_out+4] << q_fraqP1;
            #endif
            #if OUTPUTBUFFER > 7
           temp5 = bias_ptr[outFeaturesPerTile*c_out+5] << q_fraqP1;
            #endif
           temp6 = bias_ptr[outFeaturesPerTile*c_out+OUTPUTBUFFER-2]<<(q_fraqP1);
           temp7 = bias_ptr[outFeaturesPerTile*c_out+OUTPUTBUFFER-1]<<(q_fraqP1);

           unsigned int param_kw_base = param_kh_base \
           + (kh_slide_start+h_ker_half) * kernel_H_offset;
           unsigned int feat_kw_base = (h_out+kh_slide_start)*w_im_out*_layer->attributes[LAY_CONV_IN]/2;
           for(int kh=kh_slide_start; kh <= kh_slide_stop;kh++)
           {
              int kw_slide_start = Max(-w_out, -w_ker_half);           // Handle borders
              int kw_slide_stop = Min(w_im_out-1-w_out,w_ker_half);           // Handle borders
              
              int param_id_base = param_kw_base \
                                            +(kw_slide_start+w_ker_half)*kernel_W_offset; // filter tap
                                            int param_id_base1 = param_id_base+output_channel_offset;
                                            int feat_id_base  = feat_kw_base \
                                            +(w_out+kw_slide_start)* _layer->attributes[LAY_CONV_IN]/2;
                                            for(int kw=kw_slide_start; kw <= kw_slide_stop;kw++)
                                            {
                                             addr0  = (uint32_t) &((v2s*)param_simd)[param_id_base];
                                             addr1  = (uint32_t) &((v2s*)param_simd)[param_id_base+1*output_channel_offset];
                                             addr2  = (uint32_t) &((v2s*)param_simd)[param_id_base+2*output_channel_offset];
                                             addr3  = (uint32_t) &((v2s*)param_simd)[param_id_base+3*output_channel_offset];
                                             addr4  = (uint32_t) &((v2s*)param_simd)[param_id_base+4*output_channel_offset];
                                             addr5  = (uint32_t) &((v2s*)param_simd)[param_id_base+5*output_channel_offset];
                                             addr6  = (uint32_t) &((v2s*)param_simd)[param_id_base+output_channel_offset*(OUTPUTBUFFER-2)];
                                             addr7  = (uint32_t) &((v2s*)param_simd)[param_id_base+output_channel_offset*(OUTPUTBUFFER-1)];

                 // asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload first weight
                 // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload first weight
                 v2s* in_addr = &((v2s*)inFeatures)[feat_id_base];
                 for(int i=0; i <  c_in_max;i++) // i=c_in
                 {
                  // loop unfolding
                  // temp  = __SUMDOTP2(((v2s*)inFeatures)[feat_id_base + i], \
                  //  param_simd[param_id_base + i], temp);
                  // temp1 = __SUMDOTP2(((v2s*)inFeatures)[feat_id_base + i], \
                  //  param_simd[param_id_base1 + i], temp1);
                   // v2s inF_temp = ((v2s*)inFeatures)[feat_id_base + i];
                   // temp  = __SUMDOTP2(inF_temp, \
                   // param_simd[param_id_base + i], temp);
                   // asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp),  "+r" (addr0) : "r" (inF_temp) ); 
                   // temp1 = __SUMDOTP2(inF_temp, \
                   // param_simd[param_id_base1 + i], temp1);
                   // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr1) : "r" (inF_temp) ); 
                   // v2s inF_temp = ((v2s*)inFeatures)[feat_id_base + i];
                 v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
                 v2s inF_temp2;// = ((v2s*)inFeatures)[2*i+1];


                 // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
                 asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
                 // #ifdef FMINTILING
                 //    asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
                 // #endif
                 #if OUTPUTBUFFER > 2
              SDOTP_GENERIC(temp, ((v2s*)addr0)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 3
              SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 4
              SDOTP_GENERIC(temp2, ((v2s*)addr2)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 5
              SDOTP_GENERIC(temp3, ((v2s*)addr3)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 6
              SDOTP_GENERIC(temp4, ((v2s*)addr4)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 7
              SDOTP_GENERIC(temp5, ((v2s*)addr5)[i], inF_temp);
               #endif
              
              SDOTP_GENERIC(temp6, ((v2s*)addr6)[i],inF_temp);
              SDOTP_GENERIC(temp7, ((v2s*)addr7)[i],inF_temp);
// do it twice for FMINTILING
  // #ifdef FMINTILING
  //               #if OUTPUTBUFFER > 2
  //             SDOTP_GENERIC(temp, ((v2s*)addr0)[2*i+1], inF_temp2);
  //              #endif
  //              #if OUTPUTBUFFER > 3
  //             SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp2);
  //              #endif
  //              #if OUTPUTBUFFER > 4
  //             SDOTP_GENERIC(temp2, ((v2s*)addr2)[i], inF_temp2);
  //              #endif
  //              #if OUTPUTBUFFER > 5
  //             SDOTP_GENERIC(temp3, ((v2s*)addr3)[i], inF_temp2);
  //              #endif
  //              #if OUTPUTBUFFER > 6
  //             SDOTP_GENERIC(temp4, ((v2s*)addr4)[i], inF_temp2);
  //              #endif
  //              #if OUTPUTBUFFER > 7
  //             SDOTP_GENERIC(temp5, ((v2s*)addr5)[i], inF_temp2);
  //              #endif
              
  //             SDOTP_GENERIC(temp6, ((v2s*)addr6)[i],inF_temp2);
  //             SDOTP_GENERIC(temp7, ((v2s*)addr7)[i],inF_temp2);
  // #endif
            // }
            }
  #ifdef FMINTILING
          if(_layer->attributes[LAY_CONV_IN]%4!=0) { // add contribution of left over input channel (input channels not multiple of 4)
               v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
               // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
               asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
               // asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
               
               // MANUAL loop unfolding
               // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
               // __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+o_rel)) + i], temp[o_rel]);
             #if OUTPUTBUFFER > 2
              SDOTP_GENERIC(temp, ((v2s*)addr0)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 3
              SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 4
              SDOTP_GENERIC(temp2, ((v2s*)addr2)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 5
              SDOTP_GENERIC(temp3, ((v2s*)addr3)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 6
              SDOTP_GENERIC(temp4, ((v2s*)addr4)[i], inF_temp);
               #endif
               #if OUTPUTBUFFER > 7
              SDOTP_GENERIC(temp5, ((v2s*)addr5)[i], inF_temp);
               #endif
              SDOTP_GENERIC(temp6, ((v2s*)addr6)[i],inF_temp);
              SDOTP_GENERIC(temp7, ((v2s*)addr7)[i],inF_temp);
             }

  #endif
                  
              param_id_base  += kernel_W_offset; // filter tap
              param_id_base1 += kernel_W_offset; // filter tap
              feat_id_base   +=  _layer->attributes[LAY_CONV_IN]/2;
            }

                  #if OUTPUTBUFFER > 2
            outFeatures_ptr[(outFeaturesPerTile*c_out+0)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp) >> q_fraqP1);
                  #endif
                  #if OUTPUTBUFFER > 3
            outFeatures_ptr[(outFeaturesPerTile*c_out+1)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp1) >> q_fraqP1);
                  #endif
                  #if OUTPUTBUFFER > 4
            outFeatures_ptr[(outFeaturesPerTile*c_out+2)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp2) >> q_fraqP1);
                  #endif
                  #if OUTPUTBUFFER > 5
            outFeatures_ptr[(outFeaturesPerTile*c_out+3)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp3) >> q_fraqP1);
                  #endif
                  #if OUTPUTBUFFER > 6
            outFeatures_ptr[(outFeaturesPerTile*c_out+4)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp4) >> q_fraqP1);
                  #endif
                  #if OUTPUTBUFFER > 7
            outFeatures_ptr[(outFeaturesPerTile*c_out+5)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp5) >> q_fraqP1);
                  #endif

            outFeatures_ptr[(outFeaturesPerTile*c_out+OUTPUTBUFFER-2)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp6) >> q_fraqP1);
            outFeatures_ptr[(outFeaturesPerTile*c_out+OUTPUTBUFFER-1)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp7) >> q_fraqP1);
            param_kw_base += kernel_H_offset;
            feat_kw_base  += w_im_out*_layer->attributes[LAY_CONV_IN]/2;
          }
        }
      }
      param_kh_base += outFeaturesPerTile * output_channel_offset;
    }  
    break;
     #endif
     #if OUTPUTBUFFER > 4
    case 4:
    for(int c_out = 0; c_out < outFeatureTiles; c_out++) {
     for(int h_out =0; h_out < h_im_out; h_out++) 
     {
       for(int w_out=0;w_out<w_im_out; w_out++)
       {

           int kh_slide_start = Max(-h_out, -h_ker_half);                  // Handle borders
           int kh_slide_stop = Min(h_im_out-1-h_out,h_ker_half);           // Handle borders
           temp = bias_ptr[outFeaturesPerTile*c_out] << q_fraqP1;
           temp1 =  bias_ptr[(outFeaturesPerTile*c_out+1)] << q_fraqP1;
           temp = bias_ptr[outFeaturesPerTile*c_out] << q_fraqP1;
           temp1 = bias_ptr[outFeaturesPerTile*c_out+1] << q_fraqP1;
           temp6 = bias_ptr[outFeaturesPerTile*c_out+2]<<(q_fraqP1);
           temp7 = bias_ptr[outFeaturesPerTile*c_out+3]<<(q_fraqP1);

           unsigned int param_kw_base = param_kh_base \
           + (kh_slide_start+h_ker_half) * kernel_H_offset;
           unsigned int feat_kw_base = (h_out+kh_slide_start)*w_im_out*_layer->attributes[LAY_CONV_IN]/2;
           for(int kh=kh_slide_start; kh <= kh_slide_stop;kh++)
           {
              int kw_slide_start = Max(-w_out, -w_ker_half);           // Handle borders
              int kw_slide_stop = Min(w_im_out-1-w_out,w_ker_half);           // Handle borders
              
              int param_id_base = param_kw_base \
                                            +(kw_slide_start+w_ker_half)*kernel_W_offset; // filter tap
                                            int param_id_base1 = param_id_base+output_channel_offset;
                                            int feat_id_base  = feat_kw_base \
                                            +(w_out+kw_slide_start)* _layer->attributes[LAY_CONV_IN]/2;
                                            for(int kw=kw_slide_start; kw <= kw_slide_stop;kw++)
                                            {
                                             addr0  = (uint32_t) &((v2s*)param_simd)[param_id_base];
                                             addr1  = (uint32_t) &((v2s*)param_simd)[param_id_base+1*output_channel_offset];
                                             addr6  = (uint32_t) &((v2s*)param_simd)[param_id_base+output_channel_offset*2];
                                             addr7  = (uint32_t) &((v2s*)param_simd)[param_id_base+output_channel_offset*3];

                 // asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload first weight
                 // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload first weight
                 v2s* in_addr = &((v2s*)inFeatures)[feat_id_base];
                 v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
                 v2s inF_temp2;// = ((v2s*)inFeatures)[2*i+1];


                 for(int i=0; i <  c_in_max;i++) // i=c_in
                 {
                 // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
                 asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
                 // #ifdef FMINTILING
                 //    asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
                 // #endif
                  // loop unfolding
                  // temp  = __SUMDOTP2(((v2s*)inFeatures)[feat_id_base + i], \
                  //  param_simd[param_id_base + i], temp);
                  // temp1 = __SUMDOTP2(((v2s*)inFeatures)[feat_id_base + i], \
                  //  param_simd[param_id_base1 + i], temp1);
                   // v2s inF_temp = ((v2s*)inFeatures)[feat_id_base + i];
                   // temp  = __SUMDOTP2(inF_temp, \
                   // param_simd[param_id_base + i], temp);
                   // asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp),  "+r" (addr0) : "r" (inF_temp) ); 
                   // temp1 = __SUMDOTP2(inF_temp, \
                   // param_simd[param_id_base1 + i], temp1);
                   // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr1) : "r" (inF_temp) ); 
                   // v2s inF_temp = ((v2s*)inFeatures)[feat_id_base + i];

                    SDOTP_GENERIC(temp, ((v2s*)addr0)[i], inF_temp);
                    SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp);
                    SDOTP_GENERIC(temp6, ((v2s*)addr6)[i], inF_temp);
                    SDOTP_GENERIC(temp7, ((v2s*)addr7)[i], inF_temp);
                    // #ifdef FMINTILING
                    //   SDOTP_GENERIC(temp, ((v2s*)addr0)[i], inF_temp2);
                    //   SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp2);
                    //   SDOTP_GENERIC(temp6, ((v2s*)addr6)[i], inF_temp2);
                    //   SDOTP_GENERIC(temp7, ((v2s*)addr7)[i], inF_temp2);
                    // #endif

                  }
  // #ifdef FMINTILING
  //         if(_layer->attributes[LAY_CONV_IN]%4!=0) { // add contribution of left over input channel (input channels not multiple of 4)
  //              v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
  //              // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
  //              asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
  //              // asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
               
  //              // MANUAL loop unfolding
  //              // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
  //              // __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+o_rel)) + i], temp[o_rel]);
  //             SDOTP_GENERIC(temp, ((v2s*)addr0)[i], inF_temp);
  //             SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp);
  //             SDOTP_GENERIC(temp6, ((v2s*)addr6)[i],inF_temp);
  //             SDOTP_GENERIC(temp7, ((v2s*)addr7)[i],inF_temp);
  //            }

  // #endif
              param_id_base  += kernel_W_offset; // filter tap
              param_id_base1 += kernel_W_offset; // filter tap
              feat_id_base   +=  _layer->attributes[LAY_CONV_IN]/2;
            }

            outFeatures_ptr[(outFeaturesPerTile*c_out+0)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp) >> q_fraqP1);
            outFeatures_ptr[(outFeaturesPerTile*c_out+1)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp1) >> q_fraqP1);
            outFeatures_ptr[(outFeaturesPerTile*c_out+2)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp6) >> q_fraqP1);
            outFeatures_ptr[(outFeaturesPerTile*c_out+3)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp7) >> q_fraqP1);

            param_kw_base += kernel_H_offset;
            feat_kw_base  += w_im_out*_layer->attributes[LAY_CONV_IN]/2;
          }
        }
      }
      param_kh_base += outFeaturesPerTile * output_channel_offset;
    }  
    break;
#endif
    case 2:
    for(int c_out = 0; c_out < outFeatureTiles; c_out++) {
     for(int h_out =0; h_out < h_im_out; h_out++) 
     {
       for(int w_out=0;w_out<w_im_out; w_out++)
       {

           int kh_slide_start = Max(-h_out, -h_ker_half);                  // Handle borders
           int kh_slide_stop = Min(h_im_out-1-h_out,h_ker_half);           // Handle borders
           
           temp6 = bias_ptr[outFeaturesPerTile*c_out+0]<<(q_fraqP1);
           temp7 = bias_ptr[outFeaturesPerTile*c_out+1]<<(q_fraqP1);

           unsigned int param_kw_base = param_kh_base \
           + (kh_slide_start+h_ker_half) * kernel_H_offset;
           unsigned int feat_kw_base = (h_out+kh_slide_start)*w_im_out*_layer->attributes[LAY_CONV_IN]/2;
           for(int kh=kh_slide_start; kh <= kh_slide_stop;kh++)
           {
              int kw_slide_start = Max(-w_out, -w_ker_half);           // Handle borders
              int kw_slide_stop = Min(w_im_out-1-w_out,w_ker_half);           // Handle borders
              
              int param_id_base = param_kw_base \
                                            +(kw_slide_start+w_ker_half)*kernel_W_offset; // filter tap
                                            int param_id_base1 = param_id_base+output_channel_offset;
                                            int feat_id_base  = feat_kw_base \
                                            +(w_out+kw_slide_start)* _layer->attributes[LAY_CONV_IN]/2;
                                            for(int kw=kw_slide_start; kw <= kw_slide_stop;kw++)
                                            {

                                             addr6  = (uint32_t) &((v2s*)param_simd)[param_id_base+output_channel_offset*0];
                                             addr7  = (uint32_t) &((v2s*)param_simd)[param_id_base+output_channel_offset*1];

                 // asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr6) : "r" (x0) ); // preload first weight
                 // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr7) : "r" (x0) ); // preload first weight

                 v2s* in_addr = &((v2s*)inFeatures)[feat_id_base];
                 v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
                 v2s inF_temp2;// = ((v2s*)inFeatures)[2*i+1];


                 for(int i=0; i <  c_in_max;i++) // i=c_in
                 {
                 // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
                 asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
                 // #ifdef FMINTILING
                 //    asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
                 // #endif
                  // loop unfolding
                  // temp  = __SUMDOTP2(((v2s*)inFeatures)[feat_id_base + i], \
                  //  param_simd[param_id_base + i], temp);
                  // temp1 = __SUMDOTP2(((v2s*)inFeatures)[feat_id_base + i], \
                  //  param_simd[param_id_base1 + i], temp1);
                   // v2s inF_temp = ((v2s*)inFeatures)[feat_id_base + i];
                   // temp  = __SUMDOTP2(inF_temp, \
                   // param_simd[param_id_base + i], temp);
                   // asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp),  "+r" (addr0) : "r" (inF_temp) ); 
                   // temp1 = __SUMDOTP2(inF_temp, \
                   // param_simd[param_id_base1 + i], temp1);
                   // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr1) : "r" (inF_temp) ); 
                   // v2s inF_temp = ((v2s*)inFeatures)[feat_id_base + i];


                    SDOTP_GENERIC(temp6, ((v2s*)addr6)[i], inF_temp);
                    SDOTP_GENERIC(temp7, ((v2s*)addr7)[i], inF_temp);
                    // #ifdef FMINTILING
                    // SDOTP_GENERIC(temp6, ((v2s*)addr6)[i], inF_temp2);
                    // SDOTP_GENERIC(temp7, ((v2s*)addr7)[i], inF_temp2);
                    // #endif

                  }
  //             #ifdef FMINTILING
  //         if(_layer->attributes[LAY_CONV_IN]%4!=0) { // add contribution of left over input channel (input channels not multiple of 4)
  //              v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
  //              // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
  //              asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
  //              // asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
               
  //              // MANUAL loop unfolding
  //              // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
  //              // __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+o_rel)) + i], temp[o_rel]);
  //             SDOTP_GENERIC(temp6, ((v2s*)addr6)[i],inF_temp);
  //             SDOTP_GENERIC(temp7, ((v2s*)addr7)[i],inF_temp);
  //            }

  // #endif
              param_id_base  += kernel_W_offset; // filter tap
              param_id_base1 += kernel_W_offset; // filter tap
              feat_id_base   +=  _layer->attributes[LAY_CONV_IN]/2;
            }

            outFeatures_ptr[(outFeaturesPerTile*c_out+0)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp6) >> q_fraqP1);
            outFeatures_ptr[(outFeaturesPerTile*c_out+1)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp7) >> q_fraqP1);
            param_kw_base += kernel_H_offset;
            feat_kw_base  += w_im_out*_layer->attributes[LAY_CONV_IN]/2;
          }
        }
      }
      param_kh_base += outFeaturesPerTile * output_channel_offset;
    }  
    break;
    case 1:
    for(int c_out = 0; c_out < outFeatureTiles; c_out++) {
     for(int h_out =0; h_out < h_im_out; h_out++) 
     {
       for(int w_out=0;w_out<w_im_out; w_out++)
       {

           int kh_slide_start = Max(-h_out, -h_ker_half);                  // Handle borders
           int kh_slide_stop = Min(h_im_out-1-h_out,h_ker_half);           // Handle borders
           
           temp6 = bias_ptr[outFeaturesPerTile*c_out+0]<<(q_fraqP1);
          // temp7 = bias_ptr[outFeaturesPerTile*c_out+1]<<(q_fraqP1);

           unsigned int param_kw_base = param_kh_base \
           + (kh_slide_start+h_ker_half) * kernel_H_offset;
           unsigned int feat_kw_base = (h_out+kh_slide_start)*w_im_out*_layer->attributes[LAY_CONV_IN]/2;
           for(int kh=kh_slide_start; kh <= kh_slide_stop;kh++)
           {
              int kw_slide_start = Max(-w_out, -w_ker_half);           // Handle borders
              int kw_slide_stop = Min(w_im_out-1-w_out,w_ker_half);           // Handle borders
              
              int param_id_base = param_kw_base \
                                            +(kw_slide_start+w_ker_half)*kernel_W_offset; // filter tap
                                            int param_id_base1 = param_id_base+output_channel_offset;
                                            int feat_id_base  = feat_kw_base \
                                            +(w_out+kw_slide_start)* _layer->attributes[LAY_CONV_IN]/2;
                                            for(int kw=kw_slide_start; kw <= kw_slide_stop;kw++)
                                            {

                 // addr6  = &((v2s*)param_simd)[param_id_base+output_channel_offset*0];
                #ifdef FMINTILING
                for(int i=0; i <  2*c_in_max;i++) // i=c_in                              
                #else
                for(int i=0; i <  c_in_max;i++) // i=c_in                             
                #endif
                 
                 {
                  temp6  = __SUMDOTP2(((v2s*)inFeatures)[feat_id_base + i], \
                   param_simd[param_id_base + i], temp6);
                }
              param_id_base  += kernel_W_offset; // filter tap
              param_id_base1 += kernel_W_offset; // filter tap
              feat_id_base   +=  _layer->attributes[LAY_CONV_IN]/2;
            }

            outFeatures_ptr[(outFeaturesPerTile*c_out+0)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp6) >> q_fraqP1);
            param_kw_base += kernel_H_offset;
            feat_kw_base  += w_im_out*_layer->attributes[LAY_CONV_IN]/2;
          }
        }
      }
      param_kh_base += outFeaturesPerTile * output_channel_offset;
    }  
    break;
  }


      // move pointers for next iteration
  bias_ptr                = &bias_ptr[outFeaturesPerTile*outFeatureTiles];
      // param_simd              = &((v2s*)param_simd)[output_channel_offset*(outFeatureTiles*outFeaturesPerTile)];
  outFeatures_ptr         = &outFeatures_ptr[outFeaturesPerTile*outFeatureTiles*h_im_out*w_im_out];
  outFeaturesSize_remain -= outFeatureTiles*outFeaturesPerTile;

  if (outFeaturesSize_remain==0) break;
}
return 0;
}


 /////////////////////////////////////////////////////////////
 // ______  _______ _______ _______ _     _        _______  //
 // |     \ |______ |______ |_____| |     | |         |     //
 // |_____/ |______ |       |     | |_____| |_____    |     //
 //                                                         //
 /////////////////////////////////////////////////////////////
  #else // no vliw
/** @brief Calculates a 2D Convolution Layer PULP+VLIW+(SIMD)
 *  Implements the 2D convolution layer (standard implementation)
 *  Supports the following configurations:
 *  => No VLIW Support (see special implementation)
 *  => FMOUTINILING false (not support, not benefitial)
 *  => FMOUTTILING false/true
 *  => FixedPt, SIMD || FLOAT
 *  => MANUALLOOPUNFOLDING not implemented
 *
 *  @param _layer Layer Properties
 *  @param h_im Image Height
 *  @param w_im Image Width
 *  @param inFeatures Input Feature Map
 *  @param outFeatures Output Feature Map
 */
int NOINLINE Conv2dLayer (
// Layer Attributes
  struct layer * _layer,
  int h_im,
  int w_im,
  data_t * __restrict__ inFeatures,
  data_t * __restrict__ outFeatures) {
  //          printf("delete, just for test 4");
 int h_im_out = h_im;
 int w_im_out = w_im;
 int h_ker_half = (int)(_layer->attributes[LAY_CONV_KER]/2);
   int w_ker_half = h_ker_half; // TODO: symmetric kernel only
#ifdef SIMD
   unsigned int output_channel_offset = _layer->attributes[LAY_CONV_KER]*_layer->attributes[LAY_CONV_KER]*_layer->attributes[LAY_CONV_IN]/2;
   unsigned int kernel_H_offset = _layer->attributes[LAY_CONV_KER]*_layer->attributes[LAY_CONV_IN]/2;
   unsigned int kernel_W_offset = _layer->attributes[LAY_CONV_IN]/2;
#else
   unsigned int output_channel_offset = _layer->attributes[LAY_CONV_KER]*_layer->attributes[LAY_CONV_KER]*_layer->attributes[LAY_CONV_IN];
   unsigned int kernel_H_offset = _layer->attributes[LAY_CONV_KER]*_layer->attributes[LAY_CONV_IN];
   unsigned int kernel_W_offset = _layer->attributes[LAY_CONV_IN];
#endif
   int c_in_max;
   c_in_max = _layer->attributes[LAY_CONV_IN];
   #ifdef FixedPt
   #ifdef SIMD
   c_in_max = _layer->attributes[LAY_CONV_IN]/2;
   v2s* param_simd = (v2s*) _layer->parameters[CONV_WGHT];
   #endif
   #endif
   const int outFeaturesPerTile = 1;
   for(int c_out = 0; c_out < _layer->attributes[LAY_CONV_OUT]/outFeaturesPerTile; c_out++) {
    for(int h_out =0; h_out < h_im_out; h_out++) 
    {
     for(int w_out=0;w_out<w_im_out; w_out++)
     {

               int kh_slide_start = Max(-h_out, -h_ker_half);           // Handle borders
               int kh_slide_stop = Min(h_im_out-1-h_out,h_ker_half);           // Handle borders
#ifdef SIMD
               int32_t temp = 0;

               int32_t temp1 = 0;
#else
               int32_t  temp = 0;
#endif
               unsigned int param_kh_base = outFeaturesPerTile*c_out * output_channel_offset;
               for(int kh=kh_slide_start; kh <= kh_slide_stop;kh++)
               {
                  int kw_slide_start = Max(-w_out, -w_ker_half);           // Handle borders
                  int kw_slide_stop = Min(w_im_out-1-w_out,w_ker_half);           // Handle borders


                  unsigned int param_kw_base = param_kh_base \
                  + (kh+h_ker_half) * kernel_H_offset;
                  unsigned int feat_kw_base  = (h_out+kh)*w_im_out*_layer->attributes[LAY_CONV_IN]/2;
                  for(int kw=kw_slide_start; kw <= kw_slide_stop;kw++)
                  {
                   int param_id_base = param_kw_base \
                                                +(kw+w_ker_half)*kernel_W_offset; // filter tap
                                                int feat_id_base  = feat_kw_base \
                                                +(w_out+kw)* _layer->attributes[LAY_CONV_IN]/2;
                     for(int i=0; i <  c_in_max;i++) // i=c_in
                     {
                        unsigned int param_id; // = c_out * output_channel_offset \
                                                + c_in  \
                                                + (kh+h_ker_half) * kernel_H_offset\
                                                 +(kw+w_ker_half)*kernel_W_offset; // filter tap
                        unsigned int feat_id; //  = (h_out+kh)*w_im_out* _layer->attributes[LAY_CONV_IN]+(w_out+kw)* _layer->attributes[LAY_CONV_IN] + c_in;
                        //printf("%i\n", param_id);

#ifdef FixedPt
#ifdef SIMD
                        // param_id = c_out * output_channel_offset \
                        //                         + c_in  \
                        //                         + (kh+h_ker_half) * kernel_H_offset\
                        //                         +(kw+w_ker_half)*kernel_W_offset; // filter tap
                        // feat_id  = (h_out+kh)*w_im_out*_layer->attributes[LAY_CONV_IN]/2\
                        //            +(w_out+kw)* _layer->attributes[LAY_CONV_IN]/2 + c_in;
                        // param_id = param_id_base + c_in;
                        // feat_id  = feat_id_base + c_in;

#ifndef ASIP
                        temp = __SUMDOTP2(((v2s*)inFeatures)[feat_id_base + i], \
                         param_simd[param_id_base + i], temp);
#else // ASIP
                        temp = temp + ((v2s*)inFeatures)[feat_id_base + i] * param_simd[param_id_base + i];
#endif // ASIP
#else // not SIMD
                        temp +=  ((_layer->parameters[CONV_WGHT][(c_out * output_channel_offset \
                          + i  \
                          + (kh+h_ker_half) * kernel_H_offset\
                          +(kw+w_ker_half)*kernel_W_offset)] \
                        * inFeatures[((h_out+kh)*w_im_out* _layer->attributes[LAY_CONV_IN]\
                          +(w_out+kw)* _layer->attributes[LAY_CONV_IN] + i)]));
                         // printf("temp=%x+=%x*%x\n", temp, _layer->parameters[CONV_WGHT][(c_out * output_channel_offset \
                                                + i  \
                                                + (kh+h_ker_half) * kernel_H_offset\
                                                 +(kw+w_ker_half)*kernel_W_offset)], inFeatures[((h_out+kh)*w_im_out* _layer->attributes[LAY_CONV_IN]\
                                  +(w_out+kw)* _layer->attributes[LAY_CONV_IN] + i)]);
#endif // SIMD
#else // FixedPt
                                  temp +=  ((_layer->parameters[CONV_WGHT][(c_out * output_channel_offset \
                                    + i  \
                                    + (kh+h_ker_half) * kernel_H_offset\
                                    +(kw+w_ker_half)*kernel_W_offset)] \
                                  * inFeatures[((h_out+kh)*w_im_out* _layer->attributes[LAY_CONV_IN]\
                                  +(w_out+kw)* _layer->attributes[LAY_CONV_IN] + i)]));// >> (q_fraqP1));

#endif // FixedPt

                                }

                              }

                            }




#ifdef SIMD
                            outFeatures[outFeaturesPerTile*c_out*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp) >> q_fraqP1)+ _layer->parameters[CONV_BIAS][outFeaturesPerTile*c_out];
               // outFeatures[(outFeaturesPerTile*c_out+1)*h_im_out*w_im_out+h_out*w_im_out+w_out] = (((int32_t)temp1) >> q_fraqP1)+ _layer->parameters[CONV_BIAS][(outFeaturesPerTile*c_out+1)];
#else
                            outFeatures[c_out*h_im_out*w_im_out+h_out*w_im_out+w_out] = (temp >> (q_fraqP1))+ _layer->parameters[CONV_BIAS][c_out];
#endif // end SIMD
                          }
                        }
                      }

                      return 0;
                    }
 #endif
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
////  _____               _     _                        ///// 
//// |_   _|_      _____ | |   (_)_ __   ___  __ _ _ __  /////
////   | | \ \ /\ / / _ \| |   | | '_ \ / _ \/ _` | '__| /////
////   | |  \ V  V / (_) | |___| | | | |  __/ (_| | |    /////
////   |_|   \_/\_/ \___/|_____|_|_| |_|\___|\__,_|_|    /////
////                                                     /////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
// _______ _______ _____  _____       ___________      _______ _____        _____ __   _  ______
// |_____| |______   |   |_____]      |______|            |      |   |        |   | \  | |  ____
// |     | ______| __|__ |            |______|            |    __|__ |_____ __|__ |  \_| |_____|
//
/////////////////////////////////////////////////////////////// 

/** @brief Calculates two Linear Layers and accumulates them on-the-fly. (ASIP with OutputFMTiling)
 *  This is a helper function for efficient LSTM implementation. It calculates two linear layers in
 *  parallel and accumulates them on-the-fly.
 *  Supported Configurations:
 *  ASIP&SIMD&FMOUTTILING
 *  FMOUTTILING of 8,4,2,1 tile size
 *
 *  @param inFeaturesSize1 Input FM size for layer 1
 *  @param inFeaturesSize2 Input FM size for layer 2
 *  @param outFeaturesSize Output FM size
 *  @param activationFunction Type of activation Function (tanh, sigmoid, none)
 *  @param weight1 pointer to weight parameters of layer 1
 *  @param weight2 pointer to weight parameters of layer 2
 *  @param bias1 pointer to bias parametsr of layer 1
 *  @param bias2 pointer to bias parametsr of layer 2
 *  @param inFeatures1 pointer to input FM of layer 1
 *  @param inFeatures2 pointer to input FM of layer 2
 *  @param outFeatures pointer where to write to the output FM
 */
#ifdef FixedPt
#if defined FixedPt && defined FMOUTTILING && !defined VLIWEXT && defined ASIP
                    void NOINLINE TwoLinearLayersAccumulate (
        // Layer Attributes
                      int inFeaturesSize1, int inFeaturesSize2, int outFeaturesSize, int activationFunction, 
        // Layer Parameters
                      data_t * __restrict__ weight1,
                      data_t * __restrict__ weight2,
                      data_t * __restrict__ bias1,
                      data_t * __restrict__ bias2,
        // Input and Output Features
                      data_t * __restrict__ inFeatures1,
                      data_t * __restrict__ inFeatures2,
                      data_t * __restrict__ outFeatures)
                    {
                      int tileOptions[] = {10,8,4,2, 1};

                      int outFeaturesPerTile = 1;
// find appropriate tiling (TODO: do it like in RISCY)
                      for(unsigned int i=0; i<sizeof(tileOptions)/sizeof(int); i++) {
                       if(outFeaturesSize % tileOptions[i] == 0) {
                        outFeaturesPerTile = tileOptions[i];
                        break;
                      }
                    }
                    int outFeatureTiles = outFeaturesSize/outFeaturesPerTile;


                    int inFeaturesSize1P2=inFeaturesSize1/2;
                    int inFeaturesSize2P2=inFeaturesSize2/2;
                    switch(outFeaturesPerTile) {
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                      case 10:
                      for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) {
                        int32_t register_attribute ra;int32_t register_attribute rb;int32_t register_attribute rc;int32_t register_attribute rd;int32_t register_attribute re;
                        int32_t register_attribute rf;int32_t register_attribute rg;int32_t register_attribute rh;int32_t register_attribute ri;int32_t register_attribute rj;
                        ra = ((int32_t)bias1[o_tile*outFeaturesPerTile+0]+(int32_t)bias2[o_tile*outFeaturesPerTile+0])<<(q_fraqP1);
                        rb = ((int32_t)bias1[o_tile*outFeaturesPerTile+1]+(int32_t)bias2[o_tile*outFeaturesPerTile+1])<<(q_fraqP1);
                        rc = ((int32_t)bias1[o_tile*outFeaturesPerTile+2]+(int32_t)bias2[o_tile*outFeaturesPerTile+2])<<(q_fraqP1);
                        rd = ((int32_t)bias1[o_tile*outFeaturesPerTile+3]+(int32_t)bias2[o_tile*outFeaturesPerTile+3])<<(q_fraqP1);
                        re = ((int32_t)bias1[o_tile*outFeaturesPerTile+4]+(int32_t)bias2[o_tile*outFeaturesPerTile+4])<<(q_fraqP1);
                        rf = ((int32_t)bias1[o_tile*outFeaturesPerTile+5]+(int32_t)bias2[o_tile*outFeaturesPerTile+5])<<(q_fraqP1);
                        rg = ((int32_t)bias1[o_tile*outFeaturesPerTile+6]+(int32_t)bias2[o_tile*outFeaturesPerTile+6])<<(q_fraqP1);
                        rh = ((int32_t)bias1[o_tile*outFeaturesPerTile+7]+(int32_t)bias2[o_tile*outFeaturesPerTile+7])<<(q_fraqP1);
                        ri = ((int32_t)bias1[o_tile*outFeaturesPerTile+8]+(int32_t)bias2[o_tile*outFeaturesPerTile+8])<<(q_fraqP1);
                        rj = ((int32_t)bias1[o_tile*outFeaturesPerTile+9]+(int32_t)bias2[o_tile*outFeaturesPerTile+9])<<(q_fraqP1);

                        for(int i=0; i<inFeaturesSize1P2; i++) { 
                          v2s inF_temp = ((v2s*)inFeatures1)[i];         
                          SDOTP_GENERIC(ra, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+0)) + i]);
                          SDOTP_GENERIC(rb, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+1)) + i]);
                          SDOTP_GENERIC(rc, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+2)) + i]);
                          SDOTP_GENERIC(rd, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+3)) + i]);
                          SDOTP_GENERIC(re, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+4)) + i]);
                          SDOTP_GENERIC(rf, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+5)) + i]);
                          SDOTP_GENERIC(rg, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+6)) + i]);
                          SDOTP_GENERIC(rh, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+7)) + i]);
                          SDOTP_GENERIC(ri, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+8)) + i]);
                          SDOTP_GENERIC(rj, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+9)) + i]);
      } // for(int i=0; i<inFeaturesSizeP2; i++)

      for(int i=0; i<inFeaturesSize2P2; i++)
      {
        v2s inF_temp = ((v2s*)inFeatures2)[i];         
        SDOTP_GENERIC(ra, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+0)) + i]);
        SDOTP_GENERIC(rb, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+1)) + i]);
        SDOTP_GENERIC(rc, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+2)) + i]);
        SDOTP_GENERIC(rd, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+3)) + i]);
        SDOTP_GENERIC(re, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+4)) + i]);
        SDOTP_GENERIC(rf, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+5)) + i]);
        SDOTP_GENERIC(rg, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+6)) + i]);
        SDOTP_GENERIC(rh, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+7)) + i]);
        SDOTP_GENERIC(ri, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+8)) + i]);
        SDOTP_GENERIC(rj, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+9)) + i]);

      }

#ifdef DOACTONTHEFLY
      ra = ra>>(q_fraqP1);
      rb = rb>>(q_fraqP1);
      rc = rc>>(q_fraqP1);
      rd = rd>>(q_fraqP1);
      re = re>>(q_fraqP1);
      rf = rf>>(q_fraqP1);
      rg = rg>>(q_fraqP1);
      rh = rh>>(q_fraqP1);
      ri = ri>>(q_fraqP1);
      rj = rj>>(q_fraqP1);
      switch(activationFunction) {
        case ACT_NONE: outFeatures[(o_tile*outFeaturesPerTile+0)] = ra; 
        outFeatures[(o_tile*outFeaturesPerTile+1)] = rb;
        outFeatures[(o_tile*outFeaturesPerTile+2)] = rc; 
        outFeatures[(o_tile*outFeaturesPerTile+3)] = rd;
        outFeatures[(o_tile*outFeaturesPerTile+4)] = re; 
        outFeatures[(o_tile*outFeaturesPerTile+5)] = rf;
        outFeatures[(o_tile*outFeaturesPerTile+6)] = rg; 
        outFeatures[(o_tile*outFeaturesPerTile+7)] = rh;
        outFeatures[(o_tile*outFeaturesPerTile+8)] = ri; 
        outFeatures[(o_tile*outFeaturesPerTile+9)] = rj; break;
        case ACT_TANH: outFeatures[(o_tile*outFeaturesPerTile+0)] = generic_tanh(ra); 
        outFeatures[(o_tile*outFeaturesPerTile+1)] = generic_tanh(rb);
        outFeatures[(o_tile*outFeaturesPerTile+2)] = generic_tanh(rc); 
        outFeatures[(o_tile*outFeaturesPerTile+3)] = generic_tanh(rd);
        outFeatures[(o_tile*outFeaturesPerTile+4)] = generic_tanh(re); 
        outFeatures[(o_tile*outFeaturesPerTile+5)] = generic_tanh(rf);
        outFeatures[(o_tile*outFeaturesPerTile+6)] = generic_tanh(rg); 
        outFeatures[(o_tile*outFeaturesPerTile+7)] = generic_tanh(rh);
        outFeatures[(o_tile*outFeaturesPerTile+8)] = generic_tanh(ri); 
        outFeatures[(o_tile*outFeaturesPerTile+9)] = generic_tanh(rj); break;
        case ACT_SIG:  outFeatures[(o_tile*outFeaturesPerTile+0)] = generic_sig(ra); 
        outFeatures[(o_tile*outFeaturesPerTile+1)] = generic_sig(rb);
        outFeatures[(o_tile*outFeaturesPerTile+2)] = generic_sig(rc); 
        outFeatures[(o_tile*outFeaturesPerTile+3)] = generic_sig(rd);
        outFeatures[(o_tile*outFeaturesPerTile+4)] = generic_sig(re); 
        outFeatures[(o_tile*outFeaturesPerTile+5)] = generic_sig(rf);
        outFeatures[(o_tile*outFeaturesPerTile+6)] = generic_sig(rg); 
        outFeatures[(o_tile*outFeaturesPerTile+7)] = generic_sig(rh);
        outFeatures[(o_tile*outFeaturesPerTile+8)] = generic_sig(ri); 
        outFeatures[(o_tile*outFeaturesPerTile+9)] = generic_sig(rj); break;
      }
#else
      outFeatures[(o_tile*outFeaturesPerTile+0)] = ra>>(q_fraqP1); 
      outFeatures[(o_tile*outFeaturesPerTile+1)] = rb>>(q_fraqP1);
      outFeatures[(o_tile*outFeaturesPerTile+2)] = rc>>(q_fraqP1); 
      outFeatures[(o_tile*outFeaturesPerTile+3)] = rd>>(q_fraqP1);
      outFeatures[(o_tile*outFeaturesPerTile+4)] = re>>(q_fraqP1); 
      outFeatures[(o_tile*outFeaturesPerTile+5)] = rf>>(q_fraqP1);
      outFeatures[(o_tile*outFeaturesPerTile+6)] = rg>>(q_fraqP1); 
      outFeatures[(o_tile*outFeaturesPerTile+7)] = rh>>(q_fraqP1);
      outFeatures[(o_tile*outFeaturesPerTile+8)] = ri>>(q_fraqP1); 
      outFeatures[(o_tile*outFeaturesPerTile+9)] = rj>>(q_fraqP1); 
#endif
    }
break; // case 10
case 8:
for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) {
  int32_t register_attribute ra;int32_t register_attribute rb;int32_t register_attribute rc;int32_t register_attribute rd;int32_t register_attribute re;
  int32_t register_attribute rf;int32_t register_attribute rg;int32_t register_attribute rh;int32_t register_attribute ri;int32_t register_attribute rj;
  ra = ((int32_t)bias1[o_tile*outFeaturesPerTile+0]+(int32_t)bias2[o_tile*outFeaturesPerTile+0])<<(q_fraqP1);
  rb = ((int32_t)bias1[o_tile*outFeaturesPerTile+1]+(int32_t)bias2[o_tile*outFeaturesPerTile+1])<<(q_fraqP1);
  rc = ((int32_t)bias1[o_tile*outFeaturesPerTile+2]+(int32_t)bias2[o_tile*outFeaturesPerTile+2])<<(q_fraqP1);
  rd = ((int32_t)bias1[o_tile*outFeaturesPerTile+3]+(int32_t)bias2[o_tile*outFeaturesPerTile+3])<<(q_fraqP1);
  re = ((int32_t)bias1[o_tile*outFeaturesPerTile+4]+(int32_t)bias2[o_tile*outFeaturesPerTile+4])<<(q_fraqP1);
  rf = ((int32_t)bias1[o_tile*outFeaturesPerTile+5]+(int32_t)bias2[o_tile*outFeaturesPerTile+5])<<(q_fraqP1);
  rg = ((int32_t)bias1[o_tile*outFeaturesPerTile+6]+(int32_t)bias2[o_tile*outFeaturesPerTile+6])<<(q_fraqP1);
  rh = ((int32_t)bias1[o_tile*outFeaturesPerTile+7]+(int32_t)bias2[o_tile*outFeaturesPerTile+7])<<(q_fraqP1);


  for(int i=0; i<inFeaturesSize1P2; i++)
  {

    v2s inF_temp = ((v2s*)inFeatures1)[i];         
    SDOTP_GENERIC(ra, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+0)) + i]);
    SDOTP_GENERIC(rb, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+1)) + i]);
    SDOTP_GENERIC(rc, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+2)) + i]);
    SDOTP_GENERIC(rd, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+3)) + i]);
    SDOTP_GENERIC(re, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+4)) + i]);
    SDOTP_GENERIC(rf, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+5)) + i]);
    SDOTP_GENERIC(rg, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+6)) + i]);
    SDOTP_GENERIC(rh, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+7)) + i]);
         } // for(int i=0; i<inFeaturesSizeP2; i++)

         for(int i=0; i<inFeaturesSize2P2; i++)
         {
          v2s inF_temp = ((v2s*)inFeatures2)[i];         
          SDOTP_GENERIC(ra, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+0)) + i]);
          SDOTP_GENERIC(rb, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+1)) + i]);
          SDOTP_GENERIC(rc, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+2)) + i]);
          SDOTP_GENERIC(rd, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+3)) + i]);
          SDOTP_GENERIC(re, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+4)) + i]);
          SDOTP_GENERIC(rf, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+5)) + i]);
          SDOTP_GENERIC(rg, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+6)) + i]);
          SDOTP_GENERIC(rh, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+7)) + i]);
        }

#ifdef DOACTONTHEFLY
        ra = ra>>(q_fraqP1);
        rb = rb>>(q_fraqP1);
        rc = rc>>(q_fraqP1);
        rd = rd>>(q_fraqP1);
        re = re>>(q_fraqP1);
        rf = rf>>(q_fraqP1);
        rg = rg>>(q_fraqP1);
        rh = rh>>(q_fraqP1);
        switch(activationFunction) {
          case ACT_NONE: outFeatures[(o_tile*outFeaturesPerTile+0)] = ra; 
          outFeatures[(o_tile*outFeaturesPerTile+1)] = rb;
          outFeatures[(o_tile*outFeaturesPerTile+2)] = rc; 
          outFeatures[(o_tile*outFeaturesPerTile+3)] = rd;
          outFeatures[(o_tile*outFeaturesPerTile+4)] = re; 
          outFeatures[(o_tile*outFeaturesPerTile+5)] = rf;
          outFeatures[(o_tile*outFeaturesPerTile+6)] = rg; 
          outFeatures[(o_tile*outFeaturesPerTile+7)] = rh; break;
          case ACT_TANH: outFeatures[(o_tile*outFeaturesPerTile+0)] = generic_tanh(ra); 
          outFeatures[(o_tile*outFeaturesPerTile+1)] = generic_tanh(rb);
          outFeatures[(o_tile*outFeaturesPerTile+2)] = generic_tanh(rc); 
          outFeatures[(o_tile*outFeaturesPerTile+3)] = generic_tanh(rd);
          outFeatures[(o_tile*outFeaturesPerTile+4)] = generic_tanh(re); 
          outFeatures[(o_tile*outFeaturesPerTile+5)] = generic_tanh(rf);
          outFeatures[(o_tile*outFeaturesPerTile+6)] = generic_tanh(rg); 
          outFeatures[(o_tile*outFeaturesPerTile+7)] = generic_tanh(rh); break;
          case ACT_SIG:  outFeatures[(o_tile*outFeaturesPerTile+0)] = generic_sig(ra); 
          outFeatures[(o_tile*outFeaturesPerTile+1)] = generic_sig(rb);
          outFeatures[(o_tile*outFeaturesPerTile+2)] = generic_sig(rc); 
          outFeatures[(o_tile*outFeaturesPerTile+3)] = generic_sig(rd);
          outFeatures[(o_tile*outFeaturesPerTile+4)] = generic_sig(re); 
          outFeatures[(o_tile*outFeaturesPerTile+5)] = generic_sig(rf);
          outFeatures[(o_tile*outFeaturesPerTile+6)] = generic_sig(rg); 
          outFeatures[(o_tile*outFeaturesPerTile+7)] = generic_sig(rh); break;
        }
#else
        outFeatures[(o_tile*outFeaturesPerTile+0)] = ra>>(q_fraqP1); 
        outFeatures[(o_tile*outFeaturesPerTile+1)] = rb>>(q_fraqP1);
        outFeatures[(o_tile*outFeaturesPerTile+2)] = rc>>(q_fraqP1); 
        outFeatures[(o_tile*outFeaturesPerTile+3)] = rd>>(q_fraqP1);
        outFeatures[(o_tile*outFeaturesPerTile+4)] = re>>(q_fraqP1); 
        outFeatures[(o_tile*outFeaturesPerTile+5)] = rf>>(q_fraqP1);
        outFeatures[(o_tile*outFeaturesPerTile+6)] = rg>>(q_fraqP1); 
        outFeatures[(o_tile*outFeaturesPerTile+7)] = rh>>(q_fraqP1);
#endif
      }
break; // case 8

case 4:
for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) {
  int32_t register_attribute ra;int32_t register_attribute rb;int32_t register_attribute rc;int32_t register_attribute rd;int32_t register_attribute re;
  int32_t register_attribute rf;int32_t register_attribute rg;int32_t register_attribute rh;int32_t register_attribute ri;int32_t register_attribute rj;
  ra = ((int32_t)bias1[o_tile*outFeaturesPerTile+0]+(int32_t)bias2[o_tile*outFeaturesPerTile+0])<<(q_fraqP1);
  rb = ((int32_t)bias1[o_tile*outFeaturesPerTile+1]+(int32_t)bias2[o_tile*outFeaturesPerTile+1])<<(q_fraqP1);
  rc = ((int32_t)bias1[o_tile*outFeaturesPerTile+2]+(int32_t)bias2[o_tile*outFeaturesPerTile+2])<<(q_fraqP1);
  rd = ((int32_t)bias1[o_tile*outFeaturesPerTile+3]+(int32_t)bias2[o_tile*outFeaturesPerTile+3])<<(q_fraqP1);





  for(int i=0; i<inFeaturesSize1P2; i++)
  {
    v2s inF_temp = ((v2s*)inFeatures1)[i];         
    SDOTP_GENERIC(ra, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+0)) + i]);
    SDOTP_GENERIC(rb, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+1)) + i]);
    SDOTP_GENERIC(rc, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+2)) + i]);
    SDOTP_GENERIC(rd, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+3)) + i]);

      } // for(int i=0; i<inFeaturesSizeP2; i++)
      
      for(int i=0; i<inFeaturesSize2P2; i++)
      {
        v2s inF_temp = ((v2s*)inFeatures2)[i];         
        SDOTP_GENERIC(ra, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+0)) + i]);
        SDOTP_GENERIC(rb, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+1)) + i]);
        SDOTP_GENERIC(rc, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+2)) + i]);
        SDOTP_GENERIC(rd, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+3)) + i]);
      }

#ifdef DOACTONTHEFLY
      ra = ra>>(q_fraqP1);
      rb = rb>>(q_fraqP1);
      rc = rc>>(q_fraqP1);
      rd = rd>>(q_fraqP1);
      switch(activationFunction) {
        case ACT_NONE: outFeatures[(o_tile*outFeaturesPerTile+0)] = ra; 
        outFeatures[(o_tile*outFeaturesPerTile+1)] = rb;
        outFeatures[(o_tile*outFeaturesPerTile+2)] = rc; 
        outFeatures[(o_tile*outFeaturesPerTile+3)] = rd; break;
        case ACT_TANH: outFeatures[(o_tile*outFeaturesPerTile+0)] = generic_tanh(ra); 
        outFeatures[(o_tile*outFeaturesPerTile+1)] = generic_tanh(rb);
        outFeatures[(o_tile*outFeaturesPerTile+2)] = generic_tanh(rc); 
        outFeatures[(o_tile*outFeaturesPerTile+3)] = generic_tanh(rd); break;
        case ACT_SIG:  outFeatures[(o_tile*outFeaturesPerTile+0)] = generic_sig(ra); 
        outFeatures[(o_tile*outFeaturesPerTile+1)] = generic_sig(rb);
        outFeatures[(o_tile*outFeaturesPerTile+2)] = generic_sig(rc); 
        outFeatures[(o_tile*outFeaturesPerTile+3)] = generic_sig(rd); break;
      }
#else
      outFeatures[(o_tile*outFeaturesPerTile+0)] = ra>>(q_fraqP1); 
      outFeatures[(o_tile*outFeaturesPerTile+1)] = rb>>(q_fraqP1);
      outFeatures[(o_tile*outFeaturesPerTile+2)] = rc>>(q_fraqP1); 
      outFeatures[(o_tile*outFeaturesPerTile+3)] = rd>>(q_fraqP1);
#endif
    }
break; // case 4


case 2:
for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) {
  int32_t register_attribute ra;int32_t register_attribute rb;
  ra = ((int32_t)bias1[o_tile*outFeaturesPerTile+0]+(int32_t)bias2[o_tile*outFeaturesPerTile+0])<<(q_fraqP1);
  rb = ((int32_t)bias1[o_tile*outFeaturesPerTile+1]+(int32_t)bias2[o_tile*outFeaturesPerTile+1])<<(q_fraqP1);   
  for(int i=0; i<inFeaturesSize1P2; i++)
  {
    v2s inF_temp = ((v2s*)inFeatures1)[i];         
    SDOTP_GENERIC(ra, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+0)) + i]);
    SDOTP_GENERIC(rb, inF_temp, ((v2s*)weight1)[(inFeaturesSize1P2*(o_tile*outFeaturesPerTile+1)) + i]);
      } // for(int i=0; i<inFeaturesSizeP2; i++)

      for(int i=0; i<inFeaturesSize2P2; i++)
      {

        v2s inF_temp = ((v2s*)inFeatures2)[i];         
        SDOTP_GENERIC(ra, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+0)) + i]);
        SDOTP_GENERIC(rb, inF_temp, ((v2s*)weight2)[(inFeaturesSize2P2*(o_tile*outFeaturesPerTile+1)) + i]);      
      }
#ifdef DOACTONTHEFLY
      ra = ra>>(q_fraqP1);
      rb = rb>>(q_fraqP1);
      switch(activationFunction) {
        case ACT_NONE: outFeatures[(o_tile*outFeaturesPerTile+0)] = ra; 
        outFeatures[(o_tile*outFeaturesPerTile+1)] = rb;  break;
        case ACT_TANH: outFeatures[(o_tile*outFeaturesPerTile+0)] = generic_tanh(ra); 
        outFeatures[(o_tile*outFeaturesPerTile+1)] = generic_tanh(rb);  break;
        case ACT_SIG:  outFeatures[(o_tile*outFeaturesPerTile+0)] = generic_sig(ra); 
        outFeatures[(o_tile*outFeaturesPerTile+1)] = generic_sig(rb); break;
      }
#else
      outFeatures[(o_tile*outFeaturesPerTile+0)] = ra>>(q_fraqP1);; 
      outFeatures[(o_tile*outFeaturesPerTile+1)] = rb>>(q_fraqP1);;
#endif
    }
break; // case 2
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
} // swtich(outFeaturesPerTile)
} // TwoLinearLayersAccumulate

 ///////////////////////////////////////////////////////////////////////////////////////////////
 //         _____  _______ ______  _______ _______ _______      _    _        _____ _  _  _   //
 // |      |     | |_____| |     \ |  |  | |_____| |             \  /  |        |   |  |  |   //
 // |_____ |_____| |     | |_____/ |  |  | |     | |_____         \/   |_____ __|__ |__|__|   //
 //                                                                                           //
 /////////////////////////////////////////////////////////////////////////////////////////////// 
#elif !defined(ASIP) && defined(SIMD) && defined(VLIWEXT) && defined(FMOUTTILING)
/** @brief Calculates two Linear Layers and accumulates them on-the-fly. (PULP and VLIW implementation)
 *  This is a helper function for efficient LSTM implementation. It calculates two linear layers in
 *  parallel and accumulates them on-the-fly.
 *  Supported Configurations:
 *  VLIW+SIMD
 *  FMOUTTILING false, true
 *  FMINTILING true
 *
 *  @param inFeaturesSize1 Input FM size for layer 1
 *  @param inFeaturesSize2 Input FM size for layer 2
 *  @param outFeaturesSize Output FM size
 *  @param activationFunction Type of activation Function (tanh, sigmoid, none)
 *  @param weight1 pointer to weight parameters of layer 1
 *  @param weight2 pointer to weight parameters of layer 2
 *  @param bias1 pointer to bias parametsr of layer 1
 *  @param bias2 pointer to bias parametsr of layer 2
 *  @param inFeatures1 pointer to input FM of layer 1
 *  @param inFeatures2 pointer to input FM of layer 2
 *  @param outFeatures pointer where to write to the output FM
 */
void NOINLINE TwoLinearLayersAccumulate (
        // Layer Attributes
  int inFeaturesSize1, int inFeaturesSize2, int outFeaturesSize, int activationFunction, 
        // Layer Parameters
  data_t * __restrict__ weight1,
  data_t * __restrict__ weight2,
  data_t * __restrict__ bias1,
  data_t * __restrict__ bias2,
        // Input and Output Features
  data_t * __restrict__ inFeatures1,
  data_t * __restrict__ inFeatures2,
  data_t * __restrict__ outFeatures)
{

  PROFILING_TWOLINEAR_START

#ifdef MULTICORE

#ifndef LSTM_HIGH_OPT
  /* instructions to parallelize the workload:
  each core computes a balanced number of neurons */
  int core_id = rt_core_id();
  int n_cores = NR_CORES;
  int chunck = 1;
  int chunkg_orig=1;

  int start_offset = 0;
  /* handle the case when number of neurons
  is less than number of cores: chunck=1 */
  if(outFeaturesSize <= n_cores)
  {
      // n_cores = outFeaturesSize;
      n_cores = 1;
      if (core_id == 0)
      {
        chunck = outFeaturesSize;
      } else
      {
        chunck = 0;
      }
  }
  else
  {
      int Log2Core = __builtin_pulp_fl1(n_cores);
      chunck = (outFeaturesSize >> Log2Core) + ((outFeaturesSize & (n_cores-1))!=0);
      chunkg_orig = chunck;
      // printf(" core_id %d a\n",core_id);
      if ((chunck % 2)!=0)
      {
        // printf(" core_id %d b\n",core_id);
        if ((core_id%2)==0)
        {
          // printf(" core_id %d +\n",core_id);
          chunkg_orig = chunck;
          chunck = chunck+1;
        }
        else
        {
          // printf(" core_id %d -\n",core_id);
          chunkg_orig = chunck;
          chunck = chunck-1;
          start_offset = 1;
        }
      }
  }
  // printf("core_id %d a\n",core_id);
  /* start and stop neuron to be computed, for each core */
  int start = MIN((chunkg_orig) * core_id+start_offset,outFeaturesSize);
  int stop = MIN(start + chunck, outFeaturesSize);
  int chunck_final = (stop-start);

#else // LSTM_HIGH_OPT

  int core_id      = rt_core_id();
  int n_cores      = NR_CORES;
  int start        = 0;
  int stop         = outFeaturesSize;
  int chunck_final = (stop-start);

#endif // LSTM_HIGH_OPT

  /* start and stop neuron to be computed, for each core */
  // int start = MIN((chunck+start_offset) * core_id,outFeaturesSize);
  // int stop = MIN(start + chunck, outFeaturesSize);
  // int chunck_final = (stop-start);
// #ifdef DEBUG_LSTM
  // printf("bla in default w outputtiling, core_id: %d, start: %d, stop: %d , chunk: %d, outFeaturesSize: %d, NR_CORES: %d \n", core_id, start, stop, chunck, outFeaturesSize, n_cores);
// #endif // DEBUG_LSTM

  int first=1;

#endif // MULTICORE


#if OUTPUTBUFFER > 8
  int tileOptions[] = {OUTPUTBUFFER,8,4,2,1};
#elif OUTPUTBUFFER > 4
  int tileOptions[] = {OUTPUTBUFFER,4,2,1};
#else
  int tileOptions[] = {OUTPUTBUFFER,2,1};
#endif



// todo add loop tiling
#ifdef MULTICORE
  data_t * bias_ptr1   = &bias1[start];
  data_t * bias_ptr2   = &bias2[start];
  data_t  * outFeatures_ptr = &outFeatures[start];
#else
  data_t * bias_ptr1   = bias1;
  data_t * bias_ptr2   = bias2;
  data_t * outFeatures_ptr = outFeatures;
#endif

  int outFeaturesPerTile = 1;

  register int32_t   temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9, temp10, temp11, temp12, temp13, temp14;
  register uint32_t  addr0, addr1, addr2, addr3, addr4, addr5, addr6, addr7, addr8, addr9, addr10, addr11, addr12, addr13, addr14;
  register uint32_t  in_addr;
  register int x0 asm("x0");

#ifdef MULTICORE
  int outFeatureTiles;
  int outFeaturesSize_remain = chunck_final;
#else
  int outFeatureTiles;
  int outFeaturesSize_remain = outFeaturesSize;
#endif

#ifdef MULTICORE
  v2s     * weight_ptr;
  v2s     * weight_ptr1 = &((v2s*)weight1)[start*(inFeaturesSize1/2)]; //(v2s*)weight1;
  v2s     * weight_ptr2 = &((v2s*)weight2)[start*(inFeaturesSize2/2)]; //(v2s*)weight2;
  // v2s     * weight_ptr = &((v2s*)weight)[start*inFeaturesSizeP2_p1];
  // data_t  * outFeatures_ptr = &outFeatures[start];
#else
  v2s     * weight_ptr;
  v2s     * weight_ptr1=(v2s*)weight1;
  v2s     * weight_ptr2=(v2s*)weight2;
#endif
  int inFeaturesSizeP2, inFeaturesSizeP4;
  data_t     * inFeatures;

#if defined(PROFILING_NEW) || defined(PROFILING)
if(core_id==0)
{
  PROFILING_LSTM_AMDAHL_SERIELL_END
  PROFILING_LSTM_AMDAHL_PARALLEL_START
}
#endif

  for(unsigned int i=0; i<sizeof(tileOptions)/sizeof(int); i++) {
    outFeaturesPerTile = tileOptions[i];
    outFeatureTiles = outFeaturesSize_remain/outFeaturesPerTile;
    if(outFeatureTiles == 0) continue;

    switch(outFeaturesPerTile) {
   #if OUTPUTBUFFER > 2
     case OUTPUTBUFFER:

     for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
     {
        // printf("o_tile=%i\n", o_tile);
        #if OUTPUTBUFFER > 2
      temp0 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+0]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+0])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 3
      temp1 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+1]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+1])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 4
      temp2 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+2]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+2])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 5
      temp3 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+3]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+3])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 6
      temp4 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+4]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+4])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 7
      temp5 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+5]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+5])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 8
      temp6 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+6]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+6])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 9
      temp7 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+7]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+7])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 10
      temp8 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+8]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+8])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 11
      temp9 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+9]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+9])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 12
      temp10 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+10]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+10])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 13
      temp11 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+11]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+11])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 14
      temp12 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+12]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+12])<<(q_fraqP1);
        #endif
      temp13 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+OUTPUTBUFFER-2]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+OUTPUTBUFFER-2])<<(q_fraqP1);
      temp14 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+OUTPUTBUFFER-1]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+OUTPUTBUFFER-1])<<(q_fraqP1);



      for(int turn=0; turn<2; turn++) {
        if(turn==0) {
          inFeaturesSizeP2 = inFeaturesSize1/2;
          inFeatures = inFeatures1;
        #ifdef FMINTILING
        inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
        #else
        inFeaturesSizeP4 = inFeaturesSizeP2; // input FM tiling
        #endif
#ifdef MULTICORE
        // if (first==1)
        // {
        //   weight_ptr = &((v2s*)weight_ptr1)[start*inFeaturesSizeP2];
        // }
        // else
        // {
          weight_ptr = weight_ptr1;
        // }
#else
        weight_ptr = weight_ptr1;
#endif
      } else {
        inFeatures = inFeatures2;
        inFeaturesSizeP2 = inFeaturesSize2/2;
        #ifdef FMINTILING
        inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
        #else
        inFeaturesSizeP4 = inFeaturesSizeP2; // input FM tiling
        #endif
#ifdef MULTICORE
        // if (first==1)
        // {
        //   weight_ptr = &((v2s*)weight_ptr2)[start*inFeaturesSizeP2];
        //   first = 0;
        // }
        // else
        // {
          weight_ptr = weight_ptr2;
        // }
#else
        weight_ptr = weight_ptr2;
#endif
      }



      addr0  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 0))];
      addr1  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 1))];
      addr2  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 2))];
      addr3  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 3))];
      addr4  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 4))];
      addr5  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 5))];
      addr6  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 6))];
      addr7  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 7))];
      addr8  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 8))];
      addr9  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 9))];
      addr10 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+10))];
      addr11 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+11))];
      addr12 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+12))];
      addr13 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+OUTPUTBUFFER-2))];
      addr14 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+OUTPUTBUFFER-1))];
        asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload first weight
        asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload first weight
        in_addr = (uint32_t)(((v2s*)inFeatures));
        for(int i=0; i<inFeaturesSizeP4; i++) {
          v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
          v2s inF_temp2;// = ((v2s*)inFeatures)[2*i+1];


            // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
            asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
#ifdef FMINTILING
            asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
#endif
            // MANUAL loop unfolding
            // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
            // __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+o_rel)) + i], temp[o_rel]);
             #if OUTPUTBUFFER > 2
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 3
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 4
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr4) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 5
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr5) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 6
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp4),  "+r" (addr6) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 7
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp5),  "+r" (addr7) : "r" (inF_temp) ); 
             #endif
             #if OUTPUTBUFFER > 9 // TODO: should probably be 8
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr8) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 9
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr9) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 10
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp8),  "+r" (addr10) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 11
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp9),  "+r" (addr11) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 12
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp10),  "+r" (addr12) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 13
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp11),  "+r" (addr13) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 14
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp12),  "+r" (addr14) : "r" (inF_temp) );
             #endif
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp13),  "+r" (addr0) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp14),  "+r" (addr1) : "r" (inF_temp) );

#ifdef FMINTILING
             #if OUTPUTBUFFER > 2
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp2) );
             #endif
             #if OUTPUTBUFFER > 3
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp2) );
             #endif
             #if OUTPUTBUFFER > 4
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr4) : "r" (inF_temp2) );
             #endif
             #if OUTPUTBUFFER > 5
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr5) : "r" (inF_temp2) );
             #endif
             #if OUTPUTBUFFER > 6
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp4),  "+r" (addr6) : "r" (inF_temp2) );
             #endif
             #if OUTPUTBUFFER > 7
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp5),  "+r" (addr7) : "r" (inF_temp2) ); 
             #endif
             #if OUTPUTBUFFER > 9
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr8) : "r" (inF_temp2) );
             #endif
             #if OUTPUTBUFFER > 9
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr9) : "r" (inF_temp2) );
             #endif
             #if OUTPUTBUFFER > 10
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp8),  "+r" (addr10) : "r" (inF_temp2) );
             #endif
             #if OUTPUTBUFFER > 11
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp9),  "+r" (addr11) : "r" (inF_temp2) );
             #endif
             #if OUTPUTBUFFER > 12
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp10),  "+r" (addr12) : "r" (inF_temp2) );
             #endif
             #if OUTPUTBUFFER > 13
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp11),  "+r" (addr13) : "r" (inF_temp2) );
             #endif
             #if OUTPUTBUFFER > 14
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp12),  "+r" (addr14) : "r" (inF_temp2) );
             #endif
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp13),  "+r" (addr0) : "r" (inF_temp2) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp14),  "+r" (addr1) : "r" (inF_temp2) );
#endif
          // }
          }
#ifdef FMINTILING
        if(inFeaturesSizeP2%2==1) { // add contribution of left over input channel (input channels not multiple of 4)
                    v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
            // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
            asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
            // asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
            
            // MANUAL loop unfolding
            // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
            // __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+o_rel)) + i], temp[o_rel]);
             #if OUTPUTBUFFER > 2
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 3
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 4
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr4) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 5
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr5) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 6
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp4),  "+r" (addr6) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 7
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp5),  "+r" (addr7) : "r" (inF_temp) ); 
             #endif
             #if OUTPUTBUFFER > 9
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr8) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 9
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr9) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 10
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp8),  "+r" (addr10) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 11
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp9),  "+r" (addr11) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 12
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp10),  "+r" (addr12) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 13
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp11),  "+r" (addr13) : "r" (inF_temp) );
             #endif
             #if OUTPUTBUFFER > 14
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp12),  "+r" (addr14) : "r" (inF_temp) );
             #endif
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp13),  "+r" (addr0) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp14),  "+r" (addr1) : "r" (inF_temp) );

          }

#endif
} // loop for fm1 and fm2
      // for(int o_rel =0;o_rel<outFeaturesPerTile;o_rel++) {
              #if OUTPUTBUFFER > 2
outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = shiftAndAct(temp0, activationFunction);
              #endif
              #if OUTPUTBUFFER > 3
outFeatures_ptr[(o_tile*outFeaturesPerTile+1)] = shiftAndAct(temp1, activationFunction);
              #endif
              #if OUTPUTBUFFER > 4
outFeatures_ptr[(o_tile*outFeaturesPerTile+2)] = shiftAndAct(temp2, activationFunction);
              #endif
              #if OUTPUTBUFFER > 5
outFeatures_ptr[(o_tile*outFeaturesPerTile+3)] = shiftAndAct(temp3, activationFunction);
              #endif
              #if OUTPUTBUFFER > 6
outFeatures_ptr[(o_tile*outFeaturesPerTile+4)] = shiftAndAct(temp4, activationFunction);
              #endif
              #if OUTPUTBUFFER > 7
outFeatures_ptr[(o_tile*outFeaturesPerTile+5)] = shiftAndAct(temp5, activationFunction);
              #endif
              #if OUTPUTBUFFER > 8
outFeatures_ptr[(o_tile*outFeaturesPerTile+6)] = shiftAndAct(temp6, activationFunction);
              #endif
              #if OUTPUTBUFFER > 9
outFeatures_ptr[(o_tile*outFeaturesPerTile+7)] = shiftAndAct(temp7, activationFunction);
              #endif
              #if OUTPUTBUFFER > 10
outFeatures_ptr[(o_tile*outFeaturesPerTile+8)] = shiftAndAct(temp8, activationFunction);
              #endif
              #if OUTPUTBUFFER > 11
outFeatures_ptr[(o_tile*outFeaturesPerTile+9)] = shiftAndAct(temp9, activationFunction);
              #endif
              #if OUTPUTBUFFER > 12
outFeatures_ptr[(o_tile*outFeaturesPerTile+10)] = shiftAndAct(temp10, activationFunction);
              #endif
              #if OUTPUTBUFFER > 13
outFeatures_ptr[(o_tile*outFeaturesPerTile+11)] = shiftAndAct(temp11, activationFunction);
              #endif
              #if OUTPUTBUFFER > 14
outFeatures_ptr[(o_tile*outFeaturesPerTile+12)] = shiftAndAct(temp12, activationFunction);
              #endif
outFeatures_ptr[(o_tile*outFeaturesPerTile+OUTPUTBUFFER-2)] = shiftAndAct(temp13, activationFunction);
outFeatures_ptr[(o_tile*outFeaturesPerTile+OUTPUTBUFFER-1)] = shiftAndAct(temp14, activationFunction);
              // printf("outFeatures[%i]=%i\n", o_rel, outFeatures[(o_tile*outFeaturesPerTile+o_rel)]);
      // }

}
break;
   #endif
   #if OUTPUTBUFFER > 8
case 8:

for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
{

  temp0 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+0]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+0])<<(q_fraqP1);
  temp1 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+1]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+1])<<(q_fraqP1);
  temp2 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+2]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+2])<<(q_fraqP1);
  temp3 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+3]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+3])<<(q_fraqP1);
  temp4 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+4]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+4])<<(q_fraqP1);
  temp5 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+5]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+5])<<(q_fraqP1);
  temp6 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+6]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+6])<<(q_fraqP1);
  temp7 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+7]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+7])<<(q_fraqP1);

        // }
        // printf("wght=%i, addr before=%i (+%i)=%i\n", weight, addr0, inFeaturesSizeP2*outFeaturesPerTile, addr0+4*inFeaturesSizeP2*outFeaturesPerTile);
  for(int turn=0; turn<2; turn++) {
    if(turn==0) {
      weight_ptr = weight_ptr1;
      inFeaturesSizeP2 = inFeaturesSize1/2;
      inFeatures = inFeatures1;
        #ifdef FMINTILING
        inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
        #else
        inFeaturesSizeP4 = inFeaturesSizeP2; // input FM tiling
        #endif
      } else {
        weight_ptr = weight_ptr2;
        inFeatures = inFeatures2;
        inFeaturesSizeP2 = inFeaturesSize2/2;
        #ifdef FMINTILING
        inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
        #else
        inFeaturesSizeP4 = inFeaturesSizeP2; // input FM tiling
        #endif
      }
      addr0 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+0))];
      addr1 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+1))];
      addr2 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+2))];
      addr3 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+3))];
      addr4 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+4))];
      addr5 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+5))];
      addr6 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+6))];
      addr7 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+7))];


        asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload first weight
        asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload first weight

        in_addr = (((v2s*)inFeatures));
        for(int i=0; i<inFeaturesSizeP4; i++) {
          v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
          v2s inF_temp2;// = ((v2s*)inFeatures)[2*i+1];


            // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
            asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
#ifdef FMINTILING
            asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
#endif
            // MANUAL loop unfolding
            // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
            // __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+o_rel)) + i], temp[o_rel]);
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr4) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr5) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp4),  "+r" (addr6) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp5),  "+r" (addr7) : "r" (inF_temp) ); 
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr0) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr1) : "r" (inF_temp) );
#ifdef FMINTILING
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp2) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp2) );
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr4) : "r" (inF_temp2) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr5) : "r" (inF_temp2) );
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp4),  "+r" (addr6) : "r" (inF_temp2) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp5),  "+r" (addr7) : "r" (inF_temp2) ); 
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr0) : "r" (inF_temp2) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr1) : "r" (inF_temp2) );
#endif
          // }
          }
#ifdef FMINTILING
        if(inFeaturesSizeP2%2==1) { // add contribution of left over input channel (input channels not multiple of 4)
            v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
            asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr4) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr5) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp4),  "+r" (addr6) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp5),  "+r" (addr7) : "r" (inF_temp) ); 
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp6),  "+r" (addr0) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp7),  "+r" (addr1) : "r" (inF_temp) );

          }
#endif
      } // fm in1 and in2
      // for(int o_rel =0;o_rel<outFeaturesPerTile;o_rel++) {
      outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = shiftAndAct(temp0, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+1)] = shiftAndAct(temp1, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+2)] = shiftAndAct(temp2, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+3)] = shiftAndAct(temp3, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+4)] = shiftAndAct(temp4, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+5)] = shiftAndAct(temp5, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+6)] = shiftAndAct(temp6, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+7)] = shiftAndAct(temp7, activationFunction);
      // }
    }
    break;
   #endif
   #if OUTPUTBUFFER > 4
    case 4:

    for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
    {

      temp0 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+0]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+0])<<(q_fraqP1);
      temp1 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+1]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+1])<<(q_fraqP1);
      temp2 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+2]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+2])<<(q_fraqP1);
      temp3 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+3]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+3])<<(q_fraqP1);
      for(int turn=0; turn<2; turn++) {
        if(turn==0) {
          weight_ptr = weight_ptr1;
          inFeaturesSizeP2 = inFeaturesSize1/2;
          inFeatures = inFeatures1;
          #ifdef FMINTILING
          inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
          #else
          inFeaturesSizeP4 = inFeaturesSizeP2; // input FM tiling
          #endif
        } else {
          weight_ptr = weight_ptr2;
          inFeatures = inFeatures2;
          inFeaturesSizeP2 = inFeaturesSize2/2;
          #ifdef FMINTILING
          inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
          #else
          inFeaturesSizeP4 = inFeaturesSizeP2; // input FM tiling
          #endif
        }
        // }
        addr0 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+0))];
        addr1 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+1))];
        addr2 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+2))];
        addr3 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+3))];


       asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload first weight
       asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload first weight

       in_addr = (uint32_t)(((v2s*)inFeatures));
       for(int i=0; i<inFeaturesSizeP4; i++) {
            v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
            v2s inF_temp2;// = ((v2s*)inFeatures)[2*i+1];


            // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
            asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
#ifdef FMINTILING
            asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
#endif
            // MANUAL loop unfolding
            // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
            // __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+o_rel)) + i], temp[o_rel]);
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr0) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr1) : "r" (inF_temp) );
#ifdef FMINTILING
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp2) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp2) );
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr0) : "r" (inF_temp2) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr1) : "r" (inF_temp2) );
#endif
          // }
          }
# ifdef FMINTILING
        if(inFeaturesSizeP2%2==1) { // add contribution of left over input channel (input channels not multiple of 4)
            v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
            asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr2) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr3) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp2),  "+r" (addr0) : "r" (inF_temp) );
            asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp3),  "+r" (addr1) : "r" (inF_temp) );

          }
# endif
      } // fm in1 and in2
      // for(int o_rel =0;o_rel<outFeaturesPerTile;o_rel++) {
      outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = shiftAndAct(temp0, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+1)] = shiftAndAct(temp1, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+2)] = shiftAndAct(temp2, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+3)] = shiftAndAct(temp3, activationFunction);
              // printf("outFeatures[%i]=%i\n", o_rel, outFeatures[(o_tile*outFeaturesPerTile+o_rel)]);
      // }
    }
    break;
#endif
    case 2:

    for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
    {

      temp0 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+0]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+0])<<(q_fraqP1);
      temp1 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+1]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+1])<<(q_fraqP1);
        // temp2 = (int32_t)bias[o_tile*outFeaturesPerTile+2]<<(q_fraqP1);
        // temp3 = (int32_t)bias[o_tile*outFeaturesPerTile+3]<<(q_fraqP1);
        // }

      for(int turn=0; turn<2; turn++) {
        if(turn==0) {
          weight_ptr = &weight_ptr1[0];
          inFeaturesSizeP2 = inFeaturesSize1/2;
          inFeatures = inFeatures1;
          #ifdef FMINTILING
          inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
          #else
          inFeaturesSizeP4 = inFeaturesSizeP2; // input FM tiling
          #endif
        } else {
          weight_ptr = &weight_ptr2[0];
          inFeatures = inFeatures2;
          inFeaturesSizeP2 = inFeaturesSize2/2;
          #ifdef FMINTILING
          inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
          #else
          inFeaturesSizeP4 = inFeaturesSizeP2; // input FM tiling
          #endif
        }
        addr0 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+0))];
        addr1 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+1))];
        // addr2 = &((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+2))];
        // addr3 = &((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+3))];
        asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload no compute
        asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload no compute
        for(int i=0; i<inFeaturesSizeP2; i++) {
          v2s inF_temp = ((v2s*)inFeatures)[i];
          
          // int o_rel;
          // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
          asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (temp0),  "+r" (addr0) : "r" (inF_temp) );
          asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (temp1),  "+r" (addr1) : "r" (inF_temp) );
             // asm volatile("pl.sdotsp.h %0, %1, %2" : "+r" (temp2) : "r" (addr3), "r" (inF_temp) );
             // asm volatile("pl.sdotsp.h %0, %1, %2" : "+r" (temp3) : "r" (addr0), "r" (inF_temp) );
          // }
        }
      } // fm in1 in2 
      // for(int o_rel =0;o_rel<outFeaturesPerTile;o_rel++) {
      outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = shiftAndAct(temp0, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+1)] = shiftAndAct(temp1, activationFunction);
              // outFeatures[(o_tile*outFeaturesPerTile+2)] = temp2>>(q_fraqP1);
              // outFeatures[(o_tile*outFeaturesPerTile+3)] = temp3>>(q_fraqP1);
              // printf("outFeatures[%i]=%i\n", o_rel, outFeatures[(o_tile*outFeaturesPerTile+o_rel)]);
      // }

    }
    break;
    case 1:


    for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
    {
     for(int turn=0; turn<2; turn++) {
      if(turn==0) {
        weight_ptr = &weight_ptr1[0];
        inFeaturesSizeP2 = inFeaturesSize1/2;
        inFeatures = inFeatures1;
        #ifdef FMINTILING
        inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
        #else
        inFeaturesSizeP4 = inFeaturesSizeP2; // input FM tiling
        #endif
      } else {
        weight_ptr = &weight_ptr2[0];
        inFeatures = inFeatures2;
        inFeaturesSizeP2 = inFeaturesSize2/2;
        #ifdef FMINTILING
        inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
        #else
        inFeaturesSizeP4 = inFeaturesSizeP2; // input FM tiling
        #endif
      }
      temp0 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+0]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+0])<<(q_fraqP1);
      for(int i=0; i<inFeaturesSizeP2; i++) {
        v2s inF_temp = ((v2s*)inFeatures)[i];
        temp0 = __SUMDOTP2(inF_temp, ((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile)) + i], temp0);
      }
   } // fm in1 and in2 
   outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = shiftAndAct(temp0, activationFunction);
 }
 break;
}

 // updat pointer for next iteration
bias_ptr1                = &bias_ptr1[outFeatureTiles*outFeaturesPerTile];
bias_ptr2                = &bias_ptr2[outFeatureTiles*outFeaturesPerTile];
weight_ptr1              = &weight_ptr1[(inFeaturesSize1/2*(outFeatureTiles*outFeaturesPerTile))];
weight_ptr2              = &weight_ptr2[(inFeaturesSize2/2*(outFeatureTiles*outFeaturesPerTile))];
outFeatures_ptr         = &outFeatures_ptr[(outFeatureTiles*outFeaturesPerTile)];
outFeaturesSize_remain -= outFeatureTiles*outFeaturesPerTile;

if (outFeaturesSize_remain==0)  break;

}

#if defined(PROFILING_NEW) || defined(PROFILING)

#if defined(PROFILING_NEW) || defined(PROFILING)
PROFILING_TWOLINEAR_END
if(core_id==0)
{
  PROFILING_LSTM_AMDAHL_PARALLEL_END
  PROFILING_LSTM_AMDAHL_SERIELL_START
}
#endif

#endif
}
#elif !defined(ASIP) && defined(SIMD) && !defined(VLIWEXT) && defined(FMOUTTILING)
/** @brief Calculates two Linear Layers and accumulates them on-the-fly. (PULP and VLIW implementation)
 *  This is a helper function for efficient LSTM implementation. It calculates two linear layers in
 *  parallel and accumulates them on-the-fly.
 *  Supported Configurations:
 *  VLIW+SIMD
 *  FMOUTTILING false, true
 *  FMINTILING true
 *
 *  @param inFeaturesSize1 Input FM size for layer 1
 *  @param inFeaturesSize2 Input FM size for layer 2
 *  @param outFeaturesSize Output FM size
 *  @param activationFunction Type of activation Function (tanh, sigmoid, none)
 *  @param weight1 pointer to weight parameters of layer 1
 *  @param weight2 pointer to weight parameters of layer 2
 *  @param bias1 pointer to bias parametsr of layer 1
 *  @param bias2 pointer to bias parametsr of layer 2
 *  @param inFeatures1 pointer to input FM of layer 1
 *  @param inFeatures2 pointer to input FM of layer 2
 *  @param outFeatures pointer where to write to the output FM
 */
void NOINLINE TwoLinearLayersAccumulate (
        // Layer Attributes
  int inFeaturesSize1, int inFeaturesSize2, int outFeaturesSize, int activationFunction, 
        // Layer Parameters
  data_t * __restrict__ weight1,
  data_t * __restrict__ weight2,
  data_t * __restrict__ bias1,
  data_t * __restrict__ bias2,
        // Input and Output Features
  data_t * __restrict__ inFeatures1,
  data_t * __restrict__ inFeatures2,
  data_t * __restrict__ outFeatures)
{

  PROFILING_TWOLINEAR_START

#if OUTPUTBUFFER > 8
  int tileOptions[] = {OUTPUTBUFFER,8,4,2,1};
#elif OUTPUTBUFFER > 4
  int tileOptions[] = {OUTPUTBUFFER,4,2,1};
#else
  int tileOptions[] = {OUTPUTBUFFER,2,1};
#endif



// todo add loop tiling
  data_t * bias_ptr1   = bias1;

  data_t * bias_ptr2   = bias2;

  data_t  * outFeatures_ptr = outFeatures;
  int outFeaturesPerTile = 1;

  register_attribute int32_t   temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9, temp10, temp11, temp12, temp13, temp14;
  register_attribute uint32_t  addr0, addr1, addr2, addr3, addr4, addr5, addr6, addr7, addr8, addr9, addr10, addr11, addr12, addr13, addr14;
  register_attribute uint32_t  in_addr;

  int outFeatureTiles;
  int outFeaturesSize_remain = outFeaturesSize;
  v2s     * weight_ptr; 
  v2s     * weight_ptr1=(v2s*)weight1;
  v2s     * weight_ptr2=(v2s*)weight2;
  int inFeaturesSizeP2, inFeaturesSizeP4;
  data_t     * inFeatures;

  for(unsigned int i=0; i<sizeof(tileOptions)/sizeof(int); i++) {
    outFeaturesPerTile = tileOptions[i];
    outFeatureTiles = outFeaturesSize_remain/outFeaturesPerTile;
    if(outFeatureTiles == 0) continue;

    switch(outFeaturesPerTile) {
   #if OUTPUTBUFFER > 2
     case OUTPUTBUFFER:

     for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
     {
        // printf("o_tile=%i\n", o_tile);
        #if OUTPUTBUFFER > 2
      temp0 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+0]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+0])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 3
      temp1 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+1]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+1])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 4
      temp2 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+2]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+2])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 5
      temp3 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+3]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+3])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 6
      temp4 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+4]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+4])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 7
      temp5 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+5]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+5])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 8
      temp6 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+6]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+6])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 9
      temp7 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+7]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+7])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 10
      temp8 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+8]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+8])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 11
      temp9 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+9]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+9])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 12
      temp10 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+10]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+10])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 13
      temp11 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+11]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+11])<<(q_fraqP1);
        #endif
        #if OUTPUTBUFFER > 14
      temp12 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+12]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+12])<<(q_fraqP1);
        #endif
      temp13 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+OUTPUTBUFFER-2]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+OUTPUTBUFFER-2])<<(q_fraqP1);
      temp14 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+OUTPUTBUFFER-1]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+OUTPUTBUFFER-1])<<(q_fraqP1);



      for(int turn=0; turn<2; turn++) {
        if(turn==0) {
          weight_ptr = weight_ptr1;
          inFeaturesSizeP2 = inFeaturesSize1/2;
          inFeatures = inFeatures1;
        // #ifdef FMINTILING
        // inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
        // #else
        inFeaturesSizeP4 = inFeaturesSizeP2; // input FM tiling
        // #endif
      } else {
        weight_ptr = weight_ptr2;
        inFeatures = inFeatures2;
        inFeaturesSizeP2 = inFeaturesSize2/2;
        // NO FMINTILING as this does not give any benefit (to implement uncomment all #ifdef FMINTLING parts and change indexing to 2i and 2i+1)
        // #ifdef FMINTILING
        // inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
        // #else
        inFeaturesSizeP4 = inFeaturesSizeP2; // input FM tiling
        // #endif
      }



      addr0  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 0))];
      addr1  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 1))];
      addr2  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 2))];
      addr3  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 3))];
      addr4  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 4))];
      addr5  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 5))];
      addr6  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 6))];
      addr7  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 7))];
      addr8  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 8))];
      addr9  = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+ 9))];
      addr10 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+10))];
      addr11 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+11))];
      addr12 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+12))];
      addr13 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+OUTPUTBUFFER-2))];
      addr14 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+OUTPUTBUFFER-1))];
        // asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload first weight
        // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload first weight
        in_addr = (uint32_t)(((v2s*)inFeatures));
        for(int i=0; i<inFeaturesSizeP4; i++) {
          v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
          v2s inF_temp2;// = ((v2s*)inFeatures)[2*i+1];


            // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
            asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
// #ifdef FMINTILING
//             asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
// #endif
            // MANUAL loop unfolding
            // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
            // __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+o_rel)) + i], temp[o_rel]);
             #if OUTPUTBUFFER > 2
            SDOTP_GENERIC(temp0, ((v2s*)addr0)[i], inF_temp);
             #endif
             #if OUTPUTBUFFER > 3
            SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp);
             #endif
             #if OUTPUTBUFFER > 4
            SDOTP_GENERIC(temp2, ((v2s*)addr2)[i], inF_temp);
             #endif
             #if OUTPUTBUFFER > 5
            SDOTP_GENERIC(temp3, ((v2s*)addr3)[i], inF_temp);
             #endif
             #if OUTPUTBUFFER > 6
            SDOTP_GENERIC(temp4, ((v2s*)addr4)[i], inF_temp);
             #endif
             #if OUTPUTBUFFER > 7
            SDOTP_GENERIC(temp5, ((v2s*)addr5)[i], inF_temp) ;
             #endif
             #if OUTPUTBUFFER > 9
            SDOTP_GENERIC(temp6, ((v2s*)addr6)[i], inF_temp);
             #endif
             #if OUTPUTBUFFER > 9
            SDOTP_GENERIC(temp7, ((v2s*)addr7)[i], inF_temp);
             #endif
             #if OUTPUTBUFFER > 10
            SDOTP_GENERIC(temp8, ((v2s*)addr8)[i], inF_temp);
             #endif
             #if OUTPUTBUFFER > 11
            SDOTP_GENERIC(temp9, ((v2s*)addr9)[i], inF_temp);
             #endif
             #if OUTPUTBUFFER > 12
            SDOTP_GENERIC(temp10, ((v2s*)addr10)[i], inF_temp);
             #endif
             #if OUTPUTBUFFER > 13
            SDOTP_GENERIC(temp11, ((v2s*)addr11)[i], inF_temp);
             #endif
             #if OUTPUTBUFFER > 14
            SDOTP_GENERIC(temp12, ((v2s*)addr12)[i], inF_temp);
             #endif
            SDOTP_GENERIC(temp13, ((v2s*)addr13)[i], inF_temp);
            SDOTP_GENERIC(temp14, ((v2s*)addr14)[i], inF_temp);

// #ifdef FMINTILING
//              #if OUTPUTBUFFER > 2
//             SDOTP_GENERIC(temp0, ((v2s*)addr0)[i], inF_temp2);
//              #endif
//              #if OUTPUTBUFFER > 3
//             SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp2);
//              #endif
//              #if OUTPUTBUFFER > 4
//             SDOTP_GENERIC(temp2, ((v2s*)addr2)[i], inF_temp2);
//              #endif
//              #if OUTPUTBUFFER > 5
//             SDOTP_GENERIC(temp3, ((v2s*)addr3)[i], inF_temp2);
//              #endif
//              #if OUTPUTBUFFER > 6
//             SDOTP_GENERIC(temp4, ((v2s*)addr4)[i], inF_temp2);
//              #endif
//              #if OUTPUTBUFFER > 7
//             SDOTP_GENERIC(temp5, ((v2s*)addr5)[i], inF_temp2) ;
//              #endif
//              #if OUTPUTBUFFER > 9
//             SDOTP_GENERIC(temp6, ((v2s*)addr6)[i], inF_temp2);
//              #endif
//              #if OUTPUTBUFFER > 9
//             SDOTP_GENERIC(temp7, ((v2s*)addr7)[i], inF_temp2);
//              #endif
//              #if OUTPUTBUFFER > 10
//             SDOTP_GENERIC(temp8, ((v2s*)addr8)[i], inF_temp2);
//              #endif
//              #if OUTPUTBUFFER > 11
//             SDOTP_GENERIC(temp9, ((v2s*)addr9)[i], inF_temp2);
//              #endif
//              #if OUTPUTBUFFER > 12
//             SDOTP_GENERIC(temp10, ((v2s*)addr10)[i], inF_temp2);
//              #endif
//              #if OUTPUTBUFFER > 13
//             SDOTP_GENERIC(temp11, ((v2s*)addr11)[i], inF_temp2);
//              #endif
//              #if OUTPUTBUFFER > 14
//             SDOTP_GENERIC(temp12, ((v2s*)addr12)[i], inF_temp2);
//              #endif
//             SDOTP_GENERIC(temp13, ((v2s*)addr13)[i], inF_temp2);
//             SDOTP_GENERIC(temp14, ((v2s*)addr14)[i], inF_temp2);
// #endif
          // }
          }
// #ifdef FMINTILING
//         if(inFeaturesSizeP2%2==1) { // add contribution of left over input channel (input channels not multiple of 4)
//                     v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
//             // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
//             asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
//             // asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
            
//             // MANUAL loop unfolding
//             // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
//             // __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+o_rel)) + i], temp[o_rel]);
//                           #if OUTPUTBUFFER > 2
//             SDOTP_GENERIC(temp0, ((v2s*)addr0)[i], inF_temp);
//              #endif
//              #if OUTPUTBUFFER > 3
//             SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp);
//              #endif
//              #if OUTPUTBUFFER > 4
//             SDOTP_GENERIC(temp2, ((v2s*)addr2)[i], inF_temp);
//              #endif
//              #if OUTPUTBUFFER > 5
//             SDOTP_GENERIC(temp3, ((v2s*)addr3)[i], inF_temp);
//              #endif
//              #if OUTPUTBUFFER > 6
//             SDOTP_GENERIC(temp4, ((v2s*)addr4)[i], inF_temp);
//              #endif
//              #if OUTPUTBUFFER > 7
//             SDOTP_GENERIC(temp5, ((v2s*)addr5)[i], inF_temp) ;
//              #endif
//              #if OUTPUTBUFFER > 9
//             SDOTP_GENERIC(temp6, ((v2s*)addr6)[i], inF_temp);
//              #endif
//              #if OUTPUTBUFFER > 9
//             SDOTP_GENERIC(temp7, ((v2s*)addr7)[i], inF_temp);
//              #endif
//              #if OUTPUTBUFFER > 10
//             SDOTP_GENERIC(temp8, ((v2s*)addr8)[i], inF_temp);
//              #endif
//              #if OUTPUTBUFFER > 11
//             SDOTP_GENERIC(temp9, ((v2s*)addr9)[i], inF_temp);
//              #endif
//              #if OUTPUTBUFFER > 12
//             SDOTP_GENERIC(temp10, ((v2s*)addr10)[i], inF_temp);
//              #endif
//              #if OUTPUTBUFFER > 13
//             SDOTP_GENERIC(temp11, ((v2s*)addr11)[i], inF_temp);
//              #endif
//              #if OUTPUTBUFFER > 14
//             SDOTP_GENERIC(temp12, ((v2s*)addr12)[i], inF_temp);
//              #endif
//             SDOTP_GENERIC(temp13, ((v2s*)addr13)[i], inF_temp);
//             SDOTP_GENERIC(temp14, ((v2s*)addr14)[i], inF_temp);

//           }

// #endif
} // loop for fm1 and fm2
      // for(int o_rel =0;o_rel<outFeaturesPerTile;o_rel++) {
              #if OUTPUTBUFFER > 2
outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = shiftAndAct(temp0, activationFunction);
              #endif
              #if OUTPUTBUFFER > 3
outFeatures_ptr[(o_tile*outFeaturesPerTile+1)] = shiftAndAct(temp1, activationFunction);
              #endif
              #if OUTPUTBUFFER > 4
outFeatures_ptr[(o_tile*outFeaturesPerTile+2)] = shiftAndAct(temp2, activationFunction);
              #endif
              #if OUTPUTBUFFER > 5
outFeatures_ptr[(o_tile*outFeaturesPerTile+3)] = shiftAndAct(temp3, activationFunction);
              #endif
              #if OUTPUTBUFFER > 6
outFeatures_ptr[(o_tile*outFeaturesPerTile+4)] = shiftAndAct(temp4, activationFunction);
              #endif
              #if OUTPUTBUFFER > 7
outFeatures_ptr[(o_tile*outFeaturesPerTile+5)] = shiftAndAct(temp5, activationFunction);
              #endif
              #if OUTPUTBUFFER > 8
outFeatures_ptr[(o_tile*outFeaturesPerTile+6)] = shiftAndAct(temp6, activationFunction);
              #endif
              #if OUTPUTBUFFER > 9
outFeatures_ptr[(o_tile*outFeaturesPerTile+7)] = shiftAndAct(temp7, activationFunction);
              #endif
              #if OUTPUTBUFFER > 10
outFeatures_ptr[(o_tile*outFeaturesPerTile+8)] = shiftAndAct(temp8, activationFunction);
              #endif
              #if OUTPUTBUFFER > 11
outFeatures_ptr[(o_tile*outFeaturesPerTile+9)] = shiftAndAct(temp9, activationFunction);
              #endif
              #if OUTPUTBUFFER > 12
outFeatures_ptr[(o_tile*outFeaturesPerTile+10)] = shiftAndAct(temp10, activationFunction);
              #endif
              #if OUTPUTBUFFER > 13
outFeatures_ptr[(o_tile*outFeaturesPerTile+11)] = shiftAndAct(temp11, activationFunction);
              #endif
              #if OUTPUTBUFFER > 14
outFeatures_ptr[(o_tile*outFeaturesPerTile+12)] = shiftAndAct(temp12, activationFunction);
              #endif
outFeatures_ptr[(o_tile*outFeaturesPerTile+OUTPUTBUFFER-2)] = shiftAndAct(temp13, activationFunction);
outFeatures_ptr[(o_tile*outFeaturesPerTile+OUTPUTBUFFER-1)] = shiftAndAct(temp14, activationFunction);
              // printf("outFeatures[%i]=%i\n", o_rel, outFeatures[(o_tile*outFeaturesPerTile+o_rel)]);
      // }

}
break;
   #endif
   #if OUTPUTBUFFER > 8
case 8:

for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
{

  temp0 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+0]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+0])<<(q_fraqP1);
  temp1 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+1]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+1])<<(q_fraqP1);
  temp2 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+2]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+2])<<(q_fraqP1);
  temp3 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+3]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+3])<<(q_fraqP1);
  temp4 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+4]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+4])<<(q_fraqP1);
  temp5 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+5]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+5])<<(q_fraqP1);
  temp6 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+6]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+6])<<(q_fraqP1);
  temp7 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+7]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+7])<<(q_fraqP1);

        // }
        // printf("wght=%i, addr before=%i (+%i)=%i\n", weight, addr0, inFeaturesSizeP2*outFeaturesPerTile, addr0+4*inFeaturesSizeP2*outFeaturesPerTile);
  for(int turn=0; turn<2; turn++) {
    if(turn==0) {
      weight_ptr = weight_ptr1;
      inFeaturesSizeP2 = inFeaturesSize1/2;
      inFeatures = inFeatures1;
        // #ifdef FMINTILING
        // inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
        // #else
        inFeaturesSizeP4 = inFeaturesSizeP2; // input FM tiling
        // #endif
      } else {
        weight_ptr = weight_ptr2;
        inFeatures = inFeatures2;
        inFeaturesSizeP2 = inFeaturesSize2/2;
        // #ifdef FMINTILING
        // inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
        // #else
        inFeaturesSizeP4 = inFeaturesSizeP2; // input FM tiling
        // #endif
      }
      addr0 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+0))];
      addr1 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+1))];
      addr2 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+2))];
      addr3 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+3))];
      addr4 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+4))];
      addr5 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+5))];
      addr6 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+6))];
      addr7 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+7))];


        // asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload first weight
        // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload first weight

        in_addr = (((v2s*)inFeatures));
        for(int i=0; i<inFeaturesSizeP4; i++) {
          v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
          v2s inF_temp2;// = ((v2s*)inFeatures)[2*i+1];


            // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
            asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
// #ifdef FMINTILING
//             asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
// #endif
            // MANUAL loop unfolding
            // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
            // __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+o_rel)) + i], temp[o_rel]);
            SDOTP_GENERIC(temp0, ((v2s*)addr0)[i], inF_temp);
            SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp);
            SDOTP_GENERIC(temp2, ((v2s*)addr2)[i], inF_temp);
            SDOTP_GENERIC(temp3, ((v2s*)addr3)[i], inF_temp);
            SDOTP_GENERIC(temp4, ((v2s*)addr4)[i], inF_temp);
            SDOTP_GENERIC(temp5, ((v2s*)addr5)[i], inF_temp);
            SDOTP_GENERIC(temp6, ((v2s*)addr6)[i], inF_temp);
            SDOTP_GENERIC(temp7, ((v2s*)addr7)[i], inF_temp);
// #ifdef FMINTILING
//             SDOTP_GENERIC(temp0, ((v2s*)addr0)[i], inF_temp2);
//             SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp2);
//             SDOTP_GENERIC(temp2, ((v2s*)addr2)[i], inF_temp2);
//             SDOTP_GENERIC(temp3, ((v2s*)addr3)[i], inF_temp2);
//             SDOTP_GENERIC(temp4, ((v2s*)addr4)[i], inF_temp2);
//             SDOTP_GENERIC(temp5, ((v2s*)addr5)[i], inF_temp2);
//             SDOTP_GENERIC(temp6, ((v2s*)addr6)[i], inF_temp2);
//             SDOTP_GENERIC(temp7, ((v2s*)addr7)[i], inF_temp2);
// #endif
          // }
          }
// #ifdef FMINTILING
//         if(inFeaturesSizeP2%2==1) { // add contribution of left over input channel (input channels not multiple of 4)
//             v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
//             asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
//             SDOTP_GENERIC(temp0, ((v2s*)addr0)[i], inF_temp);
//             SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp);
//             SDOTP_GENERIC(temp2, ((v2s*)addr2)[i], inF_temp);
//             SDOTP_GENERIC(temp3, ((v2s*)addr3)[i], inF_temp);
//             SDOTP_GENERIC(temp4, ((v2s*)addr4)[i], inF_temp);
//             SDOTP_GENERIC(temp5, ((v2s*)addr5)[i], inF_temp);
//             SDOTP_GENERIC(temp6, ((v2s*)addr6)[i], inF_temp);
//             SDOTP_GENERIC(temp7, ((v2s*)addr7)[i], inF_temp);

//           }
// #endif
      } // fm in1 and in2
      // for(int o_rel =0;o_rel<outFeaturesPerTile;o_rel++) {
      outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = shiftAndAct(temp0, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+1)] = shiftAndAct(temp1, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+2)] = shiftAndAct(temp2, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+3)] = shiftAndAct(temp3, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+4)] = shiftAndAct(temp4, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+5)] = shiftAndAct(temp5, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+6)] = shiftAndAct(temp6, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+7)] = shiftAndAct(temp7, activationFunction);
      // }
    }
    break;
   #endif
   #if OUTPUTBUFFER > 4
    case 4:

    for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
    {

      temp0 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+0]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+0])<<(q_fraqP1);
      temp1 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+1]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+1])<<(q_fraqP1);
      temp2 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+2]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+2])<<(q_fraqP1);
      temp3 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+3]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+3])<<(q_fraqP1);
      for(int turn=0; turn<2; turn++) {
        if(turn==0) {
          weight_ptr = weight_ptr1;
          inFeaturesSizeP2 = inFeaturesSize1/2;
          inFeatures = inFeatures1;
          #ifdef FMINTILING
          inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
          #else
          inFeaturesSizeP4 = inFeaturesSizeP2; // input FM tiling
          #endif
        } else {
          weight_ptr = weight_ptr2;
          inFeatures = inFeatures2;
          inFeaturesSizeP2 = inFeaturesSize2/2;
          #ifdef FMINTILING
          inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
          #else
          inFeaturesSizeP4 = inFeaturesSizeP2; // input FM tiling
          #endif
        }
        // }
        addr0 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+0))];
        addr1 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+1))];
        addr2 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+2))];
        addr3 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+3))];


       // asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload first weight
       // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload first weight

       in_addr = (uint32_t)(((v2s*)inFeatures));
       for(int i=0; i<inFeaturesSizeP4; i++) {
            v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
            v2s inF_temp2;// = ((v2s*)inFeatures)[2*i+1];


            // [INFO] lwincr with 2i, 2i+1 not mapped by compiler => inline assembly
            asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
// #ifdef FMINTILING
//             asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp2), "+r" (in_addr)); // v2s inF_temp2 = ((v2s*)inFeatures)[2i+1];
// #endif
            // MANUAL loop unfolding
            // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
            // __SUMDOTP2(inF_temp, ((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+o_rel)) + i], temp[o_rel]);
            SDOTP_GENERIC(temp0, ((v2s*)addr0)[i], inF_temp);
            SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp);
            SDOTP_GENERIC(temp2, ((v2s*)addr2)[i], inF_temp);
            SDOTP_GENERIC(temp3, ((v2s*)addr3)[i], inF_temp);
// #ifdef FMINTILING
            
//             SDOTP_GENERIC(temp0, ((v2s*)addr0)[i], inF_temp2);
//             SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp2);
//             SDOTP_GENERIC(temp2, ((v2s*)addr2)[i], inF_temp2);
//             SDOTP_GENERIC(temp3, ((v2s*)addr3)[i], inF_temp2);
// #endif
          // }
          }
// # ifdef FMINTILING
//         if(inFeaturesSizeP2%2==1) { // add contribution of left over input channel (input channels not multiple of 4)
//             v2s inF_temp;//  = ((v2s*)inFeatures)[2*i];
            
//             asm volatile("p.lw %0, 4(%1!)" : "=r" (inF_temp), "+r" (in_addr));  // v2s inF_temp  = ((v2s*)inFeatures)[2i+0];
//             SDOTP_GENERIC(temp0, ((v2s*)addr0)[i], inF_temp);
//             SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp);
//             SDOTP_GENERIC(temp2, ((v2s*)addr2)[i], inF_temp);
//             SDOTP_GENERIC(temp3, ((v2s*)addr3)[i], inF_temp);

//           }
// # endif
      } // fm in1 and in2
      // for(int o_rel =0;o_rel<outFeaturesPerTile;o_rel++) {
      outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = shiftAndAct(temp0, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+1)] = shiftAndAct(temp1, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+2)] = shiftAndAct(temp2, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+3)] = shiftAndAct(temp3, activationFunction);
              // printf("outFeatures[%i]=%i\n", o_rel, outFeatures[(o_tile*outFeaturesPerTile+o_rel)]);
      // }
    }
    break;
#endif
    case 2:

    for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
    {

      temp0 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+0]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+0])<<(q_fraqP1);
      temp1 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+1]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+1])<<(q_fraqP1);
        // temp2 = (int32_t)bias[o_tile*outFeaturesPerTile+2]<<(q_fraqP1);
        // temp3 = (int32_t)bias[o_tile*outFeaturesPerTile+3]<<(q_fraqP1);
        // }

      for(int turn=0; turn<2; turn++) {
        if(turn==0) {
          weight_ptr = &weight_ptr1[0];
          inFeaturesSizeP2 = inFeaturesSize1/2;
          inFeatures = inFeatures1;
          // #ifdef FMINTILING
          // inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
          // #else
          inFeaturesSizeP4 = inFeaturesSizeP2; // input FM tiling
          // #endif
        } else {
          weight_ptr = &weight_ptr2[0];
          inFeatures = inFeatures2;
          inFeaturesSizeP2 = inFeaturesSize2/2;
          // #ifdef FMINTILING
          // inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
          // #else
          inFeaturesSizeP4 = inFeaturesSizeP2; // input FM tiling
          // #endif
        }
        addr0 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+0))];
        addr1 = (uint32_t) &((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+1))];
        // addr2 = &((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+2))];
        // addr3 = &((v2s*)weight)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile+3))];
        // asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (x0), "+r" (addr0) : "r" (x0) ); // preload no compute
        // asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (x0), "+r" (addr1) : "r" (x0) ); // preload no compute
        for(int i=0; i<inFeaturesSizeP2; i++) {
          v2s inF_temp = ((v2s*)inFeatures)[i];
          
          // int o_rel;
          // for (o_rel=0; o_rel < outFeaturesPerTile; o_rel++) {    
          SDOTP_GENERIC(temp0, ((v2s*)addr0)[i], inF_temp);
          SDOTP_GENERIC(temp1, ((v2s*)addr1)[i], inF_temp);
             // asm volatile("pl.sdotsp.h %0, %1, %2" : "+r" (temp2) : "r" (addr3), "r" (inF_temp) );
             // asm volatile("pl.sdotsp.h %0, %1, %2" : "+r" (temp3) : "r" (addr0), "r" (inF_temp) );
          // }
        }
      } // fm in1 in2 
      // for(int o_rel =0;o_rel<outFeaturesPerTile;o_rel++) {
      outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = shiftAndAct(temp0, activationFunction);
      outFeatures_ptr[(o_tile*outFeaturesPerTile+1)] = shiftAndAct(temp1, activationFunction);
              // outFeatures[(o_tile*outFeaturesPerTile+2)] = temp2>>(q_fraqP1);
              // outFeatures[(o_tile*outFeaturesPerTile+3)] = temp3>>(q_fraqP1);
              // printf("outFeatures[%i]=%i\n", o_rel, outFeatures[(o_tile*outFeaturesPerTile+o_rel)]);
      // }

    }
    break;
    case 1:


    for (int o_tile=0; o_tile< outFeatureTiles; o_tile++) 
    {
     // iterate through both input FMs
     for(int turn=0; turn<2; turn++) {
      if(turn==0) {
        weight_ptr = &weight_ptr1[0];
        inFeaturesSizeP2 = inFeaturesSize1/2;
        inFeatures = inFeatures1;
      } else {
        weight_ptr = &weight_ptr2[0];
        inFeatures = inFeatures2;
        inFeaturesSizeP2 = inFeaturesSize2/2;
      }
      // #ifdef FMINTILING
      // inFeaturesSizeP4 = inFeaturesSizeP2/2; // input FM tiling
      // #else
      inFeaturesSizeP4 = inFeaturesSizeP2; // input FM tiling
      // #endif
      temp0 = ((int32_t)bias_ptr1[o_tile*outFeaturesPerTile+0]+(int32_t)bias_ptr2[o_tile*outFeaturesPerTile+0])<<(q_fraqP1);
      for(int i=0; i<inFeaturesSizeP2; i++) {
        v2s inF_temp = ((v2s*)inFeatures)[i];
        temp0 = __SUMDOTP2(inF_temp, ((v2s*)weight_ptr)[(inFeaturesSizeP2*(o_tile*outFeaturesPerTile)) + i], temp0);
      }
   } // fm in1 and in2 
   outFeatures_ptr[(o_tile*outFeaturesPerTile+0)] = shiftAndAct(temp0, activationFunction);
 }
 break;
}
 // updat pointer for next iteration
bias_ptr1                = &bias_ptr1[outFeatureTiles*outFeaturesPerTile];
bias_ptr2                = &bias_ptr2[outFeatureTiles*outFeaturesPerTile];
weight_ptr1              = &weight_ptr1[(inFeaturesSize1/2*(outFeatureTiles*outFeaturesPerTile))];
weight_ptr2              = &weight_ptr2[(inFeaturesSize2/2*(outFeatureTiles*outFeaturesPerTile))];
outFeatures_ptr         = &outFeatures_ptr[(outFeatureTiles*outFeaturesPerTile)];
outFeaturesSize_remain -= outFeatureTiles*outFeaturesPerTile;
if (outFeaturesSize_remain==0) break;
}
PROFILING_TWOLINEAR_END
}
 ///////////////////////////////////////////////////////////////////////////////////////////////
 // ______  _______ _______ _______ _     _        _______      _____ _______  _____          //
 // |     \ |______ |______ |_____| |     | |         |           |   |  |  | |_____] |       //
 // |_____/ |______ |       |     | |_____| |_____    |         __|__ |  |  | |       |_____  //
 //                                                                                           //
 ///////////////////////////////////////////////////////////////////////////////////////////////
#else // no FM Tiling or not ASIP
/** @brief Calculates Two Linear Layers which are accumulated.
 *
 *  Some networks like LSTM calculate a FC layer for two different input (e.g. x and c)
 *  which are then summed together to calculate the output feature map. 
 *  This is an optimization to avoid storing back all the intermediate output FM's to
 *  calculate the 2nd layer.
 *
 *  @param inFeaturesSize1 Number of input neurons of the FC layer 1
 *  @param inFeaturesSize2 Number of input neurons of the FC layer 2
 *  @param outFeaturesSize Number of output neurons
 *  @param activationFunction Type of activation function (ACT_NONE: no activation function is used, ACT_TANH: tangent hyperbolicus, ACT_SIG: sigmoid)
 *  @param weight1 Weights of FC1
 *  @param weight2 Weights of FC2
 *  @param bias1 Bias FC1
 *  @param bias2 Bias FC2
 *  @param inFeatures1 Pointer to input FM for FC1
 *  @param inFeatures2 Pointer to input FM for FC2
 *  @param outFeatures Pointer where to store output FM
 */
void NOINLINE TwoLinearLayersAccumulate (
  // Layer Attributes
  int inFeaturesSize1, int inFeaturesSize2, int outFeaturesSize, int activationFunction, 
  // Layer Parameters
  data_t * __restrict__ weight1,
  data_t * __restrict__ weight2,
  data_t * __restrict__ bias1,
  data_t * __restrict__ bias2,
  // Input and Output Features
  data_t * __restrict__ inFeatures1,
  data_t * __restrict__ inFeatures2,
  data_t * __restrict__ outFeatures)
{

  PROFILING_TWOLINEAR_START

  int inFeaturesSize1P2=inFeaturesSize1/2;
  int inFeaturesSize2P2=inFeaturesSize2/2;

  for (int o=0; o< outFeaturesSize; o++) 
  {
    outFeatures[o] = True?
    bias1[o]+bias2[o]:
    ((data_t)0);
    int32_t temp;
    temp = ((int32_t)bias1[o]+(int32_t)bias2[o])<<(q_fraqP1);

#ifdef SIMD
    for(int i=0; i<inFeaturesSize1P2; i++)
#else // SIMD
    for(int i=0; i<inFeaturesSize1; i++)
#endif // SIMD
    {

#ifdef SIMD
# ifdef ASIP 
      temp = temp + ((v2s*)inFeatures1)[i] * ((v2s*)weight1)[(inFeaturesSize1P2*o) + i];
# else //not ASIP
      temp = __SUMDOTP2(((v2s*)inFeatures1)[i], ((v2s*)weight1)[(inFeaturesSize1P2*o) + i], temp);
# endif //ASIP
#else // not SIMD
      temp += inFeatures1[i]*weight1[inFeaturesSize1*o + i];
#endif

    }

#ifdef SIMD
      for(int i=0; i<inFeaturesSize2P2; i++)
#else
      for(int i=0; i<inFeaturesSize2; i++)
#endif
      {

#ifdef SIMD
# ifdef ASIP 
        temp = temp + ((v2s*)inFeatures2)[i] * ((v2s*)weight2)[(inFeaturesSize2P2*o) + i];
# else // not ASIP
        temp = __SUMDOTP2(((v2s*)inFeatures2)[i], ((v2s*)weight2)[(inFeaturesSize2P2*o) + i], temp);
# endif //ASIP
#else // not SIMD
        temp += inFeatures2[i]*weight2[inFeaturesSize2*o + i];
#endif // end SIMD

      }

#ifdef DOACTONTHEFLY
      temp = temp>>(q_fraqP1); 
      switch(activationFunction)
      {
        case ACT_NONE: outFeatures[o] = temp; break;
        case ACT_TANH: outFeatures[o] = generic_tanh(temp); break;
        case ACT_SIG:  outFeatures[o] = generic_sig(temp); break;
      }
#else // DOACTONTHEFLY
      outFeatures[o] = temp>>(q_fraqP1);
#endif // DOACTONTHEFLY

      }

      PROFILING_TWOLINEAR_END
    }

#endif
#endif



//////////////////////////////////////////////////////////////////////////////////////////////
#ifndef FixedPt

void NOINLINE TwoLinearLayersAccumulate (
  // Layer Attributes
  int inFeaturesSize1, int inFeaturesSize2, int outFeaturesSize, int activationFunction,
  // Layer Parameters
  data_t * __restrict__ weight1,
  data_t * __restrict__ weight2,
  data_t * __restrict__ bias1,
  data_t * __restrict__ bias2,
  // Input and Output Features
  data_t * __restrict__ inFeatures1,
  data_t * __restrict__ inFeatures2,
  data_t * __restrict__ outFeatures)
{

  PROFILING_TWOLINEAR_START

  int inFeaturesSize1P2=inFeaturesSize1/2;
  int inFeaturesSize2P2=inFeaturesSize2/2;

  for (int o=0; o< outFeaturesSize; o++) 
  {
    outFeatures[o] = True?
    bias1[o]+bias2[o]:
    ((data_t)0);
    data_t temp;
    temp = bias1[o]+bias2[o];

    for(int i=0; i<inFeaturesSize1; i++)
    {
      temp += inFeatures1[i]*weight1[inFeaturesSize1*o + i];
    }

    for(int i=0; i<inFeaturesSize2; i++)
    {
      temp += inFeatures2[i]*weight2[inFeaturesSize2*o + i];
    }

    outFeatures[o] = temp;
  }

 PROFILING_TWOLINEAR_END

}

#endif // FixedPt



//////////////////////////////////////////////////////////////////////////////////////////////
/** @brief Calculates point-wise Addition of Tensors (A+=B)
 *
 *  @param TensorSize Input Value
 *  @param FeaturesA Accumulation Tensor
 *  @param FeaturesB Addend Tensor
 */
void NOINLINE AddTensor (
  // Layer Attributes
  int TensorSize,
  // Layer Parameters
  data_t * __restrict__ FeaturesA,
  data_t * __restrict__ FeaturesB)
{

  PROFILING_ADDT_START

#ifdef ASIP

// TODO implement SIMD add on tzscale
  for(int o=0; o<TensorSize; o++)
    FeaturesA[o] += FeaturesB[o];

  return;

#else // ASIP


#ifdef MULTICORE
  /* instructions to parallelize the workload:
  each core computes a balanced number of neurons */
  int core_id = rt_core_id();
  int n_cores = NR_CORES;
  int chunck = 1;
  int chunkg_orig=1;

  int start_offset = 0;
  /* handle the case when number of neurons
  is less than number of cores: chunck=1 */
  if(TensorSize <= n_cores)
  {
      // n_cores = TensorSize;
      n_cores = 1;
      if (core_id == 0)
      {
        chunck = TensorSize;
      } else
      {
        chunck = 0;
      }
  }
  else
  {
      int Log2Core = __builtin_pulp_fl1(n_cores);
      chunck = (TensorSize >> Log2Core) + ((TensorSize & (n_cores-1))!=0);
      chunkg_orig = chunck;
      // printf(" core_id %d a\n",core_id);
      if ((chunck % 2)!=0)
      {
        // printf(" core_id %d b\n",core_id);
        if ((core_id%2)==0)
        {
          // printf(" core_id %d +\n",core_id);
          chunkg_orig = chunck;
          chunck = chunck+1;
        }
        else
        {
          // printf(" core_id %d -\n",core_id);
          chunkg_orig = chunck;
          chunck = chunck-1;
          start_offset = 1;
        }
      }
  }
  /* start and stop neuron to be computed, for each core */
  int start = MIN((chunkg_orig) * core_id+start_offset,TensorSize);
  int stop = MIN(start + chunck, TensorSize);
  int chunck_final = (stop-start);

  // printf("addtensor core_id %d - start: %d - stop: %d \n",core_id, start, stop);


#endif // MULTICORE


#ifdef SIMD

#ifdef MULTICORE

  int TensorSizeP2 = start + chunck_final/2;
  v2s * SIMD_FeaturesA = (v2s*) FeaturesA;
  v2s * SIMD_FeaturesB = (v2s*) FeaturesB;

  for(int o=start; o<stop; o++)
  {
#else // MULTICORE

  int TensorSizeP2 = TensorSize/2;
  v2s * SIMD_FeaturesA = (v2s*) FeaturesA;
  v2s * SIMD_FeaturesB = (v2s*) FeaturesB;

  for(int o=0; o<TensorSizeP2; o++)
  {
#endif // MULTICORE

    SIMD_FeaturesA[o] += SIMD_FeaturesB[o];
  }

#else // SIMD

#ifdef MULTICORE
  for(int o=start; o<stop; o++)
  {
#else // MULTICORE
  for(int o=0; o<TensorSize; o++)
  {
#endif // MULTICORE

    FeaturesA[o] += FeaturesB[o];
  }

#endif // SIMD

  PROFILING_ADDT_END

#endif // ASIP
}


//////////////////////////////////////////////////////////////////////////////////////////////
/** @brief Calculates point-wise Multiplication of Tensors (A*=B) also known as Hadamard Product
 *
 *  @param TensorSize Input Value
 *  @param FeaturesA Accumulation Tensor
 *  @param FeaturesB Multiplicand Tensor
 */
void NOINLINE HadMulTensor (
  // Layer Attributes
  int TensorSize,
  // Layer Parameters
  data_t * __restrict__ FeaturesA,
  data_t * __restrict__ FeaturesB)
{

  PROFILING_HADM_START

#ifdef MULTICORE
  /* instructions to parallelize the workload:
  each core computes a balanced number of neurons */
  int core_id = rt_core_id();
  int n_cores = NR_CORES;
  int chunck = 1;
  int chunkg_orig=1;

  int start_offset = 0;
  /* handle the case when number of neurons
  is less than number of cores: chunck=1 */
  if(TensorSize <= n_cores)
  {
      // n_cores = TensorSize;
      n_cores = 1;
      if (core_id == 0)
      {
        chunck = TensorSize;
      } else
      {
        chunck = 0;
      }
  }
  else
  {
      int Log2Core = __builtin_pulp_fl1(n_cores);
      chunck = (TensorSize >> Log2Core) + ((TensorSize & (n_cores-1))!=0);
      chunkg_orig = chunck;
      // printf(" core_id %d a\n",core_id);
      if ((chunck % 2)!=0)
      {
        // printf(" core_id %d b\n",core_id);
        if ((core_id%2)==0)
        {
          // printf(" core_id %d +\n",core_id);
          chunkg_orig = chunck;
          chunck = chunck+1;
        }
        else
        {
          // printf(" core_id %d -\n",core_id);
          chunkg_orig = chunck;
          chunck = chunck-1;
          start_offset = 1;
        }
      }
  }
  /* start and stop neuron to be computed, for each core */
  int start = MIN((chunkg_orig) * core_id+start_offset,TensorSize);
  int stop = MIN(start + chunck, TensorSize);
  int chunck_final = (stop-start);

  // printf(" hadmul core_id %d - start: %d - stop: %d \n",core_id, start, stop);

  for (int o=start; o<stop; o++) 
  {
#else
  for (int o=0; o<TensorSize; o++) 
  {
#endif

#ifdef FixedPt
    FeaturesA[o] = (FeaturesA[o]*FeaturesB[o])>>(q_fraqP1);
#else
    FeaturesA[o] *= FeaturesB[o];
#endif

  }
  PROFILING_HADM_END

}


//////////////////////////////////////////////////////////////////////////////////////////////
/** @brief Copy of Tensor A to B
 *
 *  @param TensorSize Input Value
 *  @param FeaturesA Source Tensor
 *  @param FeaturesB Destination Tensor
 */
void NOINLINE CopyTensor (
  // Layer Attributes
  int TensorSize,
  // Layer Parameters
  data_t * __restrict__ FeaturesA,
  data_t * __restrict__ FeaturesB)
 {
  PROFILING_COPY_START
  for (int o=0; o< TensorSize; o++) 
  {
    FeaturesA[o] = FeaturesB[o];
  }
  PROFILING_COPY_END
}


//////////////////////////////////////////////////////////////////////////////////////////////
/** @brief In-Place application of tangent hyperbolic on Tensor
 *
 *  @param TensorSize Input Value
 *  @param Features Input and Output of Activation Fucntion
 */
#ifdef ASIP_USETANHSIG

void NOINLINE TanhLayer (
  // Layer Attributes
  int TensorSize,
  data_t * __restrict__ Features)
{
  PROFILING_TANH_START
  for(int i=0; i<TensorSize; i++) {
    Features[i] = tzscale_tanh(Features[i]); //tzscale_tanh(Features[i],0);
  }
  PROFILING_TANH_END
}

//////////////////////////////////////////////////////////////////////////////////////////////
#else // ASIP_USETANHSIG

void NOINLINE TanhLayer (
  // Layer Attributes
  int TensorSize,
  data_t * __restrict__ Features)
{
  PROFILING_TANH_START

#ifdef MULTICORE
  /* instructions to parallelize the workload:
  each core computes a balanced number of neurons */
  int core_id = rt_core_id();
  int n_cores = NR_CORES;
  int chunck = 1;
  int chunkg_orig=1;

  int start_offset = 0;
  /* handle the case when number of neurons
  is less than number of cores: chunck=1 */
  if(TensorSize <= n_cores)
  {
      // n_cores = TensorSize;
      n_cores = 1;
      if (core_id == 0)
      {
        chunck = TensorSize;
      } else
      {
        chunck = 0;
      }
  }
  else
  {
      int Log2Core = __builtin_pulp_fl1(n_cores);
      chunck = (TensorSize >> Log2Core) + ((TensorSize & (n_cores-1))!=0);
      chunkg_orig = chunck;
      // printf(" core_id %d a\n",core_id);
      if ((chunck % 2)!=0)
      {
        // printf(" core_id %d b\n",core_id);
        if ((core_id%2)==0)
        {
          // printf(" core_id %d +\n",core_id);
          chunkg_orig = chunck;
          chunck = chunck+1;
        }
        else
        {
          // printf(" core_id %d -\n",core_id);
          chunkg_orig = chunck;
          chunck = chunck-1;
          start_offset = 1;
        }
      }
  }
  /* start and stop neuron to be computed, for each core */
  int start = MIN((chunkg_orig) * core_id+start_offset,TensorSize);
  int stop = MIN(start + chunck, TensorSize);
  int chunck_final = (stop-start);

  for (int o=start; o<stop; o++) 
  {
#else
  for (int o=0; o<TensorSize; o++) 
  {
#endif

    Features[o] = generic_tanh(Features[o]);
  }

  PROFILING_TANH_END
}

#endif // ASIP_USETANHSIG
//////////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////////////
/** @brief In-Place application of sigmoid activation on Tensor
 *
 *  @param TensorSize Input Value
 *  @param Features Input and Output of Activation Fucntion
 */
#ifdef ASIP_USETANHSIG

void NOINLINE SigLayer (
  // Layer Attributes
  int TensorSize,
  data_t * __restrict__ Features)
{
  PROFILING_TANH_START

  for(int i=0; i<TensorSize; i++) {
    //ext_nop();chess_separator_scheduler();
    Features[i] = tzscale_sig(Features[i]); //tzscale_tanh(Features[i],0);
  }
  PROFILING_TANH_END
}

//////////////////////////////////////////////////////////////////////////////////////////////
#else // ASIP_USETANHSIG

void NOINLINE SigLayer (
  // Layer Attributes
  int TensorSize,
  data_t * __restrict__ Features)
{
  PROFILING_TANH_START

  #ifdef MULTICORE
  /* instructions to parallelize the workload:
  each core computes a balanced number of neurons */
  int core_id = rt_core_id();
  int n_cores = NR_CORES;
  int chunck = 1;
  int chunkg_orig=1;

  int start_offset = 0;
  /* handle the case when number of neurons
  is less than number of cores: chunck=1 */
  if(TensorSize <= n_cores)
  {
      // n_cores = TensorSize;
      n_cores = 1;
      if (core_id == 0)
      {
        chunck = TensorSize;
      } else
      {
        chunck = 0;
      }
  }
  else
  {
      int Log2Core = __builtin_pulp_fl1(n_cores);
      chunck = (TensorSize >> Log2Core) + ((TensorSize & (n_cores-1))!=0);
      chunkg_orig = chunck;
      // printf(" core_id %d a\n",core_id);
      if ((chunck % 2)!=0)
      {
        // printf(" core_id %d b\n",core_id);
        if ((core_id%2)==0)
        {
          // printf(" core_id %d +\n",core_id);
          chunkg_orig = chunck;
          chunck = chunck+1;
        }
        else
        {
          // printf(" core_id %d -\n",core_id);
          chunkg_orig = chunck;
          chunck = chunck-1;
          start_offset = 1;
        }
      }
  }
  /* start and stop neuron to be computed, for each core */
  int start = MIN((chunkg_orig) * core_id+start_offset,TensorSize);
  int stop = MIN(start + chunck, TensorSize);
  int chunck_final = (stop-start);

  // printf(" core_id %d - start: %d - stop: %d \n",core_id, start, stop);

  for (int o=start; o<stop; o++) 
  {
#else
  for (int o=0; o<TensorSize; o++) 
  {
#endif

    Features[o] = generic_sig(Features[o]);
  }

  PROFILING_TANH_END
}

#endif // ASIP_USETANHSIG
//////////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////////////
/** @brief Fills Tensor with contstant
 *
 *  @param TensorSize Input Value
 *  @param Tensor Input and Output of Activation Fucntion
 *  @param fillValue constant value
 */
void NOINLINE fillTensor (
  // Layer Attributes
  int TensorSize,
  data_t * __restrict__ Tensor,
  data_t fillValue)
{
  PROFILING_FILL_START
  for (int o=0; o< TensorSize; o++) 
  {
    Tensor[o] = fillValue;
  }
  PROFILING_FILL_END
}



// inline data_t  ALWAYS_INLINE  Tanh_old(data_t value) {
// #ifdef TURNOFF
//   #ifdef ASIP  
//   int tmp = Max(Min(15, (((int)value+18432-2048)>>11)),0);
// #else // ASIP
//   #ifndef USE_INTRINSICS
//   int tmp = Max(Min(15, (((int)value+18432-2048)>>11)),0);
//   #else // USET_INTRINSICS
// // [START clipur version]
//    int tmp = (((int)value+18432-2048)>>11);
//    asm volatile ("p.clipur" " %[c], %[a],%[b]\n"
//         : [c] "=r" (tmp)
//         : [a] "r" (tmp), [b] "r" (lb_lut_numelements)); // TODO: check puts p.clipu instead!!
// // [END clipur version]
//   #endif // USET_INTRINSICS
// #endif // ASIP
//   return ((lut_Tanh_m[tmp]*value) >> (q_fraqP1))+lut_Tanh_q[tmp];

// #else // TURNOFF
// #ifdef FixedPt 
//   //printf("%i, ", PIHalf);
//   if(Abs(value) < tanh_threshold) 
//   {

//     int x_power = value;
//     int partSum = value;

//     for(int i = 1; i<4; i++) {
//       x_power = ((x_power * value) >> (q_fraqP1)) * value >> (q_fraqP1);
//       // printf("%i, ", x_power);
//       partSum = partSum + (tanh_coeff[i]*x_power >> (q_fraqP1));
//     }
//     // printf("%i==%i, ", partSum, (data_t)((1.0f - 2.0/(expTailor(tailorPrecission, 2*(float)(value)/(1<<(q_fraqP1)))+1))*(1<<(q_fraqP1))));
//     return partSum;
//   } else {
//     // printf("tanh");
//     return (data_t)((1.0f - 2.0/(expTailor(tailorPrecission, 2*(float)(value)/(1<<(q_frac)))+1))*(1<<(q_frac)));
//   }  
// #else // FixedPt
//          return (1.0f - 2.0/(expTailor(tailorPrecission, 2*value)+1));
// #endif // FixedPt
//   #endif // TURNOFF
// }
// // inline v2s ALWAYS_INLINE sig_SIMD(v2s value)
// // {
// //    // TODO: NOT WORKING at the moment
// // #ifdef SIMD
// // #ifdef TURNOFF
// //   // return value;
// // // printf("qwer\n");
// //   int tmp = Max(Min(8, Abs(((int)value+2048)/4096)),0);
// //   return ((lut_fakeTanh_m[tmp]*value) >> (q_fraqP1))+lut_fakeTanh_q[tmp];
// // #endif // TURNOFF
// // #endif // SIMD

// //   printf("This function should not be called this way!");

// // }


// inline data_t ALWAYS_INLINE sig_old(data_t value)
// {

// #ifdef TURNOFF
// #ifdef ASIP
//   int tmp = Max(Min(15, (((int)value+18432-2048)>>11)),0);
// #else // ASIP
//   #ifndef USE_INTRINSICS
//   int tmp = Max(Min(15, (((int)value+18432-2048)>>11)),0);
//   #else // USET_INTRINSICS
// // [START clipur version]
//    int tmp = (((int)value+2048)/4096);
//    asm volatile ("p.clipur" " %[c], %[a],%[b]\n"
//         : [c] "=r" (tmp)
//         : [a] "r" (tmp), [b] "r" (lb_lut_numelements)); // TODO: check puts p.clipu instead!!
// // [END clipur version]
//   #endif // USET_INTRINSICS
// #endif //ASIP
//   return ((lut_sig_m[tmp]*value) >> (q_fraqP1))+lut_sig_q[tmp];

// #else
//   #ifdef FixedPt
//     if(Abs(value) < sig_threshold) 
//   {
//     int x_power = value;
//     int partSum = sig_coeff[0]+ (sig_coeff[1]*value >> (q_fraqP1));;

//     for(int i = 2; i<4; i++) {
//       x_power = (x_power * value >> (q_fraqP1)) * value >> (q_fraqP1);
//       // printf("(%i), ", sig_coeff[3]);
//       partSum = partSum + ((sig_coeff[i]*x_power) >> (q_frac));
      
//     }
//     // printf("%i==%i, ", partSum, (data_t)((1.0f - 2.0/(expTailor(tailorPrecission, 2*(float)(value)/(1<<(q_fraqP1)))+1))*(1<<(q_fraqP1))));
//     return partSum;
//   } else {
//          // printf("sig, %i>>%i:", value, q_frac);
//          //printFloat((data_t)((1.0/(1.0+expTailor(tailorPrecission, -(float)(value)/(1<<(q_frac)))))*(1<<(q_frac))));

//          return (data_t)((1.0/(1.0+expTailor(tailorPrecission, -(float)(value)/(1<<(q_frac)))))*(1<<(q_frac)));
//   }
//   #else

//          return 1.0/(1.0+expTailor(tailorPrecission, -((float)value)));
//   #endif
// #endif  
// }




//////////////////////////////////////////////////////////////////////////////////////////////
/** @brief Calculates an RNN layer
 *
 *  Calculates an RNN layer based on 
 *  h_t = \\tanh(w_{ih} x_t + b_{ih}  +  w_{hh} h_{(t-1)} + b_{hh})
 *  @param inFeaturesSize Number of input neurons
 *  @param hiddenFeaturesSize Number of hidden neurons
 *  @param weight_ih_l Weights mapping input neurons to hidden neurons
 *  @param weight_hh_l Weights mapping hidden neurons to hidden neurons
 *  @param bias_ih_l Bias mapping input neurons to hidden neurons
 *  @param bias_hh_l Bias mapping hidden neurons to hidden neurons
 *  @param inFeatures Input Feature Map
 *  @param outFeatures Output Feature Map
 *  @param hiddenFeatures Hidden Feature Map
 */
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
  data_t * __restrict__ hiddenFeatures)
{

  for(int seq=0; seq< rnn_seqSize; seq++) {

#ifdef EFFICIENT_CORE_ASSIGNMENT
    printf("ERROR: not implemented RNNLayer with TILING_HARD");
#ifdef BATCHING
    LinearLayer(hiddenFeaturesSize,hiddenFeaturesSize,0,True,1,(data_t*)weight_hh_l, bias_hh_l, hiddenFeatures, outFeatures); //w_{hh} h_{(t-1)}
    LinearLayer(inFeaturesSize,hiddenFeaturesSize,0,True,1,(data_t*)weight_ih_l, bias_ih_l, inFeatures+seq*inFeaturesSize, hiddenFeatures); //w_{ih} x_t + b_{ih}
#else
    LinearLayer(hiddenFeaturesSize,hiddenFeaturesSize,0,True,(data_t*)weight_hh_l, bias_hh_l, hiddenFeatures, outFeatures); //w_{hh} h_{(t-1)}
    LinearLayer(inFeaturesSize,hiddenFeaturesSize,0,True,(data_t*)weight_ih_l, bias_ih_l, inFeatures+seq*inFeaturesSize, hiddenFeatures); //w_{ih} x_t + b_{ih}
#endif
#else
#ifdef BATCHING
    LinearLayer(hiddenFeaturesSize,hiddenFeaturesSize,True,1,(data_t*)weight_hh_l, bias_hh_l, hiddenFeatures, outFeatures); //w_{hh} h_{(t-1)}
    LinearLayer(inFeaturesSize,hiddenFeaturesSize,True,1,(data_t*)weight_ih_l, bias_ih_l, inFeatures+seq*inFeaturesSize, hiddenFeatures); //w_{ih} x_t + b_{ih} 
#else
    LinearLayer(hiddenFeaturesSize,hiddenFeaturesSize,True,(data_t*)weight_hh_l, bias_hh_l, hiddenFeatures, outFeatures); //w_{hh} h_{(t-1)}
    LinearLayer(inFeaturesSize,hiddenFeaturesSize,True,(data_t*)weight_ih_l, bias_ih_l, inFeatures+seq*inFeaturesSize, hiddenFeatures); //w_{ih} x_t + b_{ih} 
#endif
#endif

    AddTensor(hiddenFeaturesSize, outFeatures, hiddenFeatures);
    TanhLayer(hiddenFeaturesSize, outFeatures);
    CopyTensor(hiddenFeaturesSize, hiddenFeatures, outFeatures);
  }

}


#ifdef LSTM_OPT

#ifdef LSTM_HIGH_OPT

//////////////////////////////////////////////////////////////////////////////////////////////
/** @brief Calculates an LSTM layer
 *  @param inFeaturesSize Number of input neurons
 *  @param hiddenFeaturesSize Number of hidden neurons
 *  @param weight_ih_l Weights mapping input neurons to hidden neurons
 *  @param weight_hh_l Weights mapping hidden neurons to hidden neurons
 *  @param bias_ih_l Bias mapping input neurons to hidden neurons
 *  @param bias_hh_l Bias mapping hidden neurons to hidden neurons
 *  @param lstm_h hidden state tensor
 *  @param lstm_c cell state tensor
 *  @param lstm_f forget gate activation tensor
 *  @param inFeatures input feature map
 *  @param lstm_i input/update gate activation tensor
 *  @param lstm_g g tensor 
 *  @param lstm_o output gate tensor
 */
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
  data_t * __restrict__ lstm_h,
  // Hidden Features
  data_t * __restrict__ lstm_c,
  // intermediate nodes
  data_t * __restrict__ lstm_h_out,
  data_t * __restrict__ lstm_f,
  data_t * __restrict__ lstm_i,
  data_t * __restrict__ lstm_g,
  data_t * __restrict__ lstm_o
)
{


  /* instructions to parallelize the workload:
  each core computes a balanced number of neurons */
  int core_id = rt_core_id();
  int n_cores = NR_CORES;

#ifdef PROFILING_LSTM
  synch_barrier();
  if ( core_id==0 )
  {
    PROFILING_LSTM_START
  }
#endif
#ifdef PROFILING_LSTM_AMDAHL_SERIELL
  synch_barrier();
  if ( core_id==0 )
  {
    PROFILING_LSTM_AMDAHL_SERIELL_START
  }
#endif

  int chunck = 1;
  int chunkg_orig = 1;
  int start_offset = 0;
  /* handle the case when number of neurons
  is less than number of cores: chunck=1 */
  if(hiddenFeaturesSize <= n_cores)
  {
      // n_cores = TensorSize;
      n_cores = 1;
      if (core_id == 0)
      {
        chunck = hiddenFeaturesSize;
      } else
      {
        chunck = 0;
      }

      // n_cores = TensorSize;
      // n_cores = 1;
      // if (core_id < hiddenFeaturesSize)
      // {
      //   chunck = 1;
      // } else
      // {
      //   chunck = 0;
      // }
  }
  else
  {
    int Log2Core = __builtin_pulp_fl1(n_cores);
    chunck = (hiddenFeaturesSize >> Log2Core) + ((hiddenFeaturesSize & (n_cores-1))!=0);
    chunkg_orig = chunck;
    // printf(" core_id %d a\n",core_id);
    if ((chunck % 2)!=0)
    {
      // printf(" core_id %d b\n",core_id);
      if ((core_id%2)==0)
      {
        // printf(" core_id %d +\n",core_id);
        chunkg_orig = chunck;
        chunck = chunck+1;
      }
      else
      {
        // printf(" core_id %d -\n",core_id);
        chunkg_orig = chunck;
        chunck = chunck-1;
        start_offset = 1;
      }
    }
  }
  /* start and stop neuron to be computed, for each core */
  int start = MIN((chunkg_orig) * core_id+start_offset,hiddenFeaturesSize);
  int stop = MIN(start + chunck, hiddenFeaturesSize);
  int chunck_final = (stop-start);

  // printf("lstm core_id %d - start: %d - stop: %d - chunck_final %d \n",core_id, start, stop, chunck_final);


#ifdef DEBUG_LSTM
  if ( core_id<NR_CORES )
  {
    printf("lstm_in: "); PrintTensor(inFeaturesSize, inFeatures);
  }
#endif

#ifdef MULTI_INF
  for(int seq=0; seq<lstm_seqSize; seq++)
  {
#endif // MULTI_INF

    //it=σ(Wiixt+bii+Whih(t−1)+bhi)
    if ( core_id<NR_CORES )
    {

      TwoLinearLayersAccumulate (
        // Layer Attributes
        inFeaturesSize, hiddenFeaturesSize, chunck_final, ACT_SIG, 
        // Layer Parameters
        weight_ih_l + start*inFeaturesSize     + 0*inFeaturesSize*hiddenFeaturesSize, // weight1
        weight_hh_l + start*hiddenFeaturesSize + 0*hiddenFeaturesSize*hiddenFeaturesSize, // weight2
        bias_ih_l   + start + 0*hiddenFeaturesSize,   // bias1
        bias_hh_l   + start + 0*hiddenFeaturesSize,   // bias2 
#ifdef MULTI_INF
        inFeatures  + seq*inFeaturesSize,    // in1
#else
        inFeatures,    // in1
#endif
        lstm_h,        // in2
        lstm_i + start);       // out


#ifdef DEBUG_LSTM
      printf("lstm_i: "); PrintTensor(hiddenFeaturesSize, lstm_i);
#endif

#ifndef DOACTONTHEFLY
      SigLayer(hiddenFeaturesSize, lstm_i);
#endif

#ifdef DEBUG_LSTM
      printf("lstm_i: "); PrintTensor(hiddenFeaturesSize, lstm_i);
#endif
    //ft=σ(Wif xt+bif+Whf h(t−1)+bhf)

      TwoLinearLayersAccumulate (
        // Layer Attributes
        inFeaturesSize, hiddenFeaturesSize, chunck_final, ACT_SIG, 
        // Layer Parameters
        weight_ih_l + start*inFeaturesSize      + 1*inFeaturesSize*hiddenFeaturesSize, // weight1
        weight_hh_l + start*hiddenFeaturesSize  + 1*hiddenFeaturesSize*hiddenFeaturesSize, // weight2
        bias_ih_l   + start + 1*hiddenFeaturesSize,   // bias1
        bias_hh_l   + start + 1*hiddenFeaturesSize,   // bias2 
#ifdef MULTI_INF
        inFeatures  + seq*inFeaturesSize,    // in1
#else
        inFeatures,    // in1
#endif
        lstm_h,        // in2
        lstm_f + start);       // out


#ifdef DEBUG_LSTM
    if ( core_id<NR_CORES )
    {
      printf("lstm_f: "); PrintTensor(hiddenFeaturesSize, lstm_f);
    }
#endif

#ifndef DOACTONTHEFLY
      SigLayer(hiddenFeaturesSize, lstm_f);
#endif

#ifdef DEBUG_LSTM
      printf("lstm_f: "); PrintTensor(hiddenFeaturesSize, lstm_f);
#endif
    //gt=tanh(Wigxt+big+Whgh(t−1)+bhg)
      TwoLinearLayersAccumulate (
        // Layer Attributes
        inFeaturesSize, hiddenFeaturesSize, chunck_final, ACT_TANH, 
        // Layer Parameters
        weight_ih_l + start*inFeaturesSize     + 2*inFeaturesSize*hiddenFeaturesSize, // weight1
        weight_hh_l + start*hiddenFeaturesSize + 2*hiddenFeaturesSize*hiddenFeaturesSize, // weight2
        bias_ih_l   + start + 2*hiddenFeaturesSize,   // bias1
        bias_hh_l   + start + 2*hiddenFeaturesSize,   // bias2 
#ifdef MULTI_INF
        inFeatures  + seq*inFeaturesSize,    // in1
#else
        inFeatures,    // in1
#endif
        lstm_h,        // in2
        lstm_g + start);       // out

#ifdef DEBUG_LSTM
      printf("lstm_g: "); PrintTensor(hiddenFeaturesSize, lstm_g);
#endif

#ifndef DOACTONTHEFLY
      TanhLayer(hiddenFeaturesSize, lstm_g);
#endif

#ifdef DEBUG_LSTM
      printf("lstm_g: "); PrintTensor(hiddenFeaturesSize, lstm_g);
#endif
    // synch_barrier();
    //ot=σ(Wioxt+bio+Whoh(t−1)+bho)
      TwoLinearLayersAccumulate (
        // Layer Attributes
        inFeaturesSize, hiddenFeaturesSize, chunck_final,ACT_SIG, 
        // Layer Parameters
        weight_ih_l + start*inFeaturesSize     + 3*inFeaturesSize*hiddenFeaturesSize, // weight1
        weight_hh_l + start*hiddenFeaturesSize + 3*hiddenFeaturesSize*hiddenFeaturesSize, // weight2
        bias_ih_l   + start + 3*hiddenFeaturesSize,   // bias1
        bias_hh_l   + start + 3*hiddenFeaturesSize,   // bias2 
#ifdef MULTI_INF
        inFeatures  + seq*inFeaturesSize,    // in1
#else
        inFeatures,    // in1
#endif
        lstm_h,        // in2
        lstm_o + start);       // out

#ifndef DOACTONTHEFLY
      SigLayer(hiddenFeaturesSize, lstm_o);
#endif

#ifdef DEBUG_LSTM
      printf("lstm_o: "); PrintTensor(hiddenFeaturesSize, lstm_o);
#endif
    }

#if defined(PROFILING_NEW) || defined(PROFILING)
    synch_barrier();
    if(core_id==0)
    {
      PROFILING_LSTM_AMDAHL_SERIELL_END
      PROFILING_LSTM_AMDAHL_PARALLEL_START
    }
#endif
    //ct=ft*c(t−1)+it*gt
    if ( core_id<NR_CORES )
    {

      PROFILING_HADM_START  
      for (int o=start; o<stop; o++) 
      {
  #ifdef FixedPt
        lstm_c[o] = (lstm_c[o]*lstm_f[o])>>(q_fraqP1);
        lstm_i[o] = (lstm_i[o]*lstm_g[o])>>(q_fraqP1);
  #else // FixedPt
        lstm_c[o] *= lstm_f[o];
        lstm_i[o] *= lstm_g[o];
  #endif // FixedPt

      }
      PROFILING_HADM_END
      // PROFILING_LSTM_AMDAHL_PARALLEL_END
      // PROFILING_LSTM_AMDAHL_SERIELL_START
    }
    synch_barrier();
    if ( core_id<NR_CORES )
    {
      // printf("lstm_c tmp: "); PrintTensor(hiddenFeaturesSize, lstm_c);
      // printf("lstm_i tmp: "); PrintTensor(hiddenFeaturesSize, lstm_i);
      // PROFILING_LSTM_AMDAHL_SERIELL_END
      // PROFILING_LSTM_AMDAHL_PARALLEL_START
      PROFILING_ADDT_START
#ifdef SIMD

      int TensorSizeP2 = start + chunck_final/2;
      v2s * SIMD_FeaturesA = (v2s*) lstm_c;
      v2s * SIMD_FeaturesB = (v2s*) lstm_i;

      for(int o=start; o<stop; o++)
      {
        SIMD_FeaturesA[o] += SIMD_FeaturesB[o];
      }

#else // SIMD

      for(int o=start; o<stop; o++)
      {
        lstm_c[o] += lstm_i[o];
      }

#endif // SIMD

      PROFILING_ADDT_END
      // PROFILING_LSTM_AMDAHL_PARALLEL_END
      // PROFILING_LSTM_AMDAHL_SERIELL_START
      // HadMulTensor(hiddenFeaturesSize, lstm_c, lstm_f);
      // HadMulTensor(hiddenFeaturesSize, lstm_i, lstm_g);
      // AddTensor(hiddenFeaturesSize, lstm_c, lstm_i);
    }

#ifdef DEBUG_LSTM
    if ( core_id<NR_CORES )
    {
      printf("lstm_c: "); PrintTensor(hiddenFeaturesSize, lstm_c);
    }
#endif

    //ht=ottanh(ct)
    if ( rt_core_id()==0 )
    {
        PROFILING_LSTM_AMDAHL_PARALLEL_END
        PROFILING_LSTM_AMDAHL_SERIELL_START

        PROFILING_COPY_START
        for (int o=0; o< hiddenFeaturesSize; o++) 
        {
          // lstm_h_out[o] = generic_tanh(lstm_c[o]);
#ifdef FixedPt
          lstm_h_out[o] = (generic_tanh(lstm_c[o]) * lstm_o[o])>>(q_fraqP1);
#else
          lstm_h_out[o] = lstm_o[o] * generic_tanh(lstm_c[o]);
#endif

        }
        PROFILING_COPY_END

    }
    // synch_barrier();


#ifdef DEBUG_LSTM
    if ( core_id<NR_CORES )
    {
      printf("lstm_h_out: "); PrintTensor(hiddenFeaturesSize, lstm_h_out);
    }
#endif

#ifdef MULTI_INF
  }
#endif // MULTI_INF

#ifdef PROFILING_LSTM
  synch_barrier();
  if ( core_id==0 )
  {
    PROFILING_LSTM_END
  }
#endif
#ifdef PROFILING_LSTM_AMDAHL_PARALLEL
  synch_barrier();
  if ( core_id==0 )
  { 
        PROFILING_LSTM_AMDAHL_SERIELL_END
        PROFILING_LSTM_AMDAHL_PARALLEL_START
  }
#endif

}



#else //LSTM_HIGH_OPT

//////////////////////////////////////////////////////////////////////////////////////////////
/** @brief Calculates an LSTM layer
 *  @param inFeaturesSize Number of input neurons
 *  @param hiddenFeaturesSize Number of hidden neurons
 *  @param weight_ih_l Weights mapping input neurons to hidden neurons
 *  @param weight_hh_l Weights mapping hidden neurons to hidden neurons
 *  @param bias_ih_l Bias mapping input neurons to hidden neurons
 *  @param bias_hh_l Bias mapping hidden neurons to hidden neurons
 *  @param lstm_h hidden state tensor
 *  @param lstm_c cell state tensor
 *  @param lstm_f forget gate activation tensor
 *  @param inFeatures input feature map
 *  @param lstm_i input/update gate activation tensor
 *  @param lstm_g g tensor 
 *  @param lstm_o output gate tensor
 */
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
  data_t * __restrict__ lstm_h,
  // Hidden Features
  data_t * __restrict__ lstm_c,
  // intermediate nodes
  data_t * __restrict__ lstm_h_out,
  data_t * __restrict__ lstm_f,
  data_t * __restrict__ lstm_i,
  data_t * __restrict__ lstm_g,
  data_t * __restrict__ lstm_o
)
{


  /* instructions to parallelize the workload:
  each core computes a balanced number of neurons */
  int core_id = rt_core_id();
  int n_cores = NR_CORES;

  if ( core_id<NR_CORES )
  {
    PROFILING_LSTM_START
  }

  // int chunck = 1;
  // int chunkg_orig=1;

  // int start_offset = 0;
  /* handle the case when number of neurons
  is less than number of cores: chunck=1 */
  // if(TensorSize <= n_cores)
  // {
  //     // n_cores = TensorSize;
  //     n_cores = 1;
  //     if (core_id == 0)
  //     {
  //       chunck = TensorSize;
  //     } else
  //     {
  //       chunck = 0;
  //     }
  // }
  // else
  // {
  // int Log2Core = __builtin_pulp_fl1(n_cores);
  // chunck = (hiddenFeaturesSize >> Log2Core) + ((hiddenFeaturesSize & (n_cores-1))!=0);
  // chunkg_orig = chunck;
  // // printf(" core_id %d a\n",core_id);
  // if ((chunck % 2)!=0)
  // {
  //   // printf(" core_id %d b\n",core_id);
  //   if ((core_id%2)==0)
  //   {
  //     // printf(" core_id %d +\n",core_id);
  //     chunkg_orig = chunck;
  //     chunck = chunck+1;
  //   }
  //   else
  //   {
  //     // printf(" core_id %d -\n",core_id);
  //     chunkg_orig = chunck;
  //     chunck = chunck-1;
  //     start_offset = 1;
  //   }
  // }
  // // }
  // /* start and stop neuron to be computed, for each core */
  // int start = MIN((chunkg_orig) * core_id+start_offset,hiddenFeaturesSize);
  // int stop = MIN(start + chunck, hiddenFeaturesSize);
  // int chunck_final = (stop-start);

  // printf("lstm core_id %d - start: %d - stop: %d \n",core_id, start, stop);

  // int core_id = rt_core_id();

#ifdef DEBUG_LSTM
  if ( core_id<NR_CORES )
  {
    printf("lstm_in: "); PrintTensor(inFeaturesSize, inFeatures);
  }
#endif
  
  for(int seq=0; seq<lstm_seqSize; seq++) {

    //it=σ(Wiixt+bii+Whih(t−1)+bhi)
    if ( core_id<NR_CORES )
    {

      TwoLinearLayersAccumulate (
        // Layer Attributes
        inFeaturesSize, hiddenFeaturesSize, hiddenFeaturesSize, ACT_SIG, 
        // Layer Parameters
        weight_ih_l+0*inFeaturesSize*hiddenFeaturesSize, // weight1
        weight_hh_l+0*hiddenFeaturesSize*hiddenFeaturesSize, // weight2
        bias_ih_l+0*hiddenFeaturesSize,   // bias1
        bias_hh_l+0*hiddenFeaturesSize,   // bias2 
        inFeatures+seq*inFeaturesSize,    // in1
        lstm_h,        // in2
        lstm_i);       // out

    // }
    // synch_barrier();

#ifdef DEBUG_LSTM
    // if ( core_id<NR_CORES )
    // {
      printf("lstm_i: "); PrintTensor(hiddenFeaturesSize, lstm_i);
    // }
#endif

#ifndef DOACTONTHEFLY
    // if ( rt_core_id()==0 )
    // if ( core_id<NR_CORES )
    // {
      SigLayer(hiddenFeaturesSize, lstm_i);
    // }
#endif

#ifdef DEBUG_LSTM
    // if ( core_id<NR_CORES )
    // {
      printf("lstm_i: "); PrintTensor(hiddenFeaturesSize, lstm_i);
    // }
#endif
    // synch_barrier();
    //ft=σ(Wif xt+bif+Whf h(t−1)+bhf)
    // if ( rt_core_id()<NR_CORES )
    // {
      TwoLinearLayersAccumulate (
        // Layer Attributes
        inFeaturesSize, hiddenFeaturesSize, hiddenFeaturesSize, ACT_SIG, 
        // Layer Parameters
        weight_ih_l+1*inFeaturesSize*hiddenFeaturesSize, // weight1
        weight_hh_l+1*hiddenFeaturesSize*hiddenFeaturesSize, // weight2
        bias_ih_l+1*hiddenFeaturesSize,   // bias1
        bias_hh_l+1*hiddenFeaturesSize,   // bias2 
        inFeatures+seq*inFeaturesSize,    // in1
        lstm_h,        // in2
        lstm_f);       // out
    // }
    // synch_barrier();

#ifdef DEBUG_LSTM
    if ( core_id<NR_CORES )
    {
      printf("lstm_f: "); PrintTensor(hiddenFeaturesSize, lstm_f);
    }
#endif

#ifndef DOACTONTHEFLY
    // if ( rt_core_id()==0 )
    // if ( core_id<NR_CORES )
    // {
      SigLayer(hiddenFeaturesSize, lstm_f);
    // }
#endif

#ifdef DEBUG_LSTM
    // if ( core_id<NR_CORES )
    // {
      printf("lstm_f: "); PrintTensor(hiddenFeaturesSize, lstm_f);
    // }
#endif
    // synch_barrier();
    //gt=tanh(Wigxt+big+Whgh(t−1)+bhg)
    // if ( rt_core_id()<NR_CORES )
    // {
      TwoLinearLayersAccumulate (
        // Layer Attributes
        inFeaturesSize, hiddenFeaturesSize, hiddenFeaturesSize, ACT_TANH, 
        // Layer Parameters
        weight_ih_l+2*inFeaturesSize*hiddenFeaturesSize, // weight1
        weight_hh_l+2*hiddenFeaturesSize*hiddenFeaturesSize, // weight2
        bias_ih_l+2*hiddenFeaturesSize,   // bias1
        bias_hh_l+2*hiddenFeaturesSize,   // bias2 
        inFeatures+seq*inFeaturesSize,    // in1
        lstm_h,        // in2
        lstm_g);       // out
    // }
    // synch_barrier();

#ifdef DEBUG_LSTM
    // if ( core_id<NR_CORES )
    // {
      printf("lstm_g: "); PrintTensor(hiddenFeaturesSize, lstm_g);
    // }
#endif

#ifndef DOACTONTHEFLY
    // if ( rt_core_id()==0 )
    // if ( core_id<NR_CORES )
    // {
      TanhLayer(hiddenFeaturesSize, lstm_g);
    // }
#endif

#ifdef DEBUG_LSTM
    // if ( core_id<NR_CORES )
    // {
      printf("lstm_g: "); PrintTensor(hiddenFeaturesSize, lstm_g);
    // }
#endif
    // synch_barrier();
    //ot=σ(Wioxt+bio+Whoh(t−1)+bho)
    // if ( rt_core_id()<NR_CORES )
    // {
      TwoLinearLayersAccumulate (
        // Layer Attributes
        inFeaturesSize, hiddenFeaturesSize, hiddenFeaturesSize,ACT_SIG, 
        // Layer Parameters
        weight_ih_l+3*inFeaturesSize*hiddenFeaturesSize, // weight1
        weight_hh_l+3*hiddenFeaturesSize*hiddenFeaturesSize, // weight2
        bias_ih_l+3*hiddenFeaturesSize,   // bias1
        bias_hh_l+3*hiddenFeaturesSize,   // bias2 
        inFeatures+seq*inFeaturesSize,    // in1
        lstm_h,        // in2
        lstm_o);       // out
    // }
    // synch_barrier();

#ifndef DOACTONTHEFLY
    // if ( rt_core_id()==0 )
    // if ( core_id<NR_CORES )
    // {
      SigLayer(hiddenFeaturesSize, lstm_o);
    // }
#endif

#ifdef DEBUG_LSTM
    // if ( core_id<NR_CORES )
    // {
      printf("lstm_o: "); PrintTensor(hiddenFeaturesSize, lstm_o);
    // }
#endif
    }
    synch_barrier();
    //ct=ft*c(t−1)+it*gt
    // if ( rt_core_id()==0 )
    if ( core_id<NR_CORES )
    {
      HadMulTensor(hiddenFeaturesSize, lstm_c, lstm_f);
      HadMulTensor(hiddenFeaturesSize, lstm_i, lstm_g);
      AddTensor(hiddenFeaturesSize, lstm_c, lstm_i);
    }

#ifdef DEBUG_LSTM
    if ( core_id<NR_CORES )
    {
      printf("lstm_c: "); PrintTensor(hiddenFeaturesSize, lstm_c);
    }
#endif

    //ht=ottanh(ct)
    if ( rt_core_id()==0 )
    {
      CopyTensor(hiddenFeaturesSize, lstm_h_out, lstm_c); // h=c
    }
    synch_barrier();

    if(core_id<NR_CORES)
    {
      // CopyTensor(hiddenFeaturesSize, lstm_h, lstm_c); // h=c
      // TanhLayer(hiddenFeaturesSize, lstm_h); // tanh(c_t)
      // HadMulTensor(hiddenFeaturesSize, lstm_h, lstm_o);
      
// #ifdef DEBUG_LSTM
//     if ( core_id<NR_CORES )
//     {
//       printf("lstm_h: "); PrintTensor(hiddenFeaturesSize, lstm_h_out);
//     }
// #endif
      TanhLayer(hiddenFeaturesSize, lstm_h_out); // tanh(c_t)
// #ifdef DEBUG_LSTM
//     if ( core_id<NR_CORES )
//     {
//       printf("lstm_h: "); PrintTensor(hiddenFeaturesSize, lstm_h_out);
//     }
// #endif
      HadMulTensor(hiddenFeaturesSize, lstm_h_out, lstm_o);
    }
    // synch_barrier();

#ifdef DEBUG_LSTM
    if ( core_id<NR_CORES )
    {
      printf("lstm_h_out: "); PrintTensor(hiddenFeaturesSize, lstm_h_out);
    }
#endif

  }

  if ( core_id<NR_CORES )
  {
    PROFILING_LSTM_END
  }

}

#endif

#else //LSTM_OPT

//////////////////////////////////////////////////////////////////////////////////////////////
/** @brief Calculates an LSTM layer
 *  @param inFeaturesSize Number of input neurons
 *  @param hiddenFeaturesSize Number of hidden neurons
 *  @param weight_ih_l Weights mapping input neurons to hidden neurons
 *  @param weight_hh_l Weights mapping hidden neurons to hidden neurons
 *  @param bias_ih_l Bias mapping input neurons to hidden neurons
 *  @param bias_hh_l Bias mapping hidden neurons to hidden neurons
 *  @param lstm_h hidden state tensor
 *  @param lstm_c cell state tensor
 *  @param lstm_f forget gate activation tensor
 *  @param inFeatures input feature map
 *  @param lstm_i input/update gate activation tensor
 *  @param lstm_g g tensor 
 *  @param lstm_o output gate tensor
 */
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
  data_t * __restrict__ lstm_h,
  // Hidden Features
  data_t * __restrict__ lstm_c,
  // intermediate nodes
  data_t * __restrict__ lstm_h_out,
  data_t * __restrict__ lstm_f,
  data_t * __restrict__ lstm_i,
  data_t * __restrict__ lstm_g,
  data_t * __restrict__ lstm_o
)
{

  PROFILING_LSTM_START

  int core_id = rt_core_id();

#ifdef DEBUG_LSTM
  if ( core_id<NR_CORES )
  {
    printf("lstm_in: "); PrintTensor(inFeaturesSize, inFeatures);
  }
#endif
  
  for(int seq=0; seq<lstm_seqSize; seq++) {

    //it=σ(Wiixt+bii+Whih(t−1)+bhi)
    if ( core_id<NR_CORES )
    {
      TwoLinearLayersAccumulate (
        // Layer Attributes
        inFeaturesSize, hiddenFeaturesSize, hiddenFeaturesSize, ACT_SIG, 
        // Layer Parameters
        weight_ih_l+0*inFeaturesSize*hiddenFeaturesSize, // weight1
        weight_hh_l+0*hiddenFeaturesSize*hiddenFeaturesSize, // weight2
        bias_ih_l+0*hiddenFeaturesSize,   // bias1
        bias_hh_l+0*hiddenFeaturesSize,   // bias2 
        inFeatures+seq*inFeaturesSize,    // in1
        lstm_h,        // in2
        lstm_i);       // out
    }
    synch_barrier();

#ifdef DEBUG_LSTM
    if ( core_id<NR_CORES )
    {
      printf("lstm_i: "); PrintTensor(hiddenFeaturesSize, lstm_i);
    }
#endif

#ifndef DOACTONTHEFLY
    // if ( rt_core_id()==0 )
    if ( core_id<NR_CORES )
    {
      SigLayer(hiddenFeaturesSize, lstm_i);
    }
#endif

#ifdef DEBUG_LSTM
    if ( core_id<NR_CORES )
    {
      printf("lstm_i: "); PrintTensor(hiddenFeaturesSize, lstm_i);
    }
#endif
    synch_barrier();
    //ft=σ(Wif xt+bif+Whf h(t−1)+bhf)
    if ( rt_core_id()<NR_CORES )
    {
      TwoLinearLayersAccumulate (
        // Layer Attributes
        inFeaturesSize, hiddenFeaturesSize, hiddenFeaturesSize, ACT_SIG, 
        // Layer Parameters
        weight_ih_l+1*inFeaturesSize*hiddenFeaturesSize, // weight1
        weight_hh_l+1*hiddenFeaturesSize*hiddenFeaturesSize, // weight2
        bias_ih_l+1*hiddenFeaturesSize,   // bias1
        bias_hh_l+1*hiddenFeaturesSize,   // bias2 
        inFeatures+seq*inFeaturesSize,    // in1
        lstm_h,        // in2
        lstm_f);       // out
    }
    synch_barrier();

#ifdef DEBUG_LSTM
    if ( core_id<NR_CORES )
    {
      printf("lstm_f: "); PrintTensor(hiddenFeaturesSize, lstm_f);
    }
#endif

#ifndef DOACTONTHEFLY
    // if ( rt_core_id()==0 )
    if ( core_id<NR_CORES )
    {
      SigLayer(hiddenFeaturesSize, lstm_f);
    }
#endif
    
#ifdef DEBUG_LSTM
    if ( core_id<NR_CORES )
    {
      printf("lstm_f: "); PrintTensor(hiddenFeaturesSize, lstm_f);
    }
#endif
    synch_barrier();
    //gt=tanh(Wigxt+big+Whgh(t−1)+bhg)
    if ( rt_core_id()<NR_CORES )
    {
      TwoLinearLayersAccumulate (
        // Layer Attributes
        inFeaturesSize, hiddenFeaturesSize, hiddenFeaturesSize, ACT_TANH, 
        // Layer Parameters
        weight_ih_l+2*inFeaturesSize*hiddenFeaturesSize, // weight1
        weight_hh_l+2*hiddenFeaturesSize*hiddenFeaturesSize, // weight2
        bias_ih_l+2*hiddenFeaturesSize,   // bias1
        bias_hh_l+2*hiddenFeaturesSize,   // bias2 
        inFeatures+seq*inFeaturesSize,    // in1
        lstm_h,        // in2
        lstm_g);       // out
    }
    synch_barrier();

#ifdef DEBUG_LSTM
    if ( core_id<NR_CORES )
    {
      printf("lstm_g: "); PrintTensor(hiddenFeaturesSize, lstm_g);
    }
#endif

#ifndef DOACTONTHEFLY
    // if ( rt_core_id()==0 )
    if ( core_id<NR_CORES )
    {
      TanhLayer(hiddenFeaturesSize, lstm_g);
    }
#endif

#ifdef DEBUG_LSTM
    if ( core_id<NR_CORES )
    {
      printf("lstm_g: "); PrintTensor(hiddenFeaturesSize, lstm_g);
    }
#endif
    synch_barrier();
    //ot=σ(Wioxt+bio+Whoh(t−1)+bho)
    if ( rt_core_id()<NR_CORES )
    {
      TwoLinearLayersAccumulate (
        // Layer Attributes
        inFeaturesSize, hiddenFeaturesSize, hiddenFeaturesSize,ACT_SIG, 
        // Layer Parameters
        weight_ih_l+3*inFeaturesSize*hiddenFeaturesSize, // weight1
        weight_hh_l+3*hiddenFeaturesSize*hiddenFeaturesSize, // weight2
        bias_ih_l+3*hiddenFeaturesSize,   // bias1
        bias_hh_l+3*hiddenFeaturesSize,   // bias2 
        inFeatures+seq*inFeaturesSize,    // in1
        lstm_h,        // in2
        lstm_o);       // out
    }
    synch_barrier();

#ifndef DOACTONTHEFLY
    // if ( rt_core_id()==0 )
    if ( core_id<NR_CORES )
    {
      SigLayer(hiddenFeaturesSize, lstm_o);
    }
#endif

#ifdef DEBUG_LSTM
    if ( core_id<NR_CORES )
    {
      printf("lstm_o: "); PrintTensor(hiddenFeaturesSize, lstm_o);
    }
#endif
    synch_barrier();
    //ct=ft*c(t−1)+it*gt
    // if ( rt_core_id()==0 )
    if ( core_id<NR_CORES )
    {
      HadMulTensor(hiddenFeaturesSize, lstm_c, lstm_f);
      HadMulTensor(hiddenFeaturesSize, lstm_i, lstm_g);
      AddTensor(hiddenFeaturesSize, lstm_c, lstm_i);
    }

#ifdef DEBUG_LSTM
    if ( core_id<NR_CORES )
    {
      printf("lstm_c: "); PrintTensor(hiddenFeaturesSize, lstm_c);
    }
#endif

    //ht=ottanh(ct)
    if ( rt_core_id()==0 )
    {
      CopyTensor(hiddenFeaturesSize, lstm_h_out, lstm_c); // h=c
    }
    synch_barrier();

    if(core_id<NR_CORES)
    {
      // CopyTensor(hiddenFeaturesSize, lstm_h, lstm_c); // h=c
      // TanhLayer(hiddenFeaturesSize, lstm_h); // tanh(c_t)
      // HadMulTensor(hiddenFeaturesSize, lstm_h, lstm_o);
      
// #ifdef DEBUG_LSTM
//     if ( core_id<NR_CORES )
//     {
//       printf("lstm_h: "); PrintTensor(hiddenFeaturesSize, lstm_h_out);
//     }
// #endif
      TanhLayer(hiddenFeaturesSize, lstm_h_out); // tanh(c_t)
// #ifdef DEBUG_LSTM
//     if ( core_id<NR_CORES )
//     {
//       printf("lstm_h: "); PrintTensor(hiddenFeaturesSize, lstm_h_out);
//     }
// #endif
      HadMulTensor(hiddenFeaturesSize, lstm_h_out, lstm_o);
    }
    synch_barrier();

#ifdef DEBUG_LSTM
    if ( core_id<NR_CORES )
    {
      printf("lstm_h_out: "); PrintTensor(hiddenFeaturesSize, lstm_h_out);
    }
#endif

  }

  PROFILING_LSTM_END

}

#endif //LSTM_OPT


//////////////////////////////////////////////////////////////////////////////////////////////
/** @brief Print 2D Tensor
 *  @param dim1 x dimension
 *  @param dim2 y dimension
 *  @param dataArray data to be printed
 */
void PrintTensor2D (
        // Layer Attributes
  int dim1, int dim2,
  data_t * __restrict__ dataArray
  )
{
  // for 1d array -> set dim2 = 1
  for (int o=0; o< dim2; o++) 
  {
    printf("[");
    for(int i=0; i<dim1; i++)
    {
      // int temp = (int)(dataArray[dim2*o+i]*1000)%1000;

      printFloat(dataArray[dim2*o+i]);
      printf(", ");
    }
    printf("], ");
  }
  printf("\n");
}


//////////////////////////////////////////////////////////////////////////////////////////////
/** @brief Print 1D Tensor
 *  @param dim1 length of tensor
 *  @param dataArray data to be printed
 */
void PrintTensor (
  // Layer Attributes
  int dim1,
  data_t * __restrict__ dataArray
  )
{
  PrintTensor2D(dim1, 1, dataArray);
}


//////////////////////////////////////////////////////////////////////////////////////////////
/** @brief Print difference of 1D Tensor
 *  @param dim1 length
 *  @param dataArray data to be printed
 *  @param data2Array data to be printed
 */
data_t PrintTensorDiff (
        // Layer Attributes
  int dim1,
  data_t * __restrict__ dataArray,
  data_t * __restrict__ data2Array
  )
{
 return PrintTensorDiff2D(dim1, 1, dataArray, data2Array);
}


//////////////////////////////////////////////////////////////////////////////////////////////
/** @brief Print difference of 2D Tensor
 *  @param dim1 x dimension
 *  @param dim2 y dimension
 *  @param dataArray data to be printed
 *  @param data2Array data to be printed
 */
data_t PrintTensorDiff2D (
        // Layer Attributes
  int dim1, int dim2,
  data_t * __restrict__ dataArray,
  data_t * __restrict__ data2Array
  )
{
   // for 1d array -> set dim2 = 1
  int temp_sum = 0;
  for (int o=0; o< dim2; o++) 
  {
    printf("[");
    for(int i=0; i<dim1; i++)
    {
     data_t temp = dataArray[dim2*o+i]-data2Array[dim2*o+i];
     printFloat(temp);
     temp_sum += (int)temp*(int)temp;
     printf(", ");
   }
   printf("], ");
 }
 printf("\n");
 temp_sum /= (dim1*dim2);
 printf("mse= %d", temp_sum);
 return (data_t)temp_sum;
}


//////////////////////////////////////////////////////////////////////////////////////////////
/** @brief Calculates average quadratic error
 *  @param dim1 x dimension
 *  @param dim2 y dimension
 *  @param dataArray data to be printed
 *  @param data2Array data to be printed
 *  @param error pointer to store error
 */
void error2D (
        // Layer Attributes
  int dim1, int dim2,
  data_t * __restrict__ dataArray,
  data_t * __restrict__ data2Array,
  data_t * __restrict__ error
  )
{
   // for 1d array -> set dim2 = 1
 data_t temp = 0;
 for (int o=0; o< dim2; o++) 
 {

  for(int i=0; i<dim1; i++)
  {
   data_t temp2 = dataArray[dim2*o+i]-data2Array[dim2*o+i];
   temp += temp2*temp2;
 }

}
(*error) =  temp/dim2/dim1;
}


//////////////////////////////////////////////////////////////////////////////////////////////
/** @brief Prints a float value (used on RISC-Y without float unit) (currently deactivated)
 *  @param value 
 */
void printFloat(data_t value) {
// #ifndef ASIP
// #ifdef FixedPt
//     // printf("%i", value);

  // float tmp = ((float)value)/(1<<(q_frac));

  // if(tmp > -1.0f && tmp < 0.0f) {
  //   printf("-%i.%03i",  (int)(tmp), (int)Abs(tmp*1000)%1000);
  // }
  // else
  // {
  //   printf("%i.%03i",  (int)(tmp), (int)Abs(tmp*1000)%1000);
  // }
// #else
//   if(value > -1.0f && value < 0.0f) {
//     printf("-%i.%03i",  (int)(value), (int)Abs(value*1000)%1000);
//     // printf("%1.3f",  (value));
//   }
//   else
//   {
//     printf("%i.%03i",  (int)(value), (int)Abs(value*1000)%1000);
//     // printf("%1.3f",  (value));
//   }
// #endif
// #else
  printf("%i",  (int)(value));
// #endif
}


//////////////////////////////////////////////////////////////////////////////////////////////
/** @brief Taylor Extension of the e^x function
 *
 *  @param n number of taylor coefficients
 *  @param x input value
 *  @return approximate value of e^x
 */
inline float  ALWAYS_INLINE  expTailor(int n, float x) 
{ 
  float sum = 1.0f;

  for (int i = n - 1; i > 0; --i ) 
    sum = 1 + x * sum / i; 

  return sum; 
} 

//////////////////////////////////////////////////////////////////////////////////////////////
/** @brief Signum Function
 *
 *  @param value
 *  @return sgn(value)
 */
inline data_t ALWAYS_INLINE  Sgn(data_t value) { // zero is positive!
  return (value >= (data_t)0.0)?+1:-1;
}

//////////////////////////////////////////////////////////////////////////////////////////////
#ifdef ASIP
inline data_t Min(data_t a, data_t b) {return (((a)<(b))?(a):(b));}
inline data_t Max(data_t a, data_t b) {return (((a)>(b))?(a):(b));}
inline data_t Abs(data_t a)           {return (((a)>(0.0))?(a):(-a));}
inline float  Min(float  a, float  b) {return (((a)<(b))?(a):(b));}
inline float  Max(float  a, float  b) {return (((a)>(b))?(a):(b));}
inline float  Abs(float  a)           {return (((a)>(0.0))?(a):(-a));}
inline int  Min(int  a, int  b) {return (((a)<(b))?(a):(b));}
inline int  Max(int  a, int  b) {return (((a)>(b))?(a):(b));}
inline int  Abs(int  a)           {return (((a)>(0.0))?(a):(-a));}

#else // ASIP

 /** @brief Intrinsic for the PULP tanh extension
 *  @param tanh_value
 */
inline int pulpRNNExt_tanh(int tanh_value) {
  int tmp;
  asm volatile("pl.tanh %0, %1" : "=r" (tmp) : "r" (tanh_value) );
  return tmp;
}

 /** @brief Intrinsic for the PULP sigmoid extension
 *  @param sig_value
 */
inline int pulpRNNExt_sig(int sig_value) {
  int tmp;
  asm volatile("pl.sig %0, %1" : "=r" (tmp) : "r" (sig_value) );
  return tmp;
}

#endif // ASIP


//////////////////////////////////////////////////////////////////////////////////////////////
#ifdef PULP_USETANHSIG

/// Select tanh function to be used
 inline data_t generic_tanh(data_t value) {return pulpRNNExt_tanh(value);}
/// Select sigmoid function to be used
 inline data_t generic_sig(data_t value) {return pulpRNNExt_sig(value);}

//////////////////////////////////////////////////////////////////////////////////////////////
#elif defined MATHH

/** @brief Tangens Hyperbolicus Activation Function
 *
 *  @param value input varialbe
 *  @return tanh of the input variable
 */
inline data_t generic_tanh(data_t value) {return (data_t)(tanh(((float)value)/(1<<q_frac))*(1<<q_frac));}


/** @brief Generic Sigmoid Activation Function
 *
 *  @param x input varialbe
 *  @return sigmoid of the input variable
 */
inline data_t generic_sig(data_t x)
{
     float x_float = (float)(x)/(1<<q_frac);
     float exp_value;
     data_t return_value;
     exp_value = exp((double) -x);
     return_value = (data_t)((1 / (1 + exp_value))*(1<<q_frac));
     return return_value;
}

//////////////////////////////////////////////////////////////////////////////////////////////
#else 

/** @brief Tangens Hyperbolicus  Activation Function
 *
 *  @param value input varialbe
 *  @return tanh of the input variable
 */
inline data_t generic_tanh(data_t value) {return Tanh(value);}

/** @brief Generic Sigmoid Activation Function
 *
 *  @param value input varialbe
 *  @return sigmoid of the input variable
 */
inline data_t generic_sig(data_t value) {return sig(value);}

#endif
//////////////////////////////////////////////////////////////////////////////////////////////
