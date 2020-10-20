/*  @file basicKernel.c
 *  @brief C Implementation for basic ML kernels (including FC, LSTM, Conv2D for the RNN ASIP
 * 
 *  C implemenentation implementing several levels of opitmizations for the RISC-Y extension and the tzscale extension
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

#ifndef INCLHEADER
#define INCLHEADER
#include <stdio.h>
// #include <stdlib.h>
#include <config.h>
//#include <math.h>
#include "basicKernel.h"
// #include "lut.h" // coefficients for taylor expansion
// #include "benchmarks.h"
#endif


#ifdef MULTICORE

/** @brief buffer of size BUFFER_LIN_B1_SIZE for first bias*/
__attribute__ ((section(".heapsram"))) data_t linear_Bias     [BUFFER_LIN_B1_SIZE];
/** @brief buffer of size BUFFER_LIN_W1_SIZE for first weights*/
__attribute__ ((section(".heapsram"))) data_t linear_Weights  [BUFFER_LIN_W1_SIZE];
/** @brief buffer of size BUFFER_LIN_B2_SIZE for second bias*/
__attribute__ ((section(".heapsram"))) data_t linear_Bias2    [BUFFER_LIN_B2_SIZE];
/** @brief buffer of size BUFFER_LIN_W1_SIZE for first weights*/
__attribute__ ((section(".heapsram"))) data_t linear_Weights2 [BUFFER_LIN_W2_SIZE];
/** @brief buffer of size BUFFER_LIN_C_SIZE for internal state*/
__attribute__ ((section(".heapsram"))) data_t linear_C        [BUFFER_LIN_C_SIZE];
/** @brief buffer of size BUFFER_LIN_H_SIZE for hidden state*/
__attribute__ ((section(".heapsram"))) data_t linear_H        [BUFFER_LIN_H_SIZE];

/** @brief buffer of size MAX_NR_TRANSACTIONS for collecting running DMA transactions*/
__attribute__ ((section(".heapsram"))) int dma_trans_ids [MAX_NR_TRANSACTIONS];

#endif



#ifndef ASIP


/** @brief Tells the PULP-SDK to initialize or reset performance counters
 */
void startPerf () {

#ifdef PROFILING

    rt_perf_reset(&perf);
    numFunctionCalls++;
    rt_perf_start(&perf);

#elif defined(PROFILING_NEW)

#ifdef TIMER
    if(perf[rt_core_id()].perf_counter_id==CSR_PCER_NB_EVENTS)
    {
      TIMER_START;
    }
    else
    {
#endif // TIMER
    numFunctionCalls++;
    perf_reset();
    perf_enable_id(perf[rt_core_id()].perf_counter_id); // start correct counter
    // perf_start(); // starts all counters, but should only start one
#ifdef TIMER
    }
#endif // TIMER

#endif // PROFILING | PROFILING_NEW
}


/** @brief Tells the PULP-SDK to stop the performance counters
 */
void endPerf () {

#ifdef PROFILING

    rt_perf_stop(&perf);

#elif defined(PROFILING_NEW)

#ifdef TIMER
    if(perf[rt_core_id()].perf_counter_id==CSR_PCER_NB_EVENTS)
    {
      TIMER_END;
    }
    else
    {
#endif // TIMER

    perf_stop();
    perf[rt_core_id()].perf_counters[perf[rt_core_id()].perf_counter_id] += cpu_perf_get(perf[rt_core_id()].perf_counter_id);

#ifdef TIMER
    }
#endif // TIMER

#endif // PROFILING | PROFILING_NEW
}


/** @brief Tells the PULP-SDK to initialize or reset performance timer
 */
void startTimer () {
#ifdef TIMER

    if ( rt_core_id()==0 )
    {
#ifdef MULTICORE
      timer_reset(timer_base_cl(0, 0, 1));
      timer_start(timer_base_cl(0, 0, 1));
#else
      timer_reset(timer_base_fc(0, 1));
      timer_start(timer_base_fc(0, 1));
      // reset_timer();
      // start_timer();
#endif // MULTICORE
    }
#endif // TIMER
}


/** @brief Tells the PULP-SDK to stop the performance timer
 */
void endTimer () {
#ifdef TIMER

    if ( rt_core_id()==0 )
    {
#ifdef MULTICORE
      timer_cl += timer_count_get(timer_base_cl(0, 0, 1));
      timer_conf_set(timer_base_cl(0, 0, 1), 0);
#else
      timer_cl += timer_count_get(timer_base_fc(0, 1));
      timer_conf_set(timer_base_fc(0, 1), 0);
      // timer_cl += get_time();
      // stop_timer();
#endif // MULTICORE
    }
#endif // TIMER
}

#endif // ifndef ASIP




/** @brief Runs a neural network
 *
 *  Iterates through all the layers while passing the intermediate FM with a double 
 *  buffering approach
 *
 *  @param network Array of concecutive layers of the current neural network
 *  @param depth Number of Layers (aka array size)
 *  @param inFeatures Input Feature Map
 *  @param buffer Buffer to store intermediate results
 *  @return Output Feature Map
 */
data_t * NOINLINE inferNetwork(
    struct layer * network,
    int depth,
    data_t * __restrict__ inFeatures,
    data_t * __restrict__ buffer)
{
    // printf("delete, just for test1, core: %d \n", rt_core_id());

  data_t * in;

  int core_id = rt_core_id();

#ifdef MULTICORE
  in  = &buffer[0];
#else
  in  = inFeatures;
#endif

  data_t * out = &buffer[BUFFER_SIZE2];
  _Bool first_layer = True;

#ifdef MULTICORE
  data_t * W1;
  data_t * B1;
  data_t * W1_next;
  data_t * B1_next;

  W1 = &linear_Weights[0];
  B1 = &linear_Bias[0];
  W1_next = &linear_Weights[BUFFER_LIN_W1_SIZE2];
  B1_next = &linear_Bias[BUFFER_LIN_B1_SIZE2];

  data_t * W2;
  data_t * B2;

  data_t * W2_next;
  data_t * B2_next;

  W2 = &linear_Weights2[0];
  B2 = &linear_Bias2[0];

  W2_next = &linear_Weights2[BUFFER_LIN_W2_SIZE2];
  B2_next = &linear_Bias2[BUFFER_LIN_B2_SIZE2];
#endif

#ifdef TIMER
  timer_cl = 0;
#endif

/*****************************************************************************
 *
 * Copy Input data (to be copied once)
 *
 *****************************************************************************/

#ifdef MULTICORE

  struct layer lay = network[0];
  unsigned short act_size;

  if (core_id==0)
  {

    if(lay.type == LINEAR)
    {
      act_size = (unsigned short) 2*lay.attributes[LAY_LIN_IN];
    }
    else if(lay.type == LSTM)
    {
      act_size = (unsigned short) 2*lay.attributes[LAY_LSTM_IN];
    }
    else
    {
      printf("\033[91mERROR - only Lin Layer or LSTM are supported!!!\033[0m\n");
    }
 
#ifdef DMA
#ifdef BATCHING
    for(int b=0; b<BATCHING; b++)
    {
      // printf("batching b %d %d %d\n", b, b*act_size/4, (((v2s*)in+b*act_size)));
      plp_dma_wait(plp_dma_memcpy((uint32_t) (((v2s*)inFeatures)), (uint32_t) (((v2s*)in+b*act_size/4)), act_size,  1));
    }
    // printf("input in: ");
    // PrintTensor(BATCHING*lay.attributes[LAY_LIN_IN], in);
#else
    plp_dma_wait(plp_dma_memcpy((uint32_t) (((v2s*)inFeatures)), (uint32_t) (((v2s*)in)), act_size,  1));
#endif
#else // no DMA

    for(int j = 0; j < act_size/2; j++)
    {
      in[j] = inFeatures[j];
    }

 #endif // DMA



/*****************************************************************************
 *
 * Copy Weight Data of first layer
 *
 *****************************************************************************/

    // printf("INFO - copy weigt data of first layer!!! \n");
    //////////////////////////////////////////////////////////////////////////////////////////////
    // LINEAR LAYER
    //////////////////////////////////////////////////////////////////////////////////////////////
    if(lay.type == LINEAR)
    {
      act_size = (unsigned short) 2*lay.attributes[LAY_LIN_IN];
    
  #ifdef DMA
      unsigned short b_size = lay.attributes[LAY_LIN_OUT]*2;
      unsigned w_size = (b_size)*(act_size);

      // copy BIAS
      plp_dma_wait(plp_dma_memcpy((uint32_t) (((v2s*)(lay.parameters[LAY_LIN_BIAS]))), (uint32_t) (((v2s*)B1)), b_size, 1));

      if (w_size >= 65532)
      {
        // printf("\033[91mWARNING - DATA too big for DMA!!!\033[0m\n");
        // printf("dma weight size: %d %d %d %x \n", w_size, b_size, act_size, (uint32_t) ((v2s*)W1));
        int nr_dma_tiles                          = w_size / 65532;
        int dima_tiles_overflow                   = w_size % 65532;
        if (dima_tiles_overflow!=0) nr_dma_tiles +=1;

        unsigned short curr_w_size = 0;
        int curr_w_idx = 0;
        int curr_w_idx_local = 0;

        // printf("nr_dma_tiles %d dima_tiles_overflow %d  \n", nr_dma_tiles, dima_tiles_overflow);

        for(int d=0; d<nr_dma_tiles; d++)
        {
          if(d==nr_dma_tiles-1)
          {
            curr_w_size = dima_tiles_overflow;
            curr_w_idx = (d*65532)/2;
            curr_w_idx_local = (curr_w_idx+1)/2;
          }
          else
          {
            curr_w_size = 65532;
            curr_w_idx = (d*65532)/2;
            curr_w_idx_local = ((curr_w_idx+1)/2);
          }

          plp_dma_wait(plp_dma_memcpy((uint32_t) (((v2s*)(lay.parameters[LAY_LIN_WEIGHTS]+curr_w_idx))), (uint32_t) (((v2s*)W1+curr_w_idx_local)), curr_w_size,  1));

        }
      }
      else
      {
        plp_dma_wait(plp_dma_memcpy((uint32_t) (((v2s*)(lay.parameters[LAY_LIN_WEIGHTS]))), (uint32_t) (((v2s*)W1)), w_size,  1));
      }

  #else // no DMA

      for(int m = 0; m < lay.attributes[LAY_LIN_OUT]; m++)
      {
        B1[m] = lay.parameters[LAY_LIN_BIAS][m];
        for(int n = 0; n < lay.attributes[LAY_LIN_IN] + W_OFFSET; n++) //+2
        {
          W1[m*(lay.attributes[LAY_LIN_IN]+W_OFFSET)+n] = lay.parameters[LAY_LIN_WEIGHTS][m*(lay.attributes[LAY_LIN_IN])+n];
        }
      }

  #endif //  DMA

    }
    //////////////////////////////////////////////////////////////////////////////////////////////
    // LSTM
    //////////////////////////////////////////////////////////////////////////////////////////////
    else if(lay.type == LSTM)
    {
        act_size = (unsigned short) 2*lay.attributes[LAY_LSTM_IN];

#ifdef DMA
          unsigned short hidden_4_size = 2*lay.attributes[LAY_LSTM_HID];
          plp_dma_wait(plp_dma_memcpy((uint32_t) (((v2s*)lay.parameters[LSTM_H])), (uint32_t) (((v2s*)linear_H)), hidden_4_size,  1));
          plp_dma_wait(plp_dma_memcpy((uint32_t) (((v2s*)lay.parameters[LSTM_C])), (uint32_t) (((v2s*)linear_C)), hidden_4_size,  1));
#else // DMA
          for(int j = 0; j < lay.attributes[LAY_LSTM_HID]; j++)
          {
            linear_H[j] = lay.parameters[LSTM_H][j];
            linear_C[j] = lay.parameters[LSTM_C][j];
          }
#endif // DMA

  #ifdef DMA
      unsigned short b_size = 2*4*lay.attributes[LAY_LSTM_HID];
      unsigned w_size = (b_size)*(act_size);
      unsigned w2_size = (b_size)*2*lay.attributes[LAY_LSTM_HID];

      // copy BIASES
      plp_dma_wait(plp_dma_memcpy((uint32_t) (((v2s*)(lay.parameters[LSTM_BIAS_IH]))), (uint32_t) (((v2s*)B1)), b_size, 1));
      plp_dma_wait(plp_dma_memcpy((uint32_t) (((v2s*)(lay.parameters[LSTM_BIAS_HH]))), (uint32_t) (((v2s*)B2)), b_size, 1));


      // printf("size %u \n", w_size);
      // Copy W1
      if (w_size < 65532)
      {

        plp_dma_wait(plp_dma_memcpy((uint32_t) (((v2s*)(lay.parameters[LSTM_WGHT_IH]))), (uint32_t) (((v2s*)W1)), w_size, 1)); //b_size*2*lay.attributes[LAY_LSTM_HID],  1));
      }
      else
      {

        int nr_dma_tiles           = w_size / 65532;
        int dima_tiles_overflow    = w_size % 65532;
        if (dima_tiles_overflow!=0) nr_dma_tiles+=1;
        unsigned short curr_w_size = 0;
        int curr_w_idx = 0;
        int curr_w_idx_local = 0;

        // printf("nr_dma_tiles %d dima_tiles_overflow %d  \n", nr_dma_tiles, dima_tiles_overflow);

        for(int d=0; d<nr_dma_tiles; d++)
        {
          if(d==nr_dma_tiles-1)
          {
            curr_w_size = dima_tiles_overflow;
            curr_w_idx = (d*65532)/2;
            curr_w_idx_local = (curr_w_idx+1)/2;
          }
          else
          {
            curr_w_size = 65532;
            curr_w_idx = (d*65532)/2;
            curr_w_idx_local = ((curr_w_idx+1)/2);
          }
          plp_dma_wait(plp_dma_memcpy((uint32_t) (((v2s*)(lay.parameters[LSTM_WGHT_IH]+curr_w_idx))), (uint32_t) (((v2s*)W1+curr_w_idx_local)), curr_w_size,  1));
        }
      }
      

      // Copy W2
      if (w2_size < 65532)
      {
        plp_dma_wait(plp_dma_memcpy((uint32_t) (((v2s*)(lay.parameters[LSTM_WGHT_HH]))), (uint32_t) (((v2s*)W2)), w2_size, 1)); //b_size*2*lay.attributes[LAY_LSTM_HID],  1));
      }
      else
      {
        // printf("\033[91mWARNING - LSTM W2 DATA too big for DMA!!!\033[0m\n");

        int nr_dma_tiles           = w2_size / 65532;
        int dima_tiles_overflow    = w2_size % 65532;
        if (dima_tiles_overflow!=0) nr_dma_tiles+=1;
        unsigned short curr_w_size = 0;
        int curr_w_idx = 0;
        int curr_w_idx_local = 0;

        // printf("nr_dma_tiles %d dima_tiles_overflow %d  \n", nr_dma_tiles, dima_tiles_overflow);

        for(int d=0; d<nr_dma_tiles; d++)
        {
          if(d==nr_dma_tiles-1)
          {
            curr_w_size = dima_tiles_overflow;
            curr_w_idx = (d*65532)/2;
            curr_w_idx_local = (curr_w_idx+1)/2;
          }
          else
          {
            curr_w_size = 65532;
            curr_w_idx = (d*65532)/2;
            curr_w_idx_local = ((curr_w_idx+1)/2);
          }
          // printf("dma_tile %d curr_w_size %d  curr_w_idx %d \n", d, curr_w_size, curr_w_idx);
          plp_dma_wait(plp_dma_memcpy((uint32_t) (((v2s*)(lay.parameters[LSTM_WGHT_HH]+curr_w_idx))), (uint32_t) (((v2s*)W2+curr_w_idx_local)), curr_w_size,  1));
        }
      }

  #else // no DMA

      for(int m = 0; m < 4*lay.attributes[LAY_LSTM_HID]; m++)
      {
        B1[m] = lay.parameters[LSTM_BIAS_IH][m];
        B2[m] = lay.parameters[LSTM_BIAS_HH][m];

        for(int n = 0; n < lay.attributes[LAY_LSTM_IN] + W_OFFSET; n++) //+2
        {
          W1[m*(lay.attributes[LAY_LSTM_IN]+W_OFFSET)+n] = lay.parameters[LSTM_WGHT_IH][m*(lay.attributes[LAY_LSTM_IN])+n];
        }
        for(int n = 0; n < lay.attributes[LAY_LSTM_HID] + W_OFFSET; n++) //+2
        {
          W2[m*(lay.attributes[LAY_LSTM_HID]+W_OFFSET)+n] = lay.parameters[LSTM_WGHT_HH][m*(lay.attributes[LAY_LSTM_HID])+n];
        }
      }
  #endif // DMA
  
    }
    //////////////////////////////////////////////////////////////////////////////////////////////
    // CONVOLUTION
    //////////////////////////////////////////////////////////////////////////////////////////////
    else
    {
      printf("\033[91mERROR - only Lin Layer or LSTM are supported!!!\033[0m\n");
    }
  }

  synch_barrier(); // TODO: needed???
          
#endif // MULTICORE

  synch_barrier();
  PROFILING_ALL_START

/*****************************************************************************
 *
 * Current Layer Execution
 *
 *****************************************************************************/
  short toFIRST = False;
  for(int i = 0; i < depth; i++)
  {

#ifdef MULTICORE
    int dma_idx = 0;

    // switch buffers
    if(toFIRST) 
    {
      W1 = &linear_Weights[BUFFER_LIN_W1_SIZE2];
      B1 = &linear_Bias[BUFFER_LIN_B1_SIZE2];
      W2 = &linear_Weights2[BUFFER_LIN_W2_SIZE2];
      B2 = &linear_Bias2[BUFFER_LIN_B2_SIZE2];

      W1_next = &linear_Weights[0];
      B1_next = &linear_Bias[0];
      W2_next = &linear_Weights2[0];
      B2_next = &linear_Bias2[0];

    }
    else 
    {
      W1 = &linear_Weights[0];
      B1 = &linear_Bias[0];
      W2 = &linear_Weights2[0];
      B2 = &linear_Bias2[0];

      W1_next = &linear_Weights[BUFFER_LIN_W1_SIZE2];
      B1_next = &linear_Bias[BUFFER_LIN_B1_SIZE2];
      W2_next = &linear_Weights2[BUFFER_LIN_W2_SIZE2];
      B2_next = &linear_Bias2[BUFFER_LIN_B2_SIZE2];

    }
#endif //MULTICORE

    // printf("INFO - Layer %d %d core %d\n", i, toFIRST, rt_core_id());
    struct layer lay = network[i];
    struct layer lay_next;

    if(i+1<depth)
    {

 /*****************************************************************************
 *
 * Start Copy Weight Data of next layer
 *
 *****************************************************************************/
    #ifdef MULTICORE
      if ( core_id==0 )
      {
        lay_next = network[i+1];

        // printf("INFO - copy weigt data of next layer!!! \n");
        //////////////////////////////////////////////////////////////////////////////////////////////
        // LINEAR
        //////////////////////////////////////////////////////////////////////////////////////////////
        if(lay_next.type == LINEAR)
        {
          // printf("INFO - copy weigt data of next layer!!! LINEAR asd\n");

          act_size = (unsigned short) 2*lay_next.attributes[LAY_LIN_IN];

      #ifdef DMA
          unsigned short b_size = lay_next.attributes[LAY_LIN_OUT]*2;
          unsigned w_size = (b_size)*(act_size);

          // Copy Bias
          dma_trans_ids[dma_idx] = plp_dma_memcpy((uint32_t) (((v2s*)(lay_next.parameters[LAY_LIN_BIAS]))), (uint32_t) (((v2s*)B1_next)), b_size, 1);
          dma_idx += 1;

          if (w_size >= 65532)
          {
            // printf("\033[91mWARNING - DATA too big for DMA 2 !!!\033[0m\n");

            // printf("dma bias size: %d \n", b_size);
            // printf("dma weight size: %d %d %d %x \n", w_size, b_size, act_size, (uint32_t) ((v2s*)W1_next));
            int nr_dma_tiles                          = w_size / 65532;
            int dima_tiles_overflow                   = w_size % 65532;
            if (dima_tiles_overflow!=0) nr_dma_tiles +=1;

            unsigned short curr_w_size = 0;
            int curr_w_idx = 0;
            int curr_w_idx_local = 0;

            // printf("core %d  nr_dma_tiles %d dima_tiles_overflow %d  \n", rt_core_id(), nr_dma_tiles, dima_tiles_overflow);

            for(int d=0; d<nr_dma_tiles; d++)
            // for(int d=0; d<3; d++)
            {
              if(d==nr_dma_tiles-1)
              {
                curr_w_size = dima_tiles_overflow/2;
                curr_w_idx = (d*65532)/2;
                curr_w_idx_local = (curr_w_idx+1)/2;
              }
              else
              {
                curr_w_size = 65532;
                curr_w_idx = (d*65532)/2;
                curr_w_idx_local = ((curr_w_idx+1)/2);
              }

              // printf("core %d %d value %d \n", rt_core_id(), 0, *(lay_next.parameters[LAY_LIN_WEIGHTS]+curr_w_idx) );
 
              dma_trans_ids[dma_idx] = plp_dma_memcpy((uint32_t) (((v2s*)(lay_next.parameters[LAY_LIN_WEIGHTS]+curr_w_idx))), (uint32_t) (((v2s*)W1_next+curr_w_idx_local)), curr_w_size,  1);
              dma_idx += 1;
            }
          }
          else
          {
            dma_trans_ids[dma_idx] = plp_dma_memcpy((uint32_t) (((v2s*)(lay_next.parameters[LAY_LIN_WEIGHTS]))), (uint32_t) (((v2s*)W1_next)), w_size,  1);
            dma_idx += 1;
          }
      #else // no DMA

          for(int m = 0; m < lay_next.attributes[LAY_LIN_OUT]; m++)
          {
            B1_next[m] = lay_next.parameters[LAY_LIN_BIAS][m];
            for(int n = 0; n < lay_next.attributes[LAY_LIN_IN] + W_OFFSET; n++) //+2
            {
              W1_next[m*(lay_next.attributes[LAY_LIN_IN]+W_OFFSET)+n] = lay_next.parameters[LAY_LIN_WEIGHTS][m*(lay_next.attributes[LAY_LIN_IN])+n];
            }
          }

      #endif // DMA

        }
        //////////////////////////////////////////////////////////////////////////////////////////////
        // LSTM
        //////////////////////////////////////////////////////////////////////////////////////////////
        else if(lay_next.type == LSTM)
        {
          // printf("INFO - copy weigt data of next layer!!! LSTM\n");

          act_size = (unsigned short) 2*lay_next.attributes[LAY_LSTM_IN];

#ifdef DMA
          unsigned short hidden_4_size = 2*lay_next.attributes[LAY_LSTM_HID];
          // plp_dma_wait(plp_dma_memcpy((uint32_t) (((v2s*)lay_next.parameters[LSTM_H])), (uint32_t) (((v2s*)linear_H)), hidden_4_size,  1));
          dma_trans_ids[dma_idx] = plp_dma_memcpy((uint32_t) (((v2s*)lay_next.parameters[LSTM_H])), (uint32_t) (((v2s*)linear_H)), hidden_4_size,  1);
          dma_idx += 1;
          // plp_dma_wait(plp_dma_memcpy((uint32_t) (((v2s*)lay_next.parameters[LSTM_C])), (uint32_t) (((v2s*)linear_C)), hidden_4_size,  1));
          dma_trans_ids[dma_idx] = plp_dma_memcpy((uint32_t) (((v2s*)lay_next.parameters[LSTM_C])), (uint32_t) (((v2s*)linear_C)), hidden_4_size,  1);
          dma_idx += 1;
#else
          for(int j = 0; j < lay_next.attributes[LAY_LSTM_HID]; j++)
          {
            linear_H[j] = lay_next.parameters[LSTM_H][j];
            linear_C[j] = lay_next.parameters[LSTM_C][j];
          }
#endif

      #ifdef DMA
          unsigned short b_size = 2*4*lay_next.attributes[LAY_LSTM_HID];
          unsigned w_size = (b_size)*(act_size);
          // printf("size %u \n", w_size);
          if (w_size >= 65532)
          {

            // plp_dma_wait(plp_dma_memcpy((uint32_t) (((v2s*)(lay_next.parameters[LAY_LIN_BIAS]))),    (uint32_t) (((v2s*)B1_next)),    b_size,  1));
            dma_trans_ids[dma_idx] = plp_dma_memcpy((uint32_t) (((v2s*)(lay_next.parameters[LAY_LIN_BIAS]))),    (uint32_t) (((v2s*)B1_next)),    b_size,  1);
            dma_idx += 1;
            // printf("dma bias size: %d \n", b_size);
            // printf("dma weight size: %d %d %d %x \n", w_size, b_size, act_size, (uint32_t) ((v2s*)W1_next));
            int nr_dma_tiles           = w_size / 65532;
            int dima_tiles_overflow    = w_size % 65532;
            if (dima_tiles_overflow!=0) nr_dma_tiles+=1;
            unsigned short curr_w_size = 0;
            int curr_w_idx;

            // printf("nr_dma_tiles %d dima_tiles_overflow %d  \n", nr_dma_tiles, dima_tiles_overflow);

            for(int d=0; d<nr_dma_tiles; d++)
            {
              if(d==nr_dma_tiles-1)
              {
                curr_w_size = dima_tiles_overflow;
                curr_w_idx = (d*65532)/2;
              }
              else
              {
                curr_w_size = 65532;
                curr_w_idx  = d*curr_w_size/2;
              }

              // printf("dma_tile %d curr_w_size %d  curr_w_idx %d \n", d, curr_w_size, curr_w_idx);
              dma_trans_ids[dma_idx] = plp_dma_memcpy((uint32_t) (((v2s*)(lay_next.parameters[LAY_LIN_WEIGHTS]+curr_w_idx))), (uint32_t) (((v2s*)W1_next+curr_w_idx)), curr_w_size,  1);
              dma_idx += 1;
            }
          }
          else
          {
            // plp_dma_wait(plp_dma_memcpy((uint32_t) (((v2s*)(lay_next.parameters[LSTM_BIAS_IH]))), (uint32_t) (((v2s*)B1_next)),     b_size,  1));
            dma_trans_ids[dma_idx] = plp_dma_memcpy((uint32_t) (((v2s*)(lay_next.parameters[LSTM_BIAS_IH]))), (uint32_t) (((v2s*)B1_next)),     b_size,  1);
            dma_idx += 1;
            // plp_dma_wait(plp_dma_memcpy((uint32_t) (((v2s*)(lay_next.parameters[LSTM_WGHT_IH]))), (uint32_t) (((v2s*)W1_next)),  w_size,  1));
            dma_trans_ids[dma_idx] = plp_dma_memcpy((uint32_t) (((v2s*)(lay_next.parameters[LSTM_WGHT_IH]))), (uint32_t) (((v2s*)W1_next)),  w_size,  1);
            dma_idx += 1;
            // plp_dma_wait(plp_dma_memcpy((uint32_t) (((v2s*)(lay_next.parameters[LSTM_BIAS_HH]))), (uint32_t) (((v2s*)B2_next)),    b_size,  1));
            dma_trans_ids[dma_idx] = plp_dma_memcpy((uint32_t) (((v2s*)(lay_next.parameters[LSTM_BIAS_HH]))), (uint32_t) (((v2s*)B2_next)),    b_size,  1);
            dma_idx += 1;
            // plp_dma_wait(plp_dma_memcpy((uint32_t) (((v2s*)(lay_next.parameters[LSTM_WGHT_HH]))), (uint32_t) (((v2s*)W2_next)), b_size*2*lay_next.attributes[LAY_LSTM_HID],  1));
            dma_trans_ids[dma_idx] = plp_dma_memcpy((uint32_t) (((v2s*)(lay_next.parameters[LSTM_WGHT_HH]))), (uint32_t) (((v2s*)W2_next)), b_size*2*lay_next.attributes[LAY_LSTM_HID],  1);
            dma_idx += 1;
          }

      #else // no DMA

          for(int m = 0; m < 4*lay_next.attributes[LAY_LSTM_HID]; m++)
          {
            B1_next[m] = lay_next.parameters[LSTM_BIAS_IH][m];
            B2_next[m] = lay_next.parameters[LSTM_BIAS_HH][m];

            for(int n = 0; n < lay_next.attributes[LAY_LSTM_IN] + W_OFFSET; n++) //+2
            {
              W1_next[m*(lay_next.attributes[LAY_LSTM_IN]+W_OFFSET)+n] = lay_next.parameters[LSTM_WGHT_IH][m*(lay_next.attributes[LAY_LSTM_IN])+n];
            }
            for(int n = 0; n < lay_next.attributes[LAY_LSTM_HID] + W_OFFSET; n++) //+2
            {
              W2_next[m*(lay_next.attributes[LAY_LSTM_HID]+W_OFFSET)+n] = lay_next.parameters[LSTM_WGHT_HH][m*(lay_next.attributes[LAY_LSTM_HID])+n];
            }
          }
      #endif // DMA
      
        }
        //////////////////////////////////////////////////////////////////////////////////////////////
        // CONVOLUTION
        //////////////////////////////////////////////////////////////////////////////////////////////
        else
        {
          printf("ERROR - only Lin Layer or LSTM are supported!!! \n");
          // return 1;
        }
      }
    #endif

    }
    // synch_barrier();


/*****************************************************************************
 *
 * LINEAR LAYER
 *
 *****************************************************************************/
      if(lay.type == LINEAR)
      {


  #ifdef DEBUG_LSTM
    #ifdef MULTICORE
        if ( core_id<lay.attributes[LAY_LIN_TILES] )
        {
    #endif
        printf("Linear (%i, %i)\n", lay.attributes[LAY_LIN_IN], lay.attributes[LAY_LIN_OUT]);
        printf("Inputs in: ");
        PrintTensor(lay.attributes[LAY_LIN_IN], in);

        // printf("Linear (%i, %i)\n", lay.attributes[LAY_LIN_IN], lay.attributes[LAY_LIN_OUT]);
        printf("bias in: ");
        PrintTensor(lay.attributes[LAY_LIN_OUT], B1);

        // printf("Linear (%i, %i)\n", lay.attributes[LAY_LIN_IN], lay.attributes[LAY_LIN_OUT]);
        printf("weights in: ");
        PrintTensor(lay.attributes[LAY_LIN_OUT]*lay.attributes[LAY_LIN_IN], W1);
    #ifdef MULTICORE
        }
    #endif
  #endif

#ifdef MULTICORE
        // printf("INFO - inside 3!!! \n");
        if ( core_id<lay.attributes[LAY_LIN_TILES] )
        {
  #ifdef TILING
          LinearLayer(lay.attributes[LAY_LIN_IN],
                      chunk, //lay.attributes[LAY_LIN_OUT],
                      True,
                      W1, //linear_Weights,
                      B1, //linear_Bias,
                      // Input and Output Features
                      in,   //inFeatures,
                      out+(start)); // outFeatures
  #else // no TILING
          // printf("INFO - inside 3a!!! \n");
          LinearLayer(lay.attributes[LAY_LIN_IN],
                      lay.attributes[LAY_LIN_OUT],
    #ifdef EFFICIENT_CORE_ASSIGNMENT
                      lay.attributes[LAY_LIN_TILE_SIZE],
    #endif
                      True,
    #ifdef BATCHING
                      BATCHING,
    #endif
                      W1, //linear_Weights,
                      B1, //linear_Bias,
                      // Input and Output Features
                      in,   //inFeatures,
                      out); // outFeatures
          // printf("INFO - inside 3b!!! \n");
  #endif // TILING
        }
        // synch_barrier();
        // printf("INFO - inside 4!!! \n");
#else
  #ifdef TILING
        LinearLayer(lay.attributes[LAY_LIN_IN],
                    chunk,
                    True,
                    lay.parameters[LAY_LIN_WEIGHTS],
                    lay.parameters[LAY_LIN_BIAS],
                    // Input and Output Features
                    in,   //inFeatures,
                    out+(start)); // outFeatures
  #else // no TILING
        LinearLayer(lay.attributes[LAY_LIN_IN],
                    lay.attributes[LAY_LIN_OUT],
      #ifdef EFFICIENT_CORE_ASSIGNMENT
                    lay.attributes[LAY_LIN_TILE_SIZE],
      #endif
                    True,
                    lay.parameters[LAY_LIN_WEIGHTS],
                    lay.parameters[LAY_LIN_BIAS],
                    // Input and Output Features
                    in,   //inFeatures,
                    out); // outFeatures
  #endif // TILING
#endif

  #ifdef DEBUG_LSTM
    #ifdef MULTICORE
        if ( core_id<lay.attributes[LAY_LIN_TILES] )
        {
    #endif
        printf("Results in: ");
      #ifdef BATCHING
        PrintTensor(2*lay.attributes[LAY_LIN_OUT], out);
      #else
        PrintTensor(lay.attributes[LAY_LIN_OUT], out);
      #endif
    #ifdef MULTICORE
        }
    #endif
  #endif


#ifdef TILING
        }
#endif // TILING

        toFIRST ^= 1;

        // switch buffer (double buffering)
        if(toFIRST)
        {
          in  = &buffer[BUFFER_SIZE2];
          out = &buffer[0];
        }
        else 
        {
          in  = &buffer[0];
          out = &buffer[BUFFER_SIZE2];
        }
      }

/*****************************************************************************
 *
 * LSTM LAYER
 *
 *****************************************************************************/
      else if (lay.type == LSTM) {

  #ifdef DEBUG_LSTM
    #ifdef MULTICORE
        if ( core_id<lay.attributes[LAY_LSTM_TILES] )
        {
    #endif
        printf("LSTM (%i, %i)\n", lay.attributes[LAY_LSTM_IN], lay.attributes[LAY_LSTM_HID]);
        printf("Inputs in: ");
        PrintTensor(lay.attributes[LAY_LSTM_IN], in);
        // printf("Linear (%i, %i)\n", lay.attributes[LAY_LIN_IN], lay.attributes[LAY_LIN_OUT]);
        printf("bias in1: ");
        PrintTensor(lay.attributes[LAY_LSTM_HID]*4, B1);
        printf("bias in2: ");
        PrintTensor(lay.attributes[LAY_LSTM_HID]*4, B2);

        // printf("Linear (%i, %i)\n", lay.attributes[LAY_LIN_IN], lay.attributes[LAY_LIN_OUT]);
        printf("weights in1: ");
        PrintTensor(lay.attributes[LAY_LSTM_HID]*4*lay.attributes[LAY_LSTM_IN], W1);
        printf("weights in2: ");
        PrintTensor(lay.attributes[LAY_LSTM_HID]*4*lay.attributes[LAY_LSTM_HID], W2);
    #ifdef MULTICORE
        }
    #endif
  #endif

        int numHidden = lay.attributes[LAY_LSTM_HID];
#ifdef MULTICORE
        // synch_barrier();
        // if ( rt_core_id()<NR_CORES )
        // {
        LSTMLayer ( // Layer Attributes
                    lay.attributes[LAY_LSTM_IN], numHidden,
                    // Layer Parameters
                    W1, //linear_Weights, //lay.parameters[LSTM_WGHT_IH],
                    W2, //linear_Weights2, //lay.parameters[LSTM_WGHT_HH],
                    B1, //linear_Bias, //lay.parameters[LSTM_BIAS_IH],
                    B2,//linear_Bias2, //lay.parameters[LSTM_BIAS_HH],
                    // Input and Output Features
                    in,
                    linear_H, //lay.parameters[LSTM_H],
                    // Hidden Features
                    linear_C, //lay.parameters[LSTM_C],
                    // intermediate nodes
                    out,
                    out + 2*numHidden*1, //f
                    out + 3*numHidden*1, //i
                    out + 4*numHidden*1, //g
                    out + 5*numHidden*1  //o
                  );
        in  =  (data_t *)linear_H;
        // }
        // synch_barrier();
#else
        LSTMLayer ( // Layer Attributes
                    lay.attributes[LAY_LSTM_IN], numHidden,
                    // Layer Parameters
                    lay.parameters[LSTM_WGHT_IH],
                    lay.parameters[LSTM_WGHT_HH],
                    lay.parameters[LSTM_BIAS_IH],
                    lay.parameters[LSTM_BIAS_HH],
                    // Input and Output Features
                    in,
                    lay.parameters[LSTM_H],
                    // Hidden Features
                    lay.parameters[LSTM_C],
                    // intermediate nodes
                    out, //h_out
                    out + 2*numHidden*1, //f
                    out + 3*numHidden*1, //i
                    out + 4*numHidden*1, //g
                    out + 1*numHidden*1//o
                  );
        // in  =  (data_t *)lay.parameters[LSTM_H];
        // in  =  (data_t *)lay.parameters[LSTM_H];
#endif


  #ifdef DEBUG_LSTM
    #ifdef MULTICORE
        if ( core_id<lay.attributes[LAY_LSTM_TILES] )
        {
    #endif
        printf("Results at: ");
// #ifdef MULTICORE
//         PrintTensor(lay.attributes[LAY_LSTM_HID], linear_H);
// #else
        PrintTensor(lay.attributes[LAY_LSTM_HID], out);
// #endif
    #ifdef MULTICORE
        }
    #endif
  #endif

        toFIRST ^= 1; 

        // switch buffers
        if(toFIRST) 
        {
          in  = &buffer[BUFFER_SIZE2];
          out = &buffer[0];
        }
        else 
        {
          in  = &buffer[0];
          out = &buffer[BUFFER_SIZE2];
        }

      }
/*****************************************************************************
 *
 * Conv2D LAYER
 *
 *****************************************************************************/
      else if(lay.type == Conv2d) {

  #ifdef DEBUG_LSTM
        printf("Conv2D (%i->%i, ker=%i^2, h*w=%i*%i)\n", lay.attributes[LAY_LIN_IN], lay.attributes[LAY_LIN_OUT], lay.attributes[LAY_CONV_KER],lay.attributes[LAY_CONV_H],lay.attributes[LAY_CONV_W]);
        printf("Inputs in: ");
        PrintTensor(lay.attributes[LAY_LIN_IN]*lay.attributes[LAY_CONV_H]*lay.attributes[LAY_CONV_W], in);
  #endif

        Conv2dLayer(&lay,
                    lay.attributes[LAY_CONV_H],
                    lay.attributes[LAY_CONV_W],
                    in,
                    out
                    );

  #ifdef DEBUG_LSTM
        printf("Conv2D (%i->%i, ker=%i^2, h*w=%i*%i)\n", lay.attributes[LAY_LIN_IN], lay.attributes[LAY_LIN_OUT], lay.attributes[LAY_CONV_KER],lay.attributes[LAY_CONV_H],lay.attributes[LAY_CONV_W]);
        printf("Results in: ");
        PrintTensor(lay.attributes[LAY_CONV_OUT]*lay.attributes[LAY_CONV_H]*lay.attributes[LAY_CONV_W], out);
  #endif

        toFIRST ^= 1; 

        // switch buffers
        if(toFIRST) 
        {
          in  = &buffer[BUFFER_SIZE2];
          out = &buffer[0];
        }
        else 
        {
          in  = &buffer[0];
          out = &buffer[BUFFER_SIZE2];
        }

      }
      else {
        printf("\033[91mERROR: not a valid layer\033[0m\n");
      }

#ifdef MULTICORE

        plp_dma_barrier();
#endif
      synch_barrier();
    }

  return &in[0]; // return address of output feature map
}
