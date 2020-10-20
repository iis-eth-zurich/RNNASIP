/** @file general.h
 *  @brief Header File for basic ML kernels (including FC, LSTM, Conv2D for the RNN ASIP
 * 
 *  Header file contains basic definitions needed for computations
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


#ifndef GENERAL_HEADER_FILE
#define GENERAL_HEADER_FILE

#include <config.h>
#include <config_profiling.h>
#ifndef ASIP
    #include "pulp.h"
    // #include "rt/rt_api.h"
    #include <math.h>
#endif


//////////////////////////////////////////////////////////////////////////////////////////////
#ifdef FixedPt
    /** Data Type, e.g. 16-bit integer (short int)*/
    typedef short int data_t;
    // #define doInFloat
    #define doTaylor
    /// Use linear approximation of tanh/sigmoid
    //#define doLinear

    /// Manual Loop unfolding is used to efficient register reuse (TODO extend compiler, to allow explicit allocation of register array and automatic loop unfolding)
    // #define MANUALLOOPUNFOLDING

#ifdef ASIP
    /// Number of registers for output FM tiling
    #define OUTPUTBUFFER 4
#else
    // #define OUTPUTBUFFER 10
#endif

    /// For fixed-point implementation integer part Q3,12
    #define q_int 3
    /// For fixed-point implementation fractional part Q3,12
    #define q_frac 12
    /// Fixed-Point Format used for shifting
    #define q_fraqP1 q_frac // to be checked
    /// PI in fixed-point format
    #define PI ((data_t)(3.1415927*(1<<q_frac)) & 0xffff)
    /// Half of pi in fixed-point format
    #define PIHalf PI/2

#ifdef doInFloat
    #define tanh_threshold 0x0//0x7fff //PI/2 //0x7fff
    #define sig_threshold 0x0//0x7fff  //128//0x7fff
#endif

#ifdef doLinear
/// turn off taylor expansion (todo)
    #define TURNOFF
#endif

#ifdef doTaylor
    #define tanh_threshold 0x7fff //PI/2 //0x7fff
    #define sig_threshold 0x7fff  //128//0x7fff
#endif

#else // FixedPt not defined 
    typedef float data_t;
    #define PI 3.1415927
    #define PIHalf PI/2
#endif // FixedPt
//////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ASIP
    #define NOINLINE 
    #define ALWAYS_INLINE 
    #define __restrict__
    #define RT_L2_DATA
#else
    /// generic definition of no_inline
    #define NOINLINE __attribute__ ((noinline))
    /// generic definition of always_inline
    #define ALWAYS_INLINE __attribute__ ((always_inline))
#endif

#define TRUE 1
#define FALSE 0
#define True 1
#define False 0

#ifdef ASIP
    inline data_t Min(data_t a, data_t b);
    inline data_t Max(data_t a, data_t b);
    inline data_t Abs(data_t a);          
    inline float Min(float a, float b);
    inline float Max(float a, float b);
    inline float Abs(float a); 

    inline int Min(int a, int b);
    inline int Max(int a, int b);
    inline int Abs(int a);

    inline int pulpRNNExt_tanh(int tanh_value);
    inline int pulpRNNExt_sig(int tanh_value);

    // Old implementation of HWLOOPS
    // #ifdef HWLOOPS
    // #define __hwloop__(times, instr_count) chess_separator_scheduler(); for(int i=0; i<times; i++) chess_loop_range(1,) { chess_separator_scheduler();hwloop_(times,(instr_count)*INSTR_STEP);chess_separator_scheduler();
    // #define __hwloop_end__ chess_separator_scheduler(); }
    // #else 
    // #define __hwloop__(times, instr_count)  for(int i=0; i<times; i++) {
    // #define __hwloop_end__ }
    // #endif

#else
    #define Min(a, b)               (((a)<(b))?(a):(b))
    #define Max(a, b)               (((a)>(b))?(a):(b))
    #define Abs(a)                  (((a)>(0.0))?(a):(-a))
#endif


/// Number of elements for taylor expansion
#define tailorPrecission 32

#define ACT_NONE 0
#define ACT_TANH 1
#define ACT_SIG 2


#ifdef ASIP

#ifdef ASIP_USETANHSIG
    inline data_t tzscale_tanh(data_t x) {return ext_act(x,0);}
    inline data_t tzscale_sig(data_t x) {return ext_act(x,1);}
    inline int32_t tzscale_tanh(int32_t x) {return ext_act(x,0);}
    inline int32_t tzscale_sig(int32_t x) {return ext_act(x,1);}
#else 
    // nothing
#endif
    inline data_t Tanh(data_t value);
    inline data_t sig(data_t value);
    // #ifndef SIMD
    // typedef int v2s;
    // #endif
#endif


//////////////////////////////////////////////////////////////////////////////////////////////
// Define Doublebuffering sizes
//////////////////////////////////////////////////////////////////////////////////////////////
#ifdef BATCHING
#define BUFFER_SIZE BATCHING*1024
#else
#define BUFFER_SIZE 2048
// #define BUFFER_SIZE 1024
#endif
#define BUFFER_SIZE2 BUFFER_SIZE/2
#define BUFFER_SIZE4 BUFFER_SIZE/4

#ifdef SWEEP
#define MAX_SWEEP 162
#endif


#ifdef LSTM_ON
#ifdef SWEEP
#define BUFFER_LIN_B1_SIZE 2*(4*MAX_SWEEP)
#else // SWEEP
#define BUFFER_LIN_B1_SIZE 2*(4*72)
#endif // SWEEP
#else
#ifdef SWEEP
#define BUFFER_LIN_B1_SIZE 2*(500)
#else
#define BUFFER_LIN_B1_SIZE 2*(500)
#endif
#endif
#define BUFFER_LIN_B1_SIZE2 BUFFER_LIN_B1_SIZE/2

#ifdef LSTM_ON
#ifdef SWEEP
#define BUFFER_LIN_W1_SIZE 2*((4*MAX_SWEEP+W_OFFSET)*(MAX_SWEEP+W_OFFSET))
#else // SWEEP
// #define BUFFER_LIN_W1_SIZE 2*((300+W_OFFSET)*(100+W_OFFSET))
#define BUFFER_LIN_W1_SIZE 2*((4*72+W_OFFSET)*(72+W_OFFSET))
#endif // SWEEP
#else
#ifdef SWEEP
#define BUFFER_LIN_W1_SIZE 2*((500+W_OFFSET)*(500+W_OFFSET))
// #define BUFFER_LIN_W1_SIZE 2*((500+W_OFFSET)*(500+W_OFFSET))
#else
// #define BUFFER_LIN_W1_SIZE 2*((250+W_OFFSET)*(200+W_OFFSET))
#define BUFFER_LIN_W1_SIZE 2*((500+W_OFFSET)*(500+W_OFFSET))
#endif
#endif
#define BUFFER_LIN_W1_SIZE2 BUFFER_LIN_W1_SIZE/2

#ifdef LSTM_ON
#ifdef SWEEP
#define BUFFER_LIN_B2_SIZE 2*(4*MAX_SWEEP)
#else // SWEEP
#define BUFFER_LIN_B2_SIZE 2*(4*72)
#endif // SWEEP
#else
#define BUFFER_LIN_B2_SIZE 0
#endif
#define BUFFER_LIN_B2_SIZE2 BUFFER_LIN_B2_SIZE/2

#ifdef LSTM_ON
#ifdef SWEEP
#define BUFFER_LIN_W2_SIZE 2*((4*MAX_SWEEP+W_OFFSET)*(MAX_SWEEP+W_OFFSET))
#else // SWEEP
// #define BUFFER_LIN_W2_SIZE 2*((300+W_OFFSET)*(100+W_OFFSET))
#define BUFFER_LIN_W2_SIZE 2*((4*72+W_OFFSET)*(72+W_OFFSET))
#endif // SWEEP
#else
#define BUFFER_LIN_W2_SIZE 0
#endif
#define BUFFER_LIN_W2_SIZE2 BUFFER_LIN_W2_SIZE/2

#ifdef LSTM_ON
#ifdef SWEEP
#define BUFFER_LIN_C_SIZE (2*MAX_SWEEP)
#else // SWEEP
#define BUFFER_LIN_C_SIZE (2*72)
#endif // SWEEP
#else
#define BUFFER_LIN_C_SIZE 0
#endif
#define BUFFER_LIN_C_SIZE2 BUFFER_LIN_C_SIZE/2

#ifdef LSTM_ON
#ifdef SWEEP
#define BUFFER_LIN_H_SIZE (2*MAX_SWEEP)
#else // SWEEP
#define BUFFER_LIN_H_SIZE (2*72)
#endif // SWEEP
#else
#define BUFFER_LIN_H_SIZE 0
#endif
#define BUFFER_LIN_H_SIZE2 BUFFER_LIN_H_SIZE/2


#define MAX_NR_TRANSACTIONS 16

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))

//////////////////////////////////////////////////////////////////////////////////////////////
// Benchmark structure defines
//////////////////////////////////////////////////////////////////////////////////////////////
/// Supported layer Types
enum layerType
{
    LINEAR = 0, /**< Linear Layer/Fully-Connected Layer */
    RNN    = 1, /**< Recurrent Neural Layer */
    LSTM   = 2, /**< Long short-term Memory Layer */
    Conv2d = 3  /**< 2D Convolution Layer */
};

/// Layer Data
struct layer {
    enum layerType type;     /**< Layer Type (FC, RNN, ...) */
    int attributes[5];       /**< Layer Attributes */
    data_t * parameters[6];  /**< Parameters (weights, bias, ...) */
};

// attributes
#define LAY_LIN_IN      0   ///< Layer Attribute ID for Input Neurons in FC Layer
#define LAY_LIN_OUT     1   ///< Layer Attribute ID for Output Neurons in FC Layer
#define LAY_LIN_BIAS    0   ///< Parameter ID in FC Layer
#define LAY_LIN_WEIGHTS 1   ///< Weight ID in FC Layer
#define LAY_LIN_TILES   2   ///< Nr of Tiles ID in FC Layer
#define LAY_LIN_TILE_SIZE  3 ///< Nr of Tiles ID in FC Layer
#define LAY_LSTM_IN     0   ///< Layer Attribute ID for Input Neurons in LSTM
#define LAY_LSTM_HID    1   ///< Layer Attribute ID for Hideen Neurons in LSTM
#define LAY_LSTM_TILES  2   ///< Nr of Tiles ID in LSTM Layer
#define LAY_LSTM_TILE_SIZE 3 ///< Nr of Tiles ID in LSTM Layer
#define LSTM_WGHT_IH    0   ///< Weight input to hidden ID in LSTM Layer
#define LSTM_WGHT_HH    1   ///< Weight hidden to hidden ID in LSTM Layer
#define LSTM_BIAS_IH    2   ///< Bias input to hidden ID in LSTM Layer
#define LSTM_BIAS_HH    3   ///< Bias hidden to hidden ID in LSTM Layer
#define LSTM_H          4   ///< Number of hidden neurons in LSTM Layer
#define LSTM_C          5   ///< Number of internal states LSTM Layer
#define CONV_WGHT       0   ///< Weight Parameter ID in 2D Conv Layer
#define CONV_BIAS       1   ///< Bias Parameter ID in 2D Conv Layer
#define LAY_CONV_IN     0   ///< Layer Attribute ID for spatial Input FM size in 2D Conv Layer
#define LAY_CONV_OUT    1   ///< Layer Attribute ID for spatial Output FM size in 2D Conv Layer
#define LAY_CONV_KER    2   ///< Layer Attribute ID for kernel size in 2D Conv Layer
#define LAY_CONV_H      3   ///< Layer Attribute ID for height of input FM in 2D Conv Layer
#define LAY_CONV_W      4   ///< Layer Attribute ID for width of input FM in 2D Conv Layer
//////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////
// Define Define-Combinations
//////////////////////////////////////////////////////////////////////////////////////////////
#ifndef ASIP
#if (defined(TILING_HARD) && defined(FMOUTTILING) && defined(MANUALLOOPUNFOLDING) && defined(VLIWEXT))
#define EFFICIENT_CORE_ASSIGNMENT 1
#endif
#endif


//////////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////////////
// Profiling defines
//////////////////////////////////////////////////////////////////////////////////////////////
#define CODE_SEGMENT "NONE"

#ifdef PROFILING_ALL
#   define CODE_SEGMENT "PROFILING_ALL"
#   define PROFILING_ALL_START startPerf();
#   define PROFILING_ALL_END endPerf();
#else
#   define PROFILING_ALL_END 
#   define PROFILING_ALL_START
#endif

#ifdef PROFILING_LINEAR
#   define CODE_SEGMENT "PROFILING_LINEAR"
#   define PROFILING_LINEAR_START startPerf();
#   define PROFILING_LINEAR_END endPerf();
#else
#   define PROFILING_LINEAR_END 
#   define PROFILING_LINEAR_START 
#endif

#ifdef PROFILING_LSTM
#   define CODE_SEGMENT "PROFILING_LSTM"
#   define PROFILING_LSTM_START startPerf();
#   define PROFILING_LSTM_END endPerf();
#else
#   define PROFILING_LSTM_END 
#   define PROFILING_LSTM_START
#endif

#ifdef PROFILING_TANH
#   define CODE_SEGMENT "PROFILING_TANH"
#   define PROFILING_TANH_START startPerf();
#   define PROFILING_TANH_END endPerf();
#else
#   define PROFILING_TANH_END 
#   define PROFILING_TANH_START
#endif

#ifdef PROFILING_FILL
#   define CODE_SEGMENT "PROFILING_FILL"
#   define PROFILING_FILL_START startPerf();
#   define PROFILING_FILL_END endPerf();
#else
#   define PROFILING_FILL_END 
#   define PROFILING_FILL_START
#endif

#ifdef PROFILING_TWOLINEAR
#   define CODE_SEGMENT "PROFILING_TWOLINEAR"
#   define PROFILING_TWOLINEAR_START startPerf();
#   define PROFILING_TWOLINEAR_END endPerf();
#else
#   define PROFILING_TWOLINEAR_END 
#   define PROFILING_TWOLINEAR_START
#endif

#ifdef PROFILING_SIG
#   define CODE_SEGMENT "PROFILING_SIG"
#   define PROFILING_SIG_START startPerf();
#   define PROFILING_SIG_END endPerf();
#else
#   define PROFILING_SIG_END 
#   define PROFILING_SIG_START
#endif

#ifdef PROFILING_ADDT
#   define CODE_SEGMENT "PROFILING_ADDT"
#   define PROFILING_ADDT_START startPerf();
#   define PROFILING_ADDT_END endPerf();
#else
#   define PROFILING_ADDT_END 
#   define PROFILING_ADDT_START
#endif

#ifdef PROFILING_HADM
#   define CODE_SEGMENT "PROFILING_HADM"
#   define PROFILING_HADM_START startPerf();
#   define PROFILING_HADM_END endPerf();
#else
#   define PROFILING_HADM_END 
#   define PROFILING_HADM_START
#endif

#ifdef PROFILING_COPY
#   define CODE_SEGMENT "PROFILING_COPY"
#   define PROFILING_COPY_START startPerf();
#   define PROFILING_COPY_END endPerf();
#else
#   define PROFILING_COPY_END 
#   define PROFILING_COPY_START
#endif

#ifdef PROFILING_LINEAR_AMDAHL_SERIELL
#   define CODE_SEGMENT "PROFILING_LINEAR_AMDAHL_SERIELL"
#   define PROFILING_LINEAR_AMDAHL_SERIELL_START startPerf();
#   define PROFILING_LINEAR_AMDAHL_SERIELL_END endPerf();
#else
#   define PROFILING_LINEAR_AMDAHL_SERIELL_END
#   define PROFILING_LINEAR_AMDAHL_SERIELL_START
#endif

#ifdef PROFILING_LINEAR_AMDAHL_PARALLEL
#   define CODE_SEGMENT "PROFILING_LINEAR_AMDAHL_PARALLEL"
#   define PROFILING_LINEAR_AMDAHL_PARALLEL_START startPerf();
#   define PROFILING_LINEAR_AMDAHL_PARALLEL_END endPerf();
#else
#   define PROFILING_LINEAR_AMDAHL_PARALLEL_END
#   define PROFILING_LINEAR_AMDAHL_PARALLEL_START
#endif

#ifdef PROFILING_LSTM_AMDAHL_SERIELL
#   define CODE_SEGMENT "PROFILING_LSTM_AMDAHL_SERIELL"
#   define PROFILING_LSTM_AMDAHL_SERIELL_START startPerf();
#   define PROFILING_LSTM_AMDAHL_SERIELL_END endPerf();
#else
#   define PROFILING_LSTM_AMDAHL_SERIELL_END
#   define PROFILING_LSTM_AMDAHL_SERIELL_START
#endif

#ifdef PROFILING_LSTM_AMDAHL_PARALLEL
#   define CODE_SEGMENT "PROFILING_LSTM_AMDAHL_PARALLEL"
#   define PROFILING_LSTM_AMDAHL_PARALLEL_START startPerf();
#   define PROFILING_LSTM_AMDAHL_PARALLEL_END endPerf();
#else
#   define PROFILING_LSTM_AMDAHL_PARALLEL_END
#   define PROFILING_LSTM_AMDAHL_PARALLEL_START
#endif

#ifdef PROFILING_TILING
#   define CODE_SEGMENT "PROFILING_TILING"
#   define PROFILING_TILING_START startPerf();
#   define PROFILING_TILING_END endPerf();
#else
#   define PROFILING_TILING_END
#   define PROFILING_TILING_START
#endif

#ifdef PROFILING_EFFICIENT_TILING
#   define CODE_SEGMENT "PROFILING_EFFICIENT_TILING"
#   define PROFILING_EFFICIENT_TILING_START startPerf();
#   define PROFILING_EFFICIENT_TILING_END endPerf();
#else
#   define PROFILING_EFFICIENT_TILING_END
#   define PROFILING_EFFICIENT_TILING_START
#endif


#ifdef TIMER
#   define TIMER_START startTimer();
#   define TIMER_END endTimer();
#else
#   define TIMER_START
#   define TIMER_END
#endif


#ifndef ASIP

#ifdef TIMER
    int timer_cl;
#endif

#ifdef PROFILING
    rt_perf_t perf;
#elif defined(PROFILING_NEW)

    /// Counter structure
    struct perf_counter {
        int perf_all_counters;
        int perf_counter_id;
        int perf_counters [CSR_PCER_NB_EVENTS];
    };

    struct perf_counter perf[NR_CORES];
#endif

    int numFunctionCalls;
    void  startPerf ();
    void  endPerf ();
    void  startTimer ();
    void  endTimer ();

#endif
//////////////////////////////////////////////////////////////////////////////////////////////


#ifdef ASIP
    #define register_attribute chess_storage(chess_register)
    // int32_t rD; v2s rs1, rs2;
    #define SDOTP_GENERIC(rD, rs1, rs2) rD = rD + (rs1 * rs2)
#else
    #define register_attribute register 
    // int32_t rD; v2s rs1, rs2;
    #define SDOTP_GENERIC(rD, rs1, rs2) rD = __SUMDOTP2(rs1, rs2, rD);
    #define PL_SDOTP0(rD, rAddr, rB) asm volatile("pl.sdotsp.h.0 %0, %1, %2" : "+r" (rD),  "+r" (rAddr) : "r" (rB) )
    #define PL_SDOTP1(rD, rAddr, rB) asm volatile("pl.sdotsp.h.1 %0, %1, %2" : "+r" (rD),  "+r" (rAddr) : "r" (rB) )
#endif

#endif // #ifdef GENERAL_HEADER_FILE
