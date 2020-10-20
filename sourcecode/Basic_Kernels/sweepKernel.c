/** @file testKernel.c
 *  @brief Test Program to test basic Kernels implemented in BasicKernel.c and the benchmark suite
 *
 *  Test program for Basic ML kernels
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

#include <stdio.h>
#include <config.h>

#include "sweep_config.h"


#ifdef ASIP

#else // no ASIP
    // #include "pulp.h"
    // #include "rt/rt_api.h"
    #include "config_profiling.h"
#endif // ASIP

#include "basicKernel.h"

/// @cond DOXYGEN_EXCLUDE
#include "benchmarks.h"
/// @endcond



#ifdef PROFILING
/** @brief used for profiling with old runtime */
int event_mask;
#endif // PROFILING


/** @brief buffer to store intermediate FM */
#ifdef MULTICORE
__attribute__ ((section(".heapsram"))) data_t buffer[BUFFER_SIZE];
#else
L2_DATA data_t buffer[BUFFER_SIZE];
#endif

// #define NR_SWEEPS 4
// int nr_sweeps = NR_SWEEPS;
// L2_DATA data_t sweep_neurons[NR_SWEEPS] = {4, 36, 68, 100};



/** @brief runs the defined model either in profiling mode or not
 */
static int run_networks()
{

    int core_id = rt_core_id();


    {
#ifdef DEBUG
        printf("Entered cluster on cluster %d core %d\n", get_cluster_id(), core_id);
#endif

        data_t tmp_avgerror = 0;
        // pointer to output FM
        data_t * m0_OutAct;

#ifdef ASIP
        long cycles_before = chess_cycle_count();
#endif // ASIP

        numFunctionCalls = 0;


        int depth   = 1;
        int dim_in  = N_INP;
        int dim_out = N_OUT;

        int nr_tiles  = NR_CORES;
        int tile_size = (dim_out / nr_tiles) + ((dim_out & (nr_tiles-1))!=0);

#ifdef LSTM_ON
        struct layer model[1] = { \
            {.type=LSTM, .attributes={ dim_in, dim_out, nr_tiles, tile_size, 0 }, .parameters={ m_lstm_weight_ih[0], m_lstm_weight_hh[0], m_lstm_bias_ih, m_lstm_bias_hh,m_lstm_h, m_lstm_c } } \
        };
#else
        struct layer model[1] = { \
            {.type=LINEAR, .attributes={ dim_in, dim_out, nr_tiles, tile_size, 0 }, .parameters={m_linear_Bias,m_linear_Weights[0],0,0,0,0} } \
        };
#endif


#ifdef PREFETCH_ICACHE
        // Prefetch the ICACHE for Marsellus
        // correct address also for gvsoc? -> seems to work also on gvsoc
        if(core_id==0)
        {
            *(int*)(ICACHE_PREFETCH) = 0xFFFF;
        }

//////////////////////////////////////////////////////////////////////////////////////////////
// Fill ICACHE with empty computations before PROFILING
//////////////////////////////////////////////////////////////////////////////////////////////
    

        for(int i=0; i<2; i++)
        {
            m0_OutAct = inferNetwork(model, depth, m_In, buffer);
        }
        synch_barrier();

#endif // PREFETCH_ICACHE


//////////////////////////////////////////////////////////////////////////////////////////////
// Profiling with OLD RT (single Core)
//////////////////////////////////////////////////////////////////////////////////////////////
#ifdef PROFILING

        rt_perf_init(&perf);// global performance counter

        int maskset[] = {\
                            (1<<RT_PERF_CYCLES) | (1<<RT_PERF_INSTR),
                            (1<<RT_PERF_ACTIVE_CYCLES),
                            (1<<RT_PERF_LD_STALL),
                            (1<<RT_PERF_JR_STALL),
                            (1<<RT_PERF_IMISS),
                            (1<<RT_PERF_LD),
                            (1<<RT_PERF_ST),
                            (1<<RT_PERF_JUMP),
                            (1<<RT_PERF_BRANCH),
                            (1<<RT_PERF_BTAKEN),
                            (1<<RT_PERF_RVC),
                            (1<<RT_PERF_LD_EXT),
                            (1<<RT_PERF_ST_EXT),
                            (1<<RT_PERF_LD_EXT_CYC),
                            (1<<RT_PERF_ST_EXT_CYC),
                            (1<<RT_PERF_TCDM_CONT)\
                        };

        // running once for every counter
        // for(unsigned int i=0; i < sizeof(maskset)/sizeof(int); i++)
        // if(core_id == 0)
        {
            rt_perf_conf(&perf, maskset[i]);
            printf("maskset[i] %d \n", maskset[i]);
            rt_perf_reset(&perf);
            PROFILING_ALL_START
#ifdef MULTICORE
            // printf("%s%d\n", "Startpe", core_id;
#else
            // printf("%s\n", "Startfc");
#endif


//////////////////////////////////////////////////////////////////////////////////////////////
// Profiling with NEW RT (Multi Core)
//////////////////////////////////////////////////////////////////////////////////////////////
#elif defined(PROFILING_NEW)

        // int maskset[] = {\
        //                     (1<<CSR_PCER_CYCLES) | (1<<CSR_PCER_INSTR), /* Count the number of cycles the core was running */ /* Count the number of instructions executed */
        //                     (1<<CSR_PCER_LD_STALL),                     /* Number of load use hazards */
        //                     (1<<CSR_PCER_JMP_STALL),                    /* Number of jump register hazards */
        //                     (1<<CSR_PCER_IMISS),                        /* Cycles waiting for instruction fetches. i.e. the number of instructions wasted due to non-ideal caches */
        //                     (1<<CSR_PCER_LD),                           /* Number of memory loads executed. Misaligned accesses are counted twice */
        //                     (1<<CSR_PCER_ST),                           /* Number of memory stores executed. Misaligned accesses are counted twice */
        //                     (1<<CSR_PCER_JUMP),                         /* Number of jump instructions seen, i.e. j, jr, jal, jalr */
        //                     (1<<CSR_PCER_BRANCH),                       /* Number of branch instructions seen, i.e. bf, bnf */
        //                     (1<<CSR_PCER_TAKEN_BRANCH),                 /* Number of taken branch instructions seen, i.e. bf, bnf */
        //                     (1<<CSR_PCER_RVC),                          /* Number of compressed instructions */
        //                     // (1<<CSR_PCER_ELW),                          /* Cycles wasted due to ELW instruction */
        //                     (1<<CSR_PCER_LD_EXT),                       /* Number of memory loads to EXT executed. Misaligned accesses are counted twice. Every non-TCDM access is considered external */
        //                     (1<<CSR_PCER_ST_EXT),                       /* Number of memory stores to EXT executed. Misaligned accesses are counted twice. Every non-TCDM access is considered external */
        //                     (1<<CSR_PCER_LD_EXT_CYC),                   /* Cycles used for memory loads to EXT. Every non-TCDM access is considered external */
        //                     (1<<CSR_PCER_ST_EXT_CYC),                   /* Cycles used for memory stores to EXT. Every non-TCDM access is considered external */
        //                     (1<<CSR_PCER_TCDM_CONT)\
        //                 };                                              /* Cycles wasted due to TCDM/log-interconnect contention */


        // running once for every counter
        // int i=0;
        // if(core_id == 0)
        // for(unsigned int i=0; i < 1; i++)
#ifdef TIMER
        for(unsigned int i=0; i < CSR_PCER_NB_EVENTS+1; i++)
#else
        // for(unsigned int i=0; i < CSR_PCER_NB_EVENTS; i++)
        for(unsigned int i=0; i < CSR_PCER_NB_EVENTS; i++)
#endif
        {
            // unsigned int i=0;

            // for(unsigned int i=0; i < CSR_PCER_NB_EVENTS; i++)
            // {
            //     perf[core_id].perf_counters[i] = 0;
            //     perf[core_id].perf_counter_id = i; // FIXME
            // }

            perf[core_id].perf_counters[i] = 0;
            perf[core_id].perf_counter_id = i;
            perf_reset();
            PROFILING_ALL_START

#endif // PROFILING_NEW
//////////////////////////////////////////////////////////////////////////////////////////////


#ifdef DEBUG
        printf("Start\n");
#endif // PRINTF_ACTIVE


        synch_barrier();
        m0_OutAct = inferNetwork(model, depth, m_In, buffer);


//////////////////////////////////////////////////////////////////////////////////////////////
// Profiling with OLD RT (Multi Core)
//////////////////////////////////////////////////////////////////////////////////////////////
#ifdef PROFILING

            PROFILING_ALL_END
            rt_perf_save(&perf);
        }

#ifdef DEBUG
        printf("END\n");
#endif // DEBUG

    int models = 0;

#ifdef MODEL0
        models |= 1<<0;
#endif // MODEL0
#ifdef MODEL1
        models |= 1<<1;
#endif // MODEL1
#ifdef MODEL2
        models |= 1<<2;
#endif // MODEL2
#ifdef MODEL3
        models |= 1<<3;
#endif // MODEL3
#ifdef MODEL4
        models |= 1<<4;
#endif // MODEL4
#ifdef MODEL5
        models |= 1<<5;
#endif // MODEL5
#ifdef MODEL6
        models |= 1<<6;
#endif // MODEL6
#ifdef MODEL7
        models |= 1<<7;
#endif // MODEL7
#ifdef MODEL8
        models |= 1<<8;
#endif // MODEL8
#ifdef MODEL9
        models |= 1<<9;
#endif // MODEL9
#ifdef MODEL10
        models |= 1<<10;
#endif // MODEL10

        printf("%s, %i, ", CODE_SEGMENT, models);
        printf("%d \n", numFunctionCalls*sizeof(int)/sizeof(maskset));
        printf("RT_PERF_CYCLES: %d \n", rt_perf_get(&perf, RT_PERF_CYCLES));
        printf("RT_PERF_INSTR: %d \n", rt_perf_get(&perf, RT_PERF_INSTR));
        printf("RT_PERF_ACTIVE_CYCLES: %d \n", rt_perf_get(&perf, RT_PERF_ACTIVE_CYCLES));
        printf("RT_PERF_LD_STALL: %d \n", rt_perf_get(&perf, RT_PERF_LD_STALL));
        printf("RT_PERF_JR_STALL: %d \n", rt_perf_get(&perf, RT_PERF_JR_STALL));
        printf("RT_PERF_IMISS: %d \n", rt_perf_get(&perf, RT_PERF_IMISS));
        printf("RT_PERF_LD: %d \n", rt_perf_get(&perf, RT_PERF_LD));
        printf("RT_PERF_ST: %d \n", rt_perf_get(&perf, RT_PERF_ST));
        printf("RT_PERF_JUMP: %d \n", rt_perf_get(&perf, RT_PERF_JUMP));
        printf("RT_PERF_BRANCH: %d \n", rt_perf_get(&perf, RT_PERF_BRANCH));
        printf("RT_PERF_BTAKEN: %d \n", rt_perf_get(&perf, RT_PERF_BTAKEN));
        printf("RT_PERF_RVC: %d \n", rt_perf_get(&perf, RT_PERF_RVC));
        printf("RT_PERF_LD_EXT: %d \n", rt_perf_get(&perf, RT_PERF_LD_EXT));
        printf("RT_PERF_ST_EXT: %d \n", rt_perf_get(&perf, RT_PERF_ST_EXT));
        printf("RT_PERF_LD_EXT_CYC: %d \n", rt_perf_get(&perf, RT_PERF_LD_EXT_CYC));
        printf("RT_PERF_ST_EXT_CYC: %d \n", rt_perf_get(&perf, RT_PERF_ST_EXT_CYC));
        printf("RT_PERF_TCDM_CONT: %d \n", rt_perf_get(&perf, RT_PERF_TCDM_CONT));

#ifdef PRINTF_ACTIVE
        printf(" Profiling Done\n");
#endif


//////////////////////////////////////////////////////////////////////////////////////////////
// Profiling with NEW RT (Multi Core)
//////////////////////////////////////////////////////////////////////////////////////////////
#elif defined(PROFILING_NEW)

            PROFILING_ALL_END
        }

        int models = 0;

#ifdef MODEL0
        models |= 1<<0;
#endif // MODEL0
#ifdef MODEL1
        models |= 1<<1;
#endif // MODEL1
#ifdef MODEL2
        models |= 1<<2;
#endif // MODEL2
#ifdef MODEL3
        models |= 1<<3;
#endif // MODEL3
#ifdef MODEL4
        models |= 1<<4;
#endif // MODEL4
#ifdef MODEL5
        models |= 1<<5;
#endif // MODEL5
#ifdef MODEL6
        models |= 1<<6;
#endif // MODEL6
#ifdef MODEL7
        models |= 1<<7;
#endif // MODEL7
#ifdef MODEL8
        models |= 1<<8;
#endif // MODEL8
#ifdef MODEL9
        models |= 1<<9;
#endif // MODEL9
#ifdef MODEL10
        models |= 1<<10;
#endif // MODEL10

    #ifdef MULTICORE
        if ( core_id < NR_CORES )
        {
    #endif
        printf("%s, %i, \n", CODE_SEGMENT, models);
        printf("%d,\n", numFunctionCalls/CSR_PCER_NB_EVENTS);
        // printf("%d\n", cpu_perf_get(0));
        // printf("%d\n", cpu_perf_get(1));
        // printf("%d\n", cpu_perf_get(10));
        // printf("%d\n", cpu_perf_get(2));
        // printf("%d\n", cpu_perf_get(3));
        // printf("%d\n", cpu_perf_get(4));
        // printf("%d\n", cpu_perf_get(5));
        // printf("%d\n", cpu_perf_get(6));
        // printf("%d\n", cpu_perf_get(7));
        // printf("%d\n", cpu_perf_get(8));
        // printf("%d\n", cpu_perf_get(9));
        // printf("%d\n", cpu_perf_get(11));
        // printf("%d\n", cpu_perf_get(12));
        // printf("%d\n", cpu_perf_get(13));
        // printf("%d\n", cpu_perf_get(14));
        // printf("%d\n", cpu_perf_get(15));
        // printf("%d\n", cpu_perf_get(16));

        #ifdef TIMER
        printf("core:%d %d \n", core_id, timer_cl);
        #endif

        if ( core_id == 0 )
        {
            printf("#### NUM_INPUT  %d \n", N_INP    );
            printf("#### NUM_OUTPUT %d \n", N_OUT    );
            printf("#### NR_CORES   %d \n", NR_CORES );
#ifdef LSTM_ON
            printf("#### LSTM_ON    %d \n", LSTM_ON  );
#endif

#if NR_CORES==1 & defined(TIMER)
            printf("#### total_cycles %d \n", timer_cl  );
        // printf("#### total_instr  351 %d \n", LSTM_ON  );
#else
            printf("#### total_cycles %d \n", timer_cl  );
        // printf("#### total_instr  351 %d \n", int(LSTM_ON)  );
#endif
        }
        for(int e=0; e < CSR_PCER_NB_EVENTS; e++)
        {
            printf("core:%d %d\n", core_id, perf[core_id].perf_counters[e]);
        }

    #ifdef MULTICORE
        }
    #endif


#endif // PROFILING_NEW
//////////////////////////////////////////////////////////////////////////////////////////////

    // }
// }

#ifdef ASIP
#ifdef PRINTF_ACTIVE
        long cycles_after = chess_cycle_count();
        int chess_storage(X31) eoc =tmp_avgerror;
        printf("eof = %i", eoc);
        printf("The result is %d\nCycle executed: %ld\n", 0, cycles_after - cycles_before);
#endif // PRINTF_ACTIVE
#endif // ASIP
    }
    synch_barrier();

    return 0;
}


/** @brief Main function calling the run_networks() function on FC or on the Cluster
 */
int main()
{

#ifdef DEBUG
    printf("Entering main controller core %d\n", get_core_id());
#endif

// multicore implementation
#ifdef MULTICORE
    cluster_start(0, run_networks);
    int retval = cluster_wait(0);

#ifdef DEBUG
    printf("Got retval from cluster %d\n", retval);
#endif

// FC implementation
#else

    run_networks();

#endif

    return 0;

}