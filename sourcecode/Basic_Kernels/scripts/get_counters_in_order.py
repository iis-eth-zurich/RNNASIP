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
import numpy as np
# import re

# print("""Usage get_counters_in_order.py NR_CORES """)

# file = open("./perf_counters.txt", "r")

nr_cores = sys.argv[1]
nr_counter =  18

cores    = []
counters = [[] for _ in range(int(nr_cores))]
# print(counters)
for i in range(int(nr_cores)):
    cores = cores + ["core:"+str(i)]

with open("./perf_counters.txt") as f:
    for line in f:
        line_split = line.split()
        # print(line_split[1])
        for idx, core in enumerate(cores):
            # print(idx, core, line_split[0])
            # if re.compile(core).match(line_split[0]):
            if core == line_split[0]:
                # print("found ", core, line_split[1])
                counters[idx].append(str(line_split[1]))
                break
                # print(counters)

# for idx, core in enumerate(cores):
#     print(core)
#     for c in range(nr_counter):
#         print(counters[idx][c])
#     print("\n")
sum_cont = 0
for c in range(nr_counter):
    for idx, core in enumerate(cores):
        if (c == nr_counter-1):
            sum_cont = sum_cont + int(counters[idx][c])
        if (idx == (int(nr_cores)-1)):
            sys.stdout.write(str(counters[idx][c]))
        else:
            sys.stdout.write(str(counters[idx][c]) + '\t')
        # print(str(counters[idx][c]) + '\t',) 
    print("")

# print("TOTAL CONTENTION: ", sum_cont)