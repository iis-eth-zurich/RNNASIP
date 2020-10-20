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

if __name__ == '__main__':

    """
    Generates the csv file containing the statistics (perf measurements). It checks for '####' at the beginning of every line, if it finds it, the second word will be the key and the third word will be the measured performance in cycles.
    Input: filename of the file containing the redirected output from console (.txt file)
    Output: .csv file containig the statistics
    """

start_token1 = 'RM'
stop_token1 = 'end_flag'

start_found = 0

log_file = open(sys.argv[1])

perf_dict = OrderedDict()

count = 0

while True:

    line       = log_file.readline()
    line_split = line.split()

    if len(line_split) == 0:
        continue

    if line_split[0]=="end":
        break
    if (line_split[0]==start_token1 and start_found == 0):
        start_found=1
        print("start found\n")

    if start_found == 0:
        continue

    if ((line_split[0]==stop_token1) and start_found == 1):
        start_found=0
        if line_split[0] == stop_token1:
            print("end found\n")
        else:
            print("end found True or False\n")

    if start_found == 1 and line_split[0]=='####':

        print(line_split)

        if (line_split[2] == "True" or line_split[2] == "False"):
            line_split[2] = int(line_split[2] == 'True')

        if line_split[1] in perf_dict:
            print("key exists\n")
            # if ((line_split[2] in perf_dict.values())):
                # print("value does not exists")
            perf_dict[line_split[1]].append(int(line_split[2]))
            # else:
                # print("value exists")
            # print(perf_dict)
        else:
            print("key doesn't exists\n")
            perf_dict[line_split[1]] = [int(line_split[2])]


result = {}

for key,value in perf_dict.items():
    if value not in result.values():
        result[key] = value
        print()

print(result)

# print(perf_dict)
print(result.values())

keys = result.keys()
# print(keys)
with open(sys.argv[1][:-4]+".csv", "w") as outfile:
    writer = csv.writer(outfile, delimiter = ",")
    writer.writerow(list(keys))
    writer.writerows(zip(*[result[key] for key in keys]))
