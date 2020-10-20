#!/bin/bash
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

## USAGE:  run_sweep name

name=${1}   # can be lstm or linear
# NR_CORES=${2}

# echo ${NR_CORES}
echo ${name}

cores=( 1 2 4 8 16)

# inp=( 2 34 66 98 130 162 194 226 258 290 322 354 386 418 450 482 )
# out=( 2 34 66 98 130 162 194 226 258 290 322 354 386 418 450 482 )
inp=( 418 )
out=( 418 )

mkdir -p measurements

# date=$(date '+%Y-%m-%d %H:%M:%S')
logfile="measurements/measurements_fcl_${name}_V"

for ((X=0; X<=20; X++))
do
  if [ -f "${logfile}$X.log" ]
  then
    echo "${logfile}$X.log exists"
  else
    logfile="${logfile}$X.log "
    break
  fi
done

csvfile="${logfile%.*}.csv"
echo $logfile

{
for c in "${cores[@]}"
do
    for i in "${inp[@]}"
    do
       for o in "${out[@]}"
        do
            echo "####################################################3"
            echo "###" $i $o $c
            echo "####################################################"
            python scripts/generate_sweep_model.py $i $o $c fcl
            make clean all run platform=gvsoc CONFIG_NB_PE=16
            echo "end_flag"
       done
    done
done
echo "end"
} > $logfile

length=${#inp[@]}
in_start=${inp[0]}
in_step=32
in_stop=${inp[-1]}

length=${#out[@]}
out_start=${out[0]}
out_stop=${out[-1]}
out_step=32

python scripts/get_statistics.py $logfile
echo "###"
echo "### Run:"
echo "### python scripts/get_plots.py --model fcl --csv $csvfile --in_start $in_start --in_stop $in_stop --in_step $in_step --out_start $in_start --out_stop $in_stop --out_step $in_step "
echo "###"
