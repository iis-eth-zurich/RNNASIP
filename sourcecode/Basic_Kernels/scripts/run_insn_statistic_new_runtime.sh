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

# create report directory if not already existing
mkdir -p reports

TRACE_FILE=reports/trace_`date "+%F-%T"`
# if [ -z ${TRACE_FILE+x} ]; then TRACE_FILE=reports/trace_`date "+%F-%T"`; fi
if [ -z ${TRACE_OPT+x} ]; then TRACE_OPT="--trace=insn"; fi
if [ -z ${TRACE_GVSOC+x} ]; then TRACE_GVSOC=gvsoc; fi
# make clean all run platform="${TRACE_GVSOC}" runner_args="${TRACE_OPT}" &> ${TRACE_FILE}

make clean all run platform=gvsoc CONFIG_NB_PE=16 runner_args=--trace=insn &> ${TRACE_FILE}

cat ${TRACE_FILE} | grep -E "insn|Start|\n"  > tmp
echo "Created traces and stored in ${TRACE_FILE}"

echo "Get start and stop address"
riscv32-unknown-elf-objdump -d build/testKernel/testKernel | grep "<inferNetwork>:" | cut -d " " -f1 > start.txt

LINE=$(riscv32-unknown-elf-objdump -d build/testKernel/testKernel | grep -n "<inferNetwork>:" | gawk '{print $1}' FS=":")
# echo "${LINE}"
riscv32-unknown-elf-objdump -d build/testKernel/testKernel | tail -n +${LINE} | grep -m 1 "ret" | gawk '{print $1}' FS=":" > stop.txt


echo "python3 scripts/create_statistic.py tmp Start 1 1 | tee ${TRACE_FILE}_summary"
python3 scripts/create_statistic.py tmp Start 1 2 | tee ${TRACE_FILE}_summary
echo "Created instruction summary in ${TRACE_FILE}_summary"
# rm tmp
# CONFIG_OPT=gvsoc/trace=insn
