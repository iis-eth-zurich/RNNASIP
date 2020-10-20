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
#* Authors:  Renzo Andri                                                      *
#*----------------------------------------------------------------------------*

# create report directory if not already existing
mkdir -p reports

if [ -z ${TRACE_FILE+x} ]; then TRACE_FILE=reports/trace_`date "+%F-%T"`; fi
if [ -z ${TRACE_OPT+x} ]; then TRACE_OPT=gvsoc/trace=insn; fi
make clean all run CONFIG_OPT="${TRACE_OPT}" &> ${TRACE_FILE}
cat ${TRACE_FILE} | grep -E "insn|Start|\n"  > tmp
echo "Created traces and stored in ${TRACE_FILE}"

python3 scripts/create_statistic.py tmp Start | tee ${TRACE_FILE}_summary
echo "Created instruction summary in ${TRACE_FILE}_summary"
rm tmp
# CONFIG_OPT=gvsoc/trace=insn
