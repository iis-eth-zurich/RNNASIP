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

function init {
cd ../../vp && source ../vp/init_vega_COS.bash
cd ../sourcecode/Basic_Kernels && make conf platform=gvsoc
sed -i '/PROFILING/s/^\/\///g' config_profiling.h
# turn on all models
sed -i '/MODEL/s/^/\/\//g' config_profiling.h


}

function activate_opt {
# comment them out
sed -i '/PULP_CL_ARCH_CFLAGS/s/^/#/g' Makefile
sed -i '/PULP_FC_ARCH_CFLAGS/s/^/#/g' Makefile
sed -i '/PULP_ARCH_LDFLAGS/s/^/#/g' Makefile
}

function deactivate_opt {
sed -i '/PULP_CL_ARCH_CFLAGS/s/^#//g' Makefile
sed -i '/PULP_FC_ARCH_CFLAGS/s/^#//g' Makefile
sed -i '/PULP_ARCH_LDFLAGS/s/^#//g' Makefile
}
