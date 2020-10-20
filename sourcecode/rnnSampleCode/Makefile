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

PULP_APP = test
PULP_APP_FC_SRCS = test.c
PULP_APP_HOST_SRCS = test.c
PULP_CFLAGS = -O3 -g -lmath  -mhwloopmin=2
PULP_ARCH_CFLAGS = -march=rv32Xpulpv -lm
include $(PULP_SDK_HOME)/install/rules/pulp_rt.mk
