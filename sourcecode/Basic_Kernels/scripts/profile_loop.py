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
#* Authors:  Renzo Andri                                                      *
#*----------------------------------------------------------------------------*

import os
import subprocess
import time

profiles = [
"#define PROFILING_ALL",
"#define PROFILING_LINEAR",
"#define PROFILING_LSTM",
"#define PROFILING_TWOLINEAR",
"#define PROFILING_SIG",
"#define PROFILING_TANH",
"#define PROFILING_ADDT",
"#define PROFILING_HADM",
"#define PROFILING_COPY",
"#define PROFILING_FILL",]

models = [
"#define MODEL0",
"#define MODEL1",
"#define MODEL2",
"#define MODEL3",
"#define MODEL5",
"#define MODEL6",
"#define MODEL7",
"#define MODEL8",
"#define MODEL9",
"#define MODEL10",
"#define MODEL11", 
"all"
]
# all_models = ""
# for model in models:
#    all_models += model+"\n"
# models.append(all_models)
os.system("touch wait.file")
os.system ("rm reports/log_profiling.txt")

data_f = open("reports/log_profiling.txt",'w');
data_f.write("Profile,model,Function_Calls,Total_cycles,Instructions,Active_cycles,Load_data_hazards,Jump_stalls,Instruction_cache_misses,Load_accesses,Store_accesses,Jumps,Branches,Branches_taken,Compressed instructions,External_load_accesses,External_store_accesses,External_load_stall_cycles,External_store_stall_cycles,TCDM_contention_cycles,CSR_hazards\n")
data_f.close()
for model in models:
   for profile in profiles:
      data_f = open("config_profiling.h",'w');
      data_f.write("#define PROFILING\n");
      data_f.write("#define OUTPUTBUFFER 4\n");
      data_f.write("//#define FMINTILING\n");
      data_f.write("//#define FMOUTTILING\n");
      data_f.write("//#define MANUALLOOPUNFOLDING\n");
      for model_i in models:
         if model_i == "all":
            continue;
         if model_i != model and model != "all":
            data_f.write("// ");
         data_f.write(model_i+"\n");
      for profile_i in profiles:
         if profile_i != profile:
            data_f.write("// ");
         data_f.write(profile_i+"\n");
      data_f.close();
      subprocess.Popen("make clean all run | tee log | grep PROFILING >> ./reports/log_profiling.txt && rm wait.file", shell=True)
      while (os.path.isfile("wait.file")):
         time.sleep(0.3);
      os.system("touch wait.file")

os.system("rm wait.file")

