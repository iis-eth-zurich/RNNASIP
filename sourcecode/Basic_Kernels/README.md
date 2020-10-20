Copyright (C) 2019-2020 ETH Zurich, Switzerland
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Authors:  Renzo Andri, Gianna Paulin



The main machine learning kernels for this project can be found in this folder (i.e. /RNNASIP/sourcecode/Basic_Kernels).

## Setup the Toolchain
Set up tool chain:
```
previous_pwd = $(pwd)
cd $PATH_TO_PULP_SDK
export PULP_RISCV_GCC_TOOLCHAIN=$PATH_TO_RISCV_GCC_TOOLCHAIN
source configs/pulpissimo_rnnext.sh
source pkg/sdk/dev/sourceme.sh 
PULP_CONFIGS_PATH=`pwd`/pulp-configs/configs/:`pwd`/pkg/sdk/dev/install/ws/configs
cd $previous_pwd
make conf
```

## Create Benchmarks
All Networks have been defined in ```scripts/BenchmarkNetworks.py```. If needed add a new network or export the networks to create the ```benchmarks.h``` header file.

```
python3 scripts/BenchmarkNetworks.py
```

## Run the network on the SDK:
Tip: ```make clean``` does not always work properly, use ```rm -rf build && make clean all run```.

### Pulpissimo
```
make clean all run
```

### PULP Open
```
make clean all run platform=gvsoc CONFIG_NB_PE=16
```

## Run the network with traces:
With the CONFIG_OPT attribute, it can be defined which traces should be shown (e.g. insn for all instructions)

### Pulpissimo
```
make all run CONFIG_OPT=gvsoc/trace=insn
```

### PULP Open
```
make all run platform=gvsoc CONFIG_NB_PE=16 runner_args="--trace=insn"
```

## Run on the RTL platform
### Pulpissimo
Dependencies: RTL platform set up and build (see corresponding README for details)
```
export VSIM_PATH=<path to the RTL platform>/sim
source $PULP_PROJECT_HOME/configs/platform-rtl.sh
make clean all run gui=1
```

## Run gate-level simulation and create vcd file for power simulation
```
export APP_PATH=$(pwd)
cd $VSIM_PATH
ln -s <path-to-gate-level-netlist> gate_level_netlist.v
make gate_all
cd $APP_PATH
make clean all run gui=1
```

## Run instruction analysis and profiling
```
# run statistics for active network
bash scripts/run_insn_statistic.sh
# run profiling for all blocks and models
python3 scripts/profile_loop.py
```

## Run verification suite
The verification can be run with the ```run_benchmark.sh``` script. The following settings can be adapted:<br/>
```
OUTPUTBUFFER_SWEEP="true"           # sweeps over all the different output FM tile sizes.
CREATE_STATISTICS="false"           # run instruction settings as well
RUN_AND_CHECK_CORRECTNESS="true"    # verifiy correctness
PLATFORM="rtl"                      # rtl|gvsoc
INPUTFMTILING="both"                # true|false|both
```

```
source run_benchmark.sh
```

## GTKWAVE
You can use the GTKWAVE simulator as follows:
```
make clean all run platform=gvsoc CONFIG_NB_PE=16 runner_args="--event=.* --event-format=vcd --event-tag=debug --event-tag=asm --event-tag=pc --event-tag=core_events --event-tag=clock"

gtkwave build/view.gtkw &
```

## Macro Configurations

The code is highly configurable with the header files *config.h* and *config\_profiling.h* . Here some basic guidelines on the usage.

If working on Pulpissmo (SoC-only) use:

- Profiling has to be done with *#define PROFILING* and deactivated *#define PROFILING\_NEW* and *#define TIMER*
- no multi-core can be allowdL
	- deactivate *#define MULTICORE*
	- number of cores must be one: *#define NR\_CORES 1*
	- define that there is no cluster: *#define SINGLECORE*
- No batching allowed:
	- Deactivate *#define BATCHING 1*

If working on a cluster setup:
- Profiling has to be done with *#define PROFILING\_NEW* and *#define TIMER* anddeactivated *#define PROFILING* 

### Profiling
The following profiling options are implemented:

- PROFILING\_LINEAR\_AMDAHL\_SERIELL
- PROFILING\_LINEAR\_AMDAHL\_PARALLEL
- PROFILING\_LSTM\_AMDAHL\_SERIELL
- PROFILING\_LSTM\_AMDAHL\_PARALLEL
- PROFILING\_TILING
- PROFILING\_EFFICIENT_TILING
- PROFILING\_ALL
- PROFILING\_LINEAR
- PROFILING\_LSTM
- PROFILING\_TWOLINEAR
- PROFILING\_TANH
- PROFILING\_SIG
- PROFILING\_COPY
- PROFILING\_HADM
- PROFILING\_ADDT

## Makefiles
- If working on Pulpissmo (SoC-only) use: *Makefile\_no\_cluster*.
- If working on PULP Open (Cluster) use: *Makefile\_default*.
