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
#* Authors:  Renzo Andri, Gianna Paulin                                       *
#*----------------------------------------------------------------------------*

# Dependencies
# centos: sudo yum install autoconf automake libmpc-devel mpfr-devel gmp-devel gawk  bison flex texinfo patchutils gcc gcc-c++ zlib-devel
# ubuntu16: sudo apt-get install autoconf automake autotools-dev curl libmpc-dev libmpfr-dev libgmp-dev gawk build-essential bison flex texinfo gperf libtool patchutils bc zlib1g-dev

export postfix="CentOS"

git clone --recursive https://github.com/pulp-platform/pulp-riscv-gnu-toolchain pulp-riscv-gnu-toolchain_$postfix

cd pulp-riscv-gnu-toolchain_$postfix
git checkout isa-renzo

echo "Make PULP"
./configure --prefix=$(pwd)/opt-riscv/ --with-arch=rv32imc --with-cmodel=medlow --enable-multilib
make 
cd ../




