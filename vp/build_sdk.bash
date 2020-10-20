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

export operatingSystem="CentOS"
export pulpEdition="gap"

#dependencies
if [ "$operatingSystem" == "Ubuntu16" ]; then
 #sudo apt install git python3-pip gawk texinfo libgmp-dev libmpfr-dev libmpc-dev swig3.0 libjpeg-dev lsb-core doxygen python-sphinx sox graphicsmagick-libmagick-dev-compat libsdl2-dev libswitch-perl libftdi1-dev cmake
 #sudo pip3 install artifactory twisted prettytable sqlalchemy pyelftools openpyxl xlsxwriter pyyaml numpy 
 echo ""
elif [ "$operatingSystem" == "CentOS" ]; then
#sudo yum install git python34-pip python34-devel gawk texinfo gmp-devel mpfr-devel libmpc-devel swig libjpeg-turbo-devel redhat-lsb-core doxygen python-sphinx sox GraphicsMagick-devel ImageMagick-devel SDL2-devel perl-Switch libftdi-devel cmake
#sudo pip3 install artifactory twisted prettytable sqlalchemy pyelftools openpyxl xlsxwriter pyyaml numpy 
 echo ""
fi

if [ ! -d "pulp-sdk_$operatingSystem_$pulpEdition" ]; then
git clone https://github.com/pulp-platform/pulp-sdk.git pulp-sdk_$operatingSystem_$pulpEdition
fi

cd pulp-sdk_$operatingSystem_$pulpEdition

git checkout f5f0d18ba7b589e3998425ea265bcc327e4f033f
git pull origin master
source configs/pulpissimo_rnnext.sh
source configs/platform-gvsoc.sh 
make all
echo "export PULP_RISCV_GCC_TOOLCHAIN=$PULP_RISCV_GCC_TOOLCHAIN" >> sourceme.sh
