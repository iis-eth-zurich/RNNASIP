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

echo "Clone Basic Tests"
git clone -v git@iis-git.ee.ethz.ch:pulp-tests/rt-tests.git
echo "All tests should work except for tests including hyperbus for pulpissio"
echo""

echo "Clone old runtime (pulp-rt) examples"
git clone -v git@github.com:pulp-platform/pulp-rt-examples.git
echo""

echo "Clone new runtime (pulp-runtime) examples"
git clone -v git@github.com:pulp-platform/pulp-runtime-examples.git
echo""
