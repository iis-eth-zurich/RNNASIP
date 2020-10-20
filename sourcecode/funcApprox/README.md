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

Authors:  Renzo Andri

# Matlab Scripts for Evaluation of the Piecewise Linear Approximation 

Transcedent function can be approximated by splitting the function in multiple intervals and linear function within the intervals. Functions like tangent hyperbolic and sigmoid are very well suited as the functions are smooth and converge quickly to a constant value.

## File overview
* ```evalClass.m``` Helper Class to evaluate approximation. The ```plot``` function does plot the evaluation versus the original version, but also returns the mean square error, maximum error and the linear coefficients (m and q).
* ```fixedPoint.m``` Models quantization with several helper functions. (see file for more details)
* ```taylorExpansion.m``` Basic scripts to calculate coefficients of taylor expansion.
* ```sig.m``` and ```tanh_eval.m``` scripts used to plot and evaluate the piecewise linear approximation.

## How to use
The scripts in ```sig.m``` and ```tanh_eval.m``` are the same and evaluate either sigmoid or the tangent hyperbolic function.

First the approximation interval is defined in ```firstPt```, ```lastPt``` and ```stepSize```, the the fixed-point format (i.e. fxPtFormat) and the function in ```f```.

Then the script includes several scripts for calculating and ploting. 
* Part 1: Calculates the approximation and plots it for visual comparison
* Part 2: Exports the data to C.
* Part 3: Calculates the approximation for a set of parameters (range and interval number)
    - 3a) plots the mean square error in 3D plot
    - 3b) plots the mean square error in 2D plot
    - 3c) plots the maximum error in 3D plot
    - 3d) plots the maximum error in 2D plot




