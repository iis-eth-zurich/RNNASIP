%% @file fixedPoint.m
%> @brief Helper Class to modelate fixed-point behaviour and ploting
%>
%> Helper Class to modelate fixed-point behaviour and ploting
%>
%> @author Renzo Andri (andrire)
%>
%> *----------------------------------------------------------------------------*
%> * Copyright (C) 2019-2020 ETH Zurich, Switzerland                            *
%> * SPDX-License-Identifier: Apache-2.0                                        *
%> *                                                                            *
%> * Licensed under the Apache License, Version 2.0 (the "License");            *
%> * you may not use this file except in compliance with the License.           *
%> * You may obtain a copy of the License at                                    *
%> *                                                                            *
%> * http://www.apache.org/licenses/LICENSE-2.0                                 *
%> *                                                                            *
%> * Unless required by applicable law or agreed to in writing, software        *
%> * distributed under the License is distributed on an "AS IS" BASIS,          *
%> * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
%> * See the License for the specific language governing permissions and        *
%> * limitations under the License.                                             *
%> *----------------------------------------------------------------------------*
%%

classdef fixedPoint
   properties
      signProp
      qFrac
      qInt
   end
   methods
      % Constructor
      function obj = fixedPoint(signProp, qInt, qFrac)
               obj.signProp = signProp;
               obj.qFrac = qFrac;
               obj.qInt = qInt;
      end
      % plot values lim1:prec:lim2 quantized and non-quantized
      function plot(obj, lim1, lim2, prec)
         
         plot(lim1:prec:lim2, obj.func(lim1:prec:lim2))
         hold on
         plot(lim1:prec:lim2, obj.discretize(obj.func(lim1:prec:lim2)))
         hold off
      end
      % Quantization including overflow (takes float and returns "quantized" float) 
      function o = discretize(obj, data)
          %part_i = bitand(int16(floor(data)), int16((2^qInt-1)*ones(size(data))));
          part_i = int16(floor(data));
          part_i = min(part_i, int16((2^obj.qInt-1)));
          part_i = max(part_i, -int16((2^obj.qInt)));
          part_f = round((data - floor(data))*2^obj.qFrac,0)/2^obj.qFrac;
          o=double(part_i)+part_f;
      end
      % converts and quantizes float to fixed-point representation
      function o = float2int(obj, data)
          o=obj.discretize(data)*2^obj.qFrac;
      end
      % prints the fixed-point values in C format.
      % (e.g. int16_t varName = {4096, 1027};
      function tmp_string = printC(obj, data, varName) 
          tmp=obj.float2int( data);
          tmp_string = ['int16_t ' varName ' = {'];
          for i=1:(size(tmp,2)-1)
              tmp_string = [tmp_string num2str(tmp(i)) ', '];
          end
          tmp_string = [tmp_string num2str(tmp(i+1)) '};'];
          
      end
      % returns the sum of two fixed-point nummbers 
      function o = add(obj, addend1, addend2)
          o = obj.discretize(obj.discretize(addend1)+obj.discretize(addend2));
      end   
      % prints the Q format
      function str=tostring(obj)
          if obj.signProp
             str = 'Q';
          else 
             str = 'UQ';
          end
          str=[str, num2str(obj.qInt), ',', num2str(obj.qFrac)];
      end
   end
end
