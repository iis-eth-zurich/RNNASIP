/** @file lut.h
 *  @brief Coefficients for taylor expansion.
 *
 * @author Renzo Andri (andrire)
 * @author Gianna Paulin (pauling)
 *
 *----------------------------------------------------------------------------*
 * Copyright (C) 2019-2020 ETH Zurich, Switzerland                            *
 * SPDX-License-Identifier: Apache-2.0                                        *
 *                                                                            *
 * Licensed under the Apache License, Version 2.0 (the "License");            *
 * you may not use this file except in compliance with the License.           *
 * You may obtain a copy of the License at                                    *
 *                                                                            *
 * http://www.apache.org/licenses/LICENSE-2.0                                 *
 *                                                                            *
 * Unless required by applicable law or agreed to in writing, software        *
 * distributed under the License is distributed on an "AS IS" BASIS,          *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
 * See the License for the specific language governing permissions and        *
 * limitations under the License.                                             *
 *----------------------------------------------------------------------------*
 */

#include "config.h"

#ifdef FixedPt

/** @brief LUT coefficients for tanh */
data_t tanh_coeff[4] = {1*(1<<(q_frac)), -1.0/3.0*(1<<(q_frac)), 2.0/215.0*(1<<(q_frac)), -17.0/315.0*(1<<(q_frac))};
/** @brief LUT coefficients for sigm */
data_t sig_coeff[4] = {1.0/2*(1<<(q_frac)), 1.0/4*(1<<(q_frac)), 1.0/48*(1<<(q_frac)), 1.0/480*(1<<(q_frac)), };

#else

/** @brief LUT coefficients for tanh */
data_t tanh_coeff[4] = {1, -1/3, 2/215, -17/315};
/** @brief LUT coefficients for sigm */
data_t sig_coeff[4] = {1/2, 1/4, 1/48, 1/480, };

#endif