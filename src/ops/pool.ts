/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

// tslint:disable-next-line:max-line-length
import {Tensor4D} from '@tensorflow/tfjs-core';
import {Conv2DInfo} from '@tensorflow/tfjs-core/dist/ops/conv_util';

function pool(x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
  if (convInfo.padInfo.type !== 'VALID' && convInfo.padInfo.type !== 'SAME') {
    throw new Error(
        `TF Backend supports only 'valid' and 'same' padding ` +
        `while padding was ${convInfo.padInfo.type}`);
  }
  const ksize = [1, convInfo.filterHeight, convInfo.filterWidth, 1];
  const strides = [1, convInfo.strideHeight, convInfo.strideWidth, 1];
  const padding = convInfo.padInfo.type;
  const dilation: [number, number] =
      [convInfo.dilationHeight, convInfo.dilationWidth];
  let basePadding: number[][];
  if (padding === 'SAME') {
    basePadding = this.withSpaceToBatchBasePaddings(
        [convInfo.filterHeight, convInfo.filterWidth], dilation);
  } else {
    basePadding = [[0, 0], [0, 0]];
  }
  const [adjustedPadding, adjustedCrops] = this.requiredSpaceToBatchPaddings(
      [convInfo.inHeight, convInfo.inWidth], dilation, basePadding);
  const dataFormat = convInfo.dataFormat === 'channelsLast' ? 'NHWC' : 'NCHW';
  const opAttrs = [
    createTypeOpAttr('T', x.dtype),
    {name: 'ksize', type: this.binding.TF_ATTR_INT, value: ksize},
    {name: 'strides', type: this.binding.TF_ATTR_INT, value: strides},
    {name: 'padding', type: this.binding.TF_ATTR_STRING, value: 'VALID'},
    {name: 'data_format', type: this.binding.TF_ATTR_STRING, value: dataFormat}
  ];
  const convertedX = this.spaceToBatchND(x, dilation, adjustedPadding);
  const y =
      this.executeSingleOutput('MaxPool', opAttrs, [convertedX]) as Tensor4D;
  return this.batchToSpaceND(y, dilation, adjustedCrops);
}
