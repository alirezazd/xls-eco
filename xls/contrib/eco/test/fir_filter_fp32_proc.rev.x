// Copyright 2026 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Revised version of fir_filter_fp32_proc.org.x: the filter order increases
// from 12 to 16 taps for a sharper roll-off. The delay line kept as proc
// state resizes from F32[12] to F32[16]; the sample channels are unchanged.
//
// The coefficients are the 16-tap triangular (Bartlett) low-pass window
// [1,2,3,4,5,6,7,8,8,7,6,5,4,3,2,1] / 72, which sums to 1.

import float32;

type F32 = float32::F32;

const NUM_TAPS = u32:16;

const COEFFS = F32[NUM_TAPS]:[
    float32::unflatten(u32:0x3C638E39),  //  1/72
    float32::unflatten(u32:0x3CE38E39),  //  2/72
    float32::unflatten(u32:0x3D2AAAAB),  //  3/72
    float32::unflatten(u32:0x3D638E39),  //  4/72
    float32::unflatten(u32:0x3D8E38E4),  //  5/72
    float32::unflatten(u32:0x3DAAAAAB),  //  6/72
    float32::unflatten(u32:0x3DC71C72),  //  7/72
    float32::unflatten(u32:0x3DE38E39),  //  8/72
    float32::unflatten(u32:0x3DE38E39),  //  8/72
    float32::unflatten(u32:0x3DC71C72),  //  7/72
    float32::unflatten(u32:0x3DAAAAAB),  //  6/72
    float32::unflatten(u32:0x3D8E38E4),  //  5/72
    float32::unflatten(u32:0x3D638E39),  //  4/72
    float32::unflatten(u32:0x3D2AAAAB),  //  3/72
    float32::unflatten(u32:0x3CE38E39),  //  2/72
    float32::unflatten(u32:0x3C638E39),  //  1/72
];

proc fir_filter_proc {
    sample_in: chan<F32> in;
    result_out: chan<F32> out;

    init { F32[NUM_TAPS]:[float32::zero(u1:0), ...] }

    config(sample_in: chan<F32> in, result_out: chan<F32> out) {
        (sample_in, result_out)
    }

    next(delay_line: F32[NUM_TAPS]) {
        let (tok, sample) = recv(join(), sample_in);

        // Shift the delay line: the newest sample enters at index 0.
        let shifted = for (i, acc): (u32, F32[NUM_TAPS]) in u32:0..NUM_TAPS {
            let v = if i == u32:0 { sample } else { delay_line[i - u32:1] };
            update(acc, i, v)
        }(F32[NUM_TAPS]:[float32::zero(u1:0), ...]);

        // Convolve the delay line with the coefficients.
        let result = for (i, acc): (u32, F32) in u32:0..NUM_TAPS {
            float32::add(acc, float32::mul(COEFFS[i], shifted[i]))
        }(float32::zero(u1:0));

        let tok = send(tok, result_out, result);
        shifted
    }
}
