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

// Run Length Encoding (RLE) encoder: compresses an input stream of repeating
// symbols into (symbol, count) pairs. Both channels carry a `last` flag that
// marks the end of a transmission; on `last` the encoder flushes its state.
//
// Self-contained ECO benchmark adapted from xls/modules/rle/rle_enc.x
// (proc RunLengthEncoder), specialized to byte symbols (SYMBOL_WIDTH = 8)
// with an 8-bit run counter (COUNT_WIDTH = 8, maximum run length 255).

import std;

// Input packet: one symbol plus the end-of-transmission flag.
struct PlainData {
    symbol: u8,
    last: bool,
}

// Output packet: a run of `count` repetitions of `symbol`.
struct CompressedData {
    symbol: u8,
    count: u8,
    last: bool,
}

// State preserved across activations.
struct RleEncState {
    // Symbol from the previous evaluation, valid if prev_count > 0.
    prev_symbol: u8,
    // Symbol count from the previous evaluation; zero means the previous
    // evaluation sent all data and counting restarts.
    prev_count: u8,
    // Whether the previous symbol was the last one in the transmission.
    prev_last: bool,
}

proc rle_enc {
    input_r: chan<PlainData> in;
    output_s: chan<CompressedData> out;

    init {
        RleEncState { prev_symbol: u8:0, prev_count: u8:0, prev_last: false }
    }

    config(input_r: chan<PlainData> in, output_s: chan<CompressedData> out) {
        (input_r, output_s)
    }

    next(state: RleEncState) {
        let zero_input = PlainData { symbol: u8:0, last: false };
        let (input_tok, input) =
            recv_if(join(), input_r, !state.prev_last, zero_input);

        let prev_symbol_valid = state.prev_count != u8:0;
        let symbol_differ =
            prev_symbol_valid && (input.symbol != state.prev_symbol);
        let overflow = state.prev_count == std::unsigned_max_value<u32:8>();

        let (symbol, count, last) = if (state.prev_last) {
            (u8:0, u8:0, false)
        } else if (symbol_differ || overflow) {
            (input.symbol, u8:1, input.last)
        } else {
            (input.symbol, state.prev_count + u8:1, input.last)
        };

        let data = CompressedData {
            symbol: state.prev_symbol,
            count: state.prev_count,
            last: state.prev_last,
        };

        let do_send = state.prev_last || symbol_differ || overflow;
        let data_tok = send_if(input_tok, output_s, do_send, data);

        RleEncState { prev_symbol: symbol, prev_count: count, prev_last: last }
    }
}
