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

// Revision of ghash.org.x: streamlined for an integrated GCM pipeline. The
// upstream controller now forms the final "lengths" block (len(AAD) ++
// len(CTXT)) itself and appends it to the input stream, as is common in
// decomposed GCM hardware. The command therefore carries a single total
// block count instead of separate AAD/ciphertext counts, the HASH_LENGTHS
// step and its block-construction logic disappear, and the FSM collapses to
// idle/hashing. Interface effects: the command channel payload is retyped
// (two u32 counts merge into one) and the proc state shrinks (one command
// field removed, step narrowed from u2 to u1).
//
// NOTE: This file exists solely to provide a rich, realistic IR for testing
// the XLS ECO toolchain. It has NOT been functionally verified for
// correctness. Use at your own risk.

import xls.modules.aes.aes_common;

type Block = aes_common::Block;

const ZERO_BLOCK = aes_common::ZERO_BLOCK;

// Simply linearizes a block of data.
fn block_to_u128(x: Block) -> uN[128] {
    let x = x[0][0] ++ x[0][1] ++ x[0][2] ++ x[0][3] ++
            x[1][0] ++ x[1][1] ++ x[1][2] ++ x[1][3] ++
            x[2][0] ++ x[2][1] ++ x[2][2] ++ x[2][3] ++
            x[3][0] ++ x[3][1] ++ x[3][2] ++ x[3][3];
    x
}

// Blockifies a uN[128].
fn u128_to_block(x: uN[128]) -> Block {
    let x = x as u8[16];
    Block:[
        [x[0], x[1], x[2], x[3]],
        [x[4], x[5], x[6], x[7]],
        [x[8], x[9], x[10], x[11]],
        [x[12], x[13], x[14], x[15]],
    ]
}

// Computes the multiplication of x and y under GF(128) with the GCM-specific
// modulus u128:0xe10...0 (or in the field defined by x^128 + x^7 + x^2 + x + 1).
// A better implementation would use a pre-programmed lookup table for product
// components, e.g., a 4-bit lookup table for each 4-bit chunk of A * B, where B
// is a pre-set "hash key", referred to as "H" in the literature.
pub fn gf128_mul(x: Block, y: Block) -> Block {
    let x = block_to_u128(x);
    let y = block_to_u128(y);

    let r = u8:0b11100001 ++ uN[120]:0;
    // TODO(rspringer): Can't currently select an element from an array or
    // tuple resulting from a for loop.
    let z_v = for (i, (last_z, last_v)) in u32:0..u32:128 {
        let z = if (x >> (u32:127 - i)) as u1 == u1:0 { last_z } else { last_z ^ last_v };
        let v = if last_v[0:1] == u1:0 { last_v >> 1 } else { (last_v >> 1) ^ r };
        (z, v)
    }((uN[128]:0, y));

    u128_to_block(z_v.0)
}

// Describes the inputs to the GHASH proc.
pub struct Command {
    // The total number of complete blocks to hash: the AAD blocks, then the
    // ciphertext blocks, then the lengths block formed by the upstream GCM
    // controller. Must be at least 1.
    blocks: u32,

    // The hash key to use for tag generation: accepted as an input rather than
    // being computed to avoid introducing an AES block here.
    hash_key: Block,
}

// The current step/state of the GHASH block's FSM.
enum Step : u1 {
    IDLE = 0,
    HASHING = 1,
}

// The carried state of the GHASH proc.
pub struct State {
    // The current FSM step, as above.
    step: Step,

    // The current command being processed.
    command: Command,

    // The number of blocks left to process.
    input_blocks_left: u32,

    // The running hash. Once the final (lengths) block has been folded in,
    // this is the output of the proc.
    last_tag: Block,
}

// Calculates the authentication tag for the Galois Counter Mode of operation
// for block ciphers.
// Since input streams can be of ~arbitrary length, this must be implemented
// as a proc instead of as a fixed function. When idle, this proc accepts a new
// command and will consume one block of input per "tick": AAD, then
// ciphertext, then the controller-formed lengths block, all read from the same
// channel. Once the last block has been hashed, the resulting tag is sent on
// the provided output channel.
pub proc ghash {
    command_in: chan<Command> in;
    input_in: chan<Block> in;
    tag_out: chan<Block> out;

    init {
        State {
            step: Step::IDLE,
            command: Command {
                blocks: u32:0,
                hash_key: ZERO_BLOCK,
            },
            input_blocks_left: u32:0,
            last_tag: ZERO_BLOCK,
        }
    }


    config(command_in: chan<Command> in, input_in: chan<Block> in,
           tag_out: chan<Block> out) {
        (command_in, input_in, tag_out)
    }

    next(state: State) {
        let (tok, command) = recv_if(
            join(), command_in, state.step == Step::IDLE, zero!<Command>());
        let command = if state.step == Step::IDLE { command } else { state.command };
        let blocks_left =
            if state.step == Step::IDLE { command.blocks } else { state.input_blocks_left };
        let last_tag = if state.step == Step::IDLE { ZERO_BLOCK } else { state.last_tag };

        // Every tick consumes one block; the last one is the lengths block.
        let (tok, input_block) = recv(tok, input_in);
        let last_tag = gf128_mul(aes_common::xor_block(last_tag, input_block), command.hash_key);

        // Will underflow if the command violates the blocks >= 1 contract.
        let blocks_left = blocks_left - u32:1;
        let done = blocks_left == u32:0;
        let tok = send_if(tok, tag_out, done, last_tag);

        State {
            step: if done { Step::IDLE } else { Step::HASHING },
            command: command,
            input_blocks_left: blocks_left,
            last_tag: last_tag,
        }
    }
}
