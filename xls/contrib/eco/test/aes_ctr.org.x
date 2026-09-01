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

// AES-CTR mode of operation, using AES as the block cipher. ECO benchmark
// adapted from xls/modules/aes/aes_ctr.x (proc aes_ctr) with the test proc
// removed; the cipher itself is imported from xls.modules.aes.

import std;
import xls.modules.aes.aes;
import xls.modules.aes.aes_common;

type Block = aes_common::Block;
type InitVector = aes_common::InitVector;
type Key = aes_common::Key;

// The command sent to the encrypting proc at the beginning of processing.
pub struct Command {
    // The number of bytes to expect in the incoming message.
    // At present, this number must be a multiple of 128.
    msg_bytes: u32,

    // The encryption key.
    key: Key,

    // The width of the encryption key.
    key_width: aes_common::KeyWidth,

    // The initialization vector for the operation.
    iv: InitVector,

    // The initial counter value. When used standalone, this should be 0, but
    // when used as part of GCM, we start encrypting the plaintext with a
    // counter value of 2.
    initial_ctr: u32,

    // The amount by which to increment ctr every cycle. Usually 1, but can be
    // non-unit when part of a parallel GCM implementation.
    ctr_stride: u32,
}

// The current FSM state of the encoding block.
pub enum Step : bool {
    IDLE = 0,
    PROCESSING = 1,
}

// The recurrent state of the proc.
pub struct State {
    step: Step,
    command: Command,
    ctr: uN[32],
    blocks_left: uN[32],
}

// Performs the actual work of encrypting (or decrypting!) a block in CTR mode.
fn aes_ctr_encrypt(key: Key, key_width: aes_common::KeyWidth, ctr: uN[128],
                   block: Block) -> Block {
    let ctr_array = ctr as u32[4];
    let ctr_enc = aes::encrypt(
        key, key_width,
        Block:[
            ctr_array[0] as u8[4],
            ctr_array[1] as u8[4],
            ctr_array[2] as u8[4],
            ctr_array[3] as u8[4]
        ]);
    Block:[
        ((ctr_enc[0] as u32) ^ (block[0] as u32)) as u8[4],
        ((ctr_enc[1] as u32) ^ (block[1] as u32)) as u8[4],
        ((ctr_enc[2] as u32) ^ (block[2] as u32)) as u8[4],
        ((ctr_enc[3] as u32) ^ (block[3] as u32)) as u8[4],
    ]
}

// Note that encryption and decryption are the _EXACT_SAME_PROCESS_!
pub proc aes_ctr {
    command_in: chan<Command> in;
    ptxt_in: chan<Block> in;
    ctxt_out: chan<Block> out;

    init {
        State {
            step: Step::IDLE,
            command: Command {
                msg_bytes: u32:0,
                key: Key:[u8:0, ...],
                key_width: aes_common::KeyWidth::KEY_128,
                iv: InitVector:uN[96]:0,
                initial_ctr: u32:0,
                ctr_stride: u32:0,
            },
            ctr: uN[32]:0,
            blocks_left: uN[32]:0,
        }
    }

    config(command_in: chan<Command> in,
           ptxt_in: chan<Block> in, ctxt_out: chan<Block> out) {
        (command_in, ptxt_in, ctxt_out)
    }

    next(state: State) {
        let step = state.step;

        let (tok, cmd) = recv_if(
            join(), command_in, step == Step::IDLE, zero!<Command>());
        let cmd = if step == Step::IDLE { cmd } else { state.command };
        let ctr = if step == Step::IDLE { cmd.initial_ctr } else { state.ctr };
        let blocks_left = if step == Step::IDLE {
            std::ceil_div(cmd.msg_bytes, u32:16)
        } else {
            state.blocks_left
        };
        let full_ctr = cmd.iv ++ ctr;

        let (tok, block) = recv_if(
            tok, ptxt_in, blocks_left != u32:0, zero!<Block>());
        let ctxt = aes_ctr_encrypt(cmd.key, cmd.key_width, full_ctr, block);
        let tok = send(tok, ctxt_out, ctxt);

        let blocks_left = blocks_left - u32:1;
        let step = if blocks_left == u32:0 { Step::IDLE }
                   else { Step::PROCESSING };

        // We don't have to worry about ctr overflowing (which would result in
        // an invalid encryption), since ctr starts at zero, and the maximum
        // possible number of blocks per command is 2^32 - 1.
        State { step: step, command: cmd, ctr: ctr + cmd.ctr_stride,
                blocks_left: blocks_left }
    }
}
