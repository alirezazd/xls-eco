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

// Revision of ram_demux.org.x: three-way demultiplexer. Replays the
// interface evolution the zstd decoder itself went through (see upstream
// xls/modules/zstd/ram_demux3.x, which realizes the same interface by
// chaining two 2-way demuxes): a third RAM port is added, the select
// channel widens to u2 (valid selects are 0..2), and each selector queue
// gains a second bit plane so every in-flight request records a 2-bit port
// index. The switch still takes effect only after every outstanding request
// has received its response.
//
// NOTE: This file exists solely to provide a rich, realistic IR for testing
// the XLS ECO toolchain. It has NOT been functionally verified for
// correctness. Use at your own risk.

import std;
import xls.examples.ram;

const RAM_SIZE = u32:32;
const RAM_DATA_WIDTH = u32:8;
const RAM_ADDR_WIDTH = std::clog2(RAM_SIZE);
const RAM_WORD_PARTITION_SIZE = u32:1;
const RAM_NUM_PARTITIONS = ram::num_partitions(RAM_WORD_PARTITION_SIZE, RAM_DATA_WIDTH);
const DEMUX_INIT_SEL = u2:0;
const DEMUX_QUEUE_LEN = u32:5;

// First bit of queue is not used to simplify the implementation.
// Queue end is encoded using one-hot and if it is equal to 1,
// then the queue is empty. Queue length should be greater or equal
// to RAM latency, otherways the demux might not work properly.
type Queue = uN[DEMUX_QUEUE_LEN + u32:1];

struct RamDemuxState {
    sel: u2,
    sel_q_rd: Queue,     // low bit plane of the per-entry port index
    sel_q_wr: Queue,
    sel_q_rd_hi: Queue,  // high bit plane of the per-entry port index
    sel_q_wr_hi: Queue,
    sel_q_rd_end: Queue,
    sel_q_wr_end: Queue,
}

pub proc ram_demux {
    type ReadReq = ram::ReadReq<RAM_ADDR_WIDTH, RAM_NUM_PARTITIONS>;
    type ReadResp = ram::ReadResp<RAM_DATA_WIDTH>;
    type WriteReq = ram::WriteReq<RAM_ADDR_WIDTH, RAM_DATA_WIDTH, RAM_NUM_PARTITIONS>;
    type WriteResp = ram::WriteResp;

    sel_req_r: chan<u2> in;
    sel_resp_s: chan<()> out;

    rd_req_r: chan<ReadReq> in;
    rd_resp_s: chan<ReadResp> out;
    wr_req_r: chan<WriteReq> in;
    wr_resp_s: chan<WriteResp> out;

    rd_req0_s: chan<ReadReq> out;
    rd_resp0_r: chan<ReadResp> in;
    wr_req0_s: chan<WriteReq> out;
    wr_resp0_r: chan<WriteResp> in;

    rd_req1_s: chan<ReadReq> out;
    rd_resp1_r: chan<ReadResp> in;
    wr_req1_s: chan<WriteReq> out;
    wr_resp1_r: chan<WriteResp> in;

    rd_req2_s: chan<ReadReq> out;
    rd_resp2_r: chan<ReadResp> in;
    wr_req2_s: chan<WriteReq> out;
    wr_resp2_r: chan<WriteResp> in;

    config(
        sel_req_r: chan<u2> in,
        sel_resp_s: chan<()> out,
        rd_req_r: chan<ReadReq> in,
        rd_resp_s: chan<ReadResp> out,
        wr_req_r: chan<WriteReq> in,
        wr_resp_s: chan<WriteResp> out,

        rd_req0_s: chan<ReadReq> out,
        rd_resp0_r: chan<ReadResp> in,
        wr_req0_s: chan<WriteReq> out,
        wr_resp0_r: chan<WriteResp> in,

        rd_req1_s: chan<ReadReq> out,
        rd_resp1_r: chan<ReadResp> in,
        wr_req1_s: chan<WriteReq> out,
        wr_resp1_r: chan<WriteResp> in,

        rd_req2_s: chan<ReadReq> out,
        rd_resp2_r: chan<ReadResp> in,
        wr_req2_s: chan<WriteReq> out,
        wr_resp2_r: chan<WriteResp> in
    ) {
        (
            sel_req_r, sel_resp_s,
            rd_req_r, rd_resp_s, wr_req_r, wr_resp_s,
            rd_req0_s, rd_resp0_r, wr_req0_s, wr_resp0_r,
            rd_req1_s, rd_resp1_r, wr_req1_s, wr_resp1_r,
            rd_req2_s, rd_resp2_r, wr_req2_s, wr_resp2_r,
        )
    }

    init {
        RamDemuxState {
            sel: DEMUX_INIT_SEL,
            sel_q_rd: Queue:0,
            sel_q_wr: Queue:0,
            sel_q_rd_hi: Queue:0,
            sel_q_wr_hi: Queue:0,
            sel_q_rd_end: Queue:1,
            sel_q_wr_end: Queue:1
        }
    }

    next(state: RamDemuxState) {
        let tok0 = join();

        // receive requests from input channel
        // conditional reading is not required here ase the queue would
        // never be full (assuming its length is greater or equal to RAM
        // latency), as there would be at maxiumum one new request added
        // to queue per cycle and the response for the first one should
        // be received after number of cycles equal to RAM latency (which
        // is less or equal to queue length)
        let (rdtok0, rd_req, rd_req_valid) = recv_non_blocking(tok0, rd_req_r, zero!<ReadReq>());
        let (sel_q_rd_end, sel_q_rd, sel_q_rd_hi) = if rd_req_valid {
            trace_fmt!("[RamDemux] Received read request: {:#x}", rd_req);
            (state.sel_q_rd_end << u32:1,
             (state.sel_q_rd << u32:1) | ((state.sel[0:1] as Queue) << u32:1),
             (state.sel_q_rd_hi << u32:1) | ((state.sel[1:2] as Queue) << u32:1))
        } else {
            (state.sel_q_rd_end, state.sel_q_rd, state.sel_q_rd_hi)
        };

        let (wrtok0, wr_req, wr_req_valid) = recv_non_blocking(tok0, wr_req_r, zero!<WriteReq>());
        let (sel_q_wr_end, sel_q_wr, sel_q_wr_hi) = if wr_req_valid {
            trace_fmt!("[RamDemux] Received write request: {:#x}", wr_req);
            (state.sel_q_wr_end << u32:1,
             (state.sel_q_wr << u32:1) | ((state.sel[0:1] as Queue) << u32:1),
             (state.sel_q_wr_hi << u32:1) | ((state.sel[1:2] as Queue) << u32:1))
        } else {
            (state.sel_q_wr_end, state.sel_q_wr, state.sel_q_wr_hi)
        };


        // send requests to output channel 0
        let rd_req0_cond = (state.sel == u2:0 && rd_req_valid);
        let rdtok1_0 = send_if(rdtok0, rd_req0_s, rd_req0_cond, rd_req);
        if rd_req0_cond {
            trace_fmt!("[RamDemux] Sent read request to channel 0: {:#x}", rd_req);
        } else {};

        let wr_req0_cond = (state.sel == u2:0 && wr_req_valid);
        let wrtok1_0 = send_if(wrtok0, wr_req0_s, wr_req0_cond, wr_req);
        if wr_req0_cond {
            trace_fmt!("[RamDemux] Sent write request to channel 0: {:#x}", wr_req);
        } else {};

        // send requests to output channel 1
        let rd_req1_cond = (state.sel == u2:1 && rd_req_valid);
        let rdtok1_1 = send_if(rdtok0, rd_req1_s, rd_req1_cond, rd_req);
        if rd_req1_cond {
            trace_fmt!("[RamDemux] Sent read request to channel 1: {:#x}", rd_req);
        } else {};

        let wr_req1_cond = (state.sel == u2:1 && wr_req_valid);
        let wrtok1_1 = send_if(wrtok0, wr_req1_s, wr_req1_cond, wr_req);
        if wr_req1_cond {
            trace_fmt!("[RamDemux] Sent write request to channel 1: {:#x}", wr_req);
        } else {};

        // send requests to output channel 2
        let rd_req2_cond = (state.sel == u2:2 && rd_req_valid);
        let rdtok1_4 = send_if(rdtok0, rd_req2_s, rd_req2_cond, rd_req);
        if rd_req2_cond {
            trace_fmt!("[RamDemux] Sent read request to channel 2: {:#x}", rd_req);
        } else {};

        let wr_req2_cond = (state.sel == u2:2 && wr_req_valid);
        let wrtok1_4 = send_if(wrtok0, wr_req2_s, wr_req2_cond, wr_req);
        if wr_req2_cond {
            trace_fmt!("[RamDemux] Sent write request to channel 2: {:#x}", wr_req);
        } else {};

        // check which channel should be used for read/write
        let rd_resp_ch = ((sel_q_rd_hi & sel_q_rd_end) != Queue:0) ++ ((sel_q_rd & sel_q_rd_end) != Queue:0);
        let wr_resp_ch = ((sel_q_wr_hi & sel_q_wr_end) != Queue:0) ++ ((sel_q_wr & sel_q_wr_end) != Queue:0);

        // receive responses from output channel 0
        let (rdtok1_2, rd_resp0, rd_resp0_valid) =
            recv_if_non_blocking(rdtok0, rd_resp0_r, rd_resp_ch == u2:0, zero!<ReadResp>());
        if rd_resp0_valid {
            trace_fmt!("[RamDemux] Received read response on channel 0: {:#x}", rd_resp0);
        } else {};
        let (wrtok1_2, wr_resp0, wr_resp0_valid) =
            recv_if_non_blocking(wrtok0, wr_resp0_r, wr_resp_ch == u2:0, zero!<WriteResp>());
        if wr_resp0_valid {
            trace_fmt!("[RamDemux] Received write response on channel 0: {:#x}", wr_resp0);
        } else {};

        // receive responses from output channel 1
        let (rdtok1_3, rd_resp1, rd_resp1_valid) =
            recv_if_non_blocking(rdtok0, rd_resp1_r, rd_resp_ch == u2:1, zero!<ReadResp>());
        if rd_resp1_valid {
            trace_fmt!("[RamDemux] Received read response on channel 1: {:#x}", rd_resp1);
        } else {};

        let (wrtok1_3, wr_resp1, wr_resp1_valid) =
            recv_if_non_blocking(wrtok0, wr_resp1_r, wr_resp_ch == u2:1, zero!<WriteResp>());
        if wr_resp1_valid {
            trace_fmt!("[RamDemux] Received write response on channel 1: {:#x}", wr_resp1);
        } else {};

        // receive responses from output channel 2
        let (rdtok1_5, rd_resp2, rd_resp2_valid) =
            recv_if_non_blocking(rdtok0, rd_resp2_r, rd_resp_ch == u2:2, zero!<ReadResp>());
        if rd_resp2_valid {
            trace_fmt!("[RamDemux] Received read response on channel 2: {:#x}", rd_resp2);
        } else {};

        let (wrtok1_5, wr_resp2, wr_resp2_valid) =
            recv_if_non_blocking(wrtok0, wr_resp2_r, wr_resp_ch == u2:2, zero!<WriteResp>());
        if wr_resp2_valid {
            trace_fmt!("[RamDemux] Received write response on channel 2: {:#x}", wr_resp2);
        } else {};

        // prepare read output values
        let (rd_resp, rd_resp_valid) = if rd_resp_ch == u2:0 {
            (rd_resp0, rd_resp0_valid)
        } else if rd_resp_ch == u2:1 {
            (rd_resp1, rd_resp1_valid)
        } else {
            (rd_resp2, rd_resp2_valid)
        };

        // prepare write output values
        let (wr_resp, wr_resp_valid) = if wr_resp_ch == u2:0 {
            (wr_resp0, wr_resp0_valid)
        } else if wr_resp_ch == u2:1 {
            (wr_resp1, wr_resp1_valid)
        } else {
            (wr_resp2, wr_resp2_valid)
        };


        let rdtok1 = join(rdtok1_0, rdtok1_1, rdtok1_4, rdtok1_2, rdtok1_3, rdtok1_5);
        let wrtok1 = join(wrtok1_0, wrtok1_1, wrtok1_4, wrtok1_2, wrtok1_3, wrtok1_5);
        let tok1 = join(rdtok1, wrtok1);

        // send responses to input channel
        let rdtok2_0 = send_if(rdtok1, rd_resp_s, rd_resp_valid, rd_resp);
        if rd_resp_valid {
            trace_fmt!("[RamDemux] Sent read response: {:#x}", rd_resp);
        } else {};

        let sel_q_rd_end = if rd_resp_valid { sel_q_rd_end >> u32:1 } else { sel_q_rd_end };

        let wrtok2_0 = send_if(wrtok1, wr_resp_s, wr_resp_valid, wr_resp);
        if wr_resp_valid {
            trace_fmt!("[RamDemux] Sent write response: {:#x}", wr_resp);
        } else {};

        let sel_q_wr_end = if wr_resp_valid { sel_q_wr_end >> u32:1 } else { sel_q_wr_end };

        // handle select (valid selects are 0..2)
        let (tok2, sel, sel_valid) = recv_non_blocking(tok1, sel_req_r, state.sel);
        if sel_valid {
            trace_fmt!("[RamDemux] Received select: {:#x}", sel);
        } else {};

        send_if(tok2, sel_resp_s, sel_valid, ());
        if sel_valid {
            trace_fmt!("[RamDemux] Sent select response");
        } else {};

        RamDemuxState { sel, sel_q_rd, sel_q_wr, sel_q_rd_hi, sel_q_wr_hi, sel_q_rd_end, sel_q_wr_end }
    }
}
