module wisard_tester
/////////////////// EDIT ME /////////////////////
// Number of classes and its log2
#(parameter N_CLASSES = 2, CLASS_WIDTH = 1, CLK_DIV_WITH = 11)
/////////////////////////////////////////////////
(
input clk_125MHz,
input rst,

output clk_out,
output tuple_bit,
output tuple_valid,
output sop,
input prediction_valid,
input [CLASS_WIDTH-1:0] predicted_class,

input [N_CLASSES-1:0] lut_out_t,
input cnt_mux_sel_sw,
output cnt_mux_sel,
output [N_CLASSES-1:0] cnt_mux_ext,

output reg led_running,
output reg led_success,
output reg led_prediction_error,
output reg led_lut_error
);


/////////////////// EDIT ME /////////////////////
// Number of files to load into the memories.
// This number can be smaller than or equal to
// the number of files the ROM memories were generated
// to load. You can choose a smaller value than that
// if you need a faster run, for example.
localparam NUM_FILES = 100;
// Log2 of the number of files generated. Not log2
// of NUM_FILES
localparam FILE_INDEX_WIDTH = 10;
// The same as address size in COIN
localparam TUPLE_WIDTH = 8;
// Ceil(Log2(TUPLE_WIDTH))
localparam BITCNT_WIDTH = 3;
// Number of RAM nodes
localparam NUM_POS = 47;
// Ceil(Log2(NUM_POS))
localparam ROM_ADDR_W = 6;
// Memory data width which is set to 32 for
// convinience. It could be set to much smaller values,
// but the ROM memories should be generated accordingly
localparam ROM_DATA_W = 32;
// Number of cycles between each tuple transmission.
// This value can not be too small otherwise this tester fails
// If you want to make it possible, implement a FIFO for dout_out
localparam INTERVAL_CYCLES = 16;
// Set to 1'b1 if you want to test just the sequential
// part of the circuit
localparam TEST_SEQUENTIAL = 1'b0;
/////////////////////////////////////////////////


localparam [1:0]  INITIAL=0, SENDING=1, INCREMENT=2, DONE=3;
reg [1:0] fsm_state, next_state;

reg [FILE_INDEX_WIDTH-1:0] file_index;
reg [ROM_ADDR_W-1:0] tuple_index;
reg [BITCNT_WIDTH-1:0] bitcnt;
reg [7:0] intcnt;

wire [ROM_DATA_W-1:0] dout_in, dout_lo;
wire [CLASS_WIDTH-1:0] dout_out;
reg [CLASS_WIDTH-1:0] dout_out_1dly;
wire [N_CLASSES-1:0] lut_out_ref_2prev;
reg [N_CLASSES-1:0] lut_out_ref_1prev, lut_out_ref;
reg bitcnt_ena_1dly;
wire lut_out_ena;
reg lut_out_ena_1dly;

wire increment_ena, intcnt_ena, bitcnt_ena, all_done_prev, wrong_prediction, lut_error;
reg all_done;
wire clk, rst_n;
reg [CLK_DIV_WITH-1:0] clk_cnt;

// Clock frequency division
always @ (posedge clk_125MHz) begin
    clk_cnt <= clk_cnt + 1;
end

assign clk = clk_cnt[CLK_DIV_WITH-1];
assign clk_out = clk;
assign rst_n = ~rst;

// File and tuple index increments
always @ (posedge clk or negedge rst_n) begin
   if (~rst_n) begin
      file_index <= 0;
      tuple_index <= 0;
   end
   else if(increment_ena) begin
      if (tuple_index<NUM_POS-1)
         tuple_index <= tuple_index + 1;
      else begin
         tuple_index <= 0;
         file_index <= file_index + 1;
      end
   end
end

// Bit index
always @ (posedge clk or negedge rst_n) begin
   if (~rst_n)
      bitcnt <= 0;
   else if(bitcnt_ena & ~increment_ena)
      bitcnt <= bitcnt + 1;
   else if (increment_ena)
      bitcnt <= 0;
end

// Interval counter
always @ (posedge clk or negedge rst_n) begin
   if (~rst_n)
      intcnt <= 0;
   else if(intcnt_ena)
      intcnt <= intcnt + 1;
   else
      intcnt <= 0;
end

// Next state logic
always @ (*) begin
   if (fsm_state==INITIAL)
      next_state = SENDING;
   else if (fsm_state==SENDING)
      next_state = all_done ? DONE : bitcnt==TUPLE_WIDTH-1 ? INCREMENT : fsm_state;
   else if (fsm_state==INCREMENT)
      next_state = all_done ? DONE : intcnt==0 ? SENDING : fsm_state;
   else
      next_state = fsm_state;
end

// FSM
always @ (posedge clk or negedge rst_n) begin
   if (~rst_n) begin
      fsm_state <= 0;
   end
   else begin
      fsm_state <= next_state;
   end
end


// Important flags
assign bitcnt_ena = fsm_state==SENDING ? 1'b1 : 1'b0;
assign increment_ena = fsm_state==SENDING && next_state==INCREMENT ? 1'b1 : 1'b0;
assign intcnt_ena = increment_ena || (intcnt>0 && intcnt<INTERVAL_CYCLES) ? 1'b1 : 1'b0;
assign all_done_prev = increment_ena && tuple_index==NUM_POS-1 && file_index==NUM_FILES-1 ? 1'b1 : 1'b0;

// Set the output bit to 0 in case only the sequential is being tested.
assign tuple_bit = cnt_mux_sel ? 1'b0 : dout_in[bitcnt];
assign tuple_valid = bitcnt_ena && bitcnt==0 ? 1'b1 : 1'b0;
assign sop = tuple_valid && tuple_index==0 ? 1'b1 : 1'b0;

assign wrong_prediction = (prediction_valid && dout_out_1dly!=predicted_class) ? 1'b1 : 1'b0;


// Leds and some other signals
always @ (posedge clk or negedge rst_n) begin
   if (~rst_n) begin
      all_done <= 0;
      bitcnt_ena_1dly <= 1'b0;
      led_prediction_error <= 0;
      led_lut_error <= 0;
   end
   else begin
      all_done <= all_done | all_done_prev;
      bitcnt_ena_1dly <= bitcnt_ena;
      led_prediction_error <= led_prediction_error | wrong_prediction;
      led_lut_error <= led_lut_error | lut_error;
   end
end

assign led_running = ~all_done;
assign led_success = all_done & ~led_lut_error & ~led_prediction_error;

// Memories instantiation
rom_tester_in  #(.D_WID(ROM_DATA_W), .NUM_RAMS(1024),.INDEX_WID(FILE_INDEX_WIDTH),.NUM_POS(NUM_POS),.A_WID(ROM_ADDR_W)) rom_tester_in_u0
( .clk(clk),
 .index(file_index),
 .addr(tuple_index),
 .dout(dout_in));

rom_tester_lo  #(.D_WID(ROM_DATA_W), .NUM_RAMS(1024),.INDEX_WID(FILE_INDEX_WIDTH),.NUM_POS(NUM_POS),.A_WID(ROM_ADDR_W)) rom_tester_lo_u0
( .clk(clk),
 .index(file_index),
 .addr(tuple_index),
 .dout(dout_lo));

rom_tester_out
#(.D_WID(CLASS_WIDTH), .NUM_POS(NUM_FILES),.A_WID(FILE_INDEX_WIDTH))
rom_tester_out_u0
(.clk(clk),
 .addr(file_index),
 .dout(dout_out));



 // Tests for just combinational blocks of just sequential blocks
assign cnt_mux_sel = TEST_SEQUENTIAL;
assign lut_out_ref_2prev = dout_lo[N_CLASSES-1:0];
assign lut_out_ena = ~bitcnt_ena & bitcnt_ena_1dly;
assign lut_error = ~cnt_mux_sel && lut_out_ena_1dly &&  lut_out_ref!=lut_out_t ? 1'b1 : 1'b0;
assign cnt_mux_ext = cnt_mux_sel ? lut_out_ref : 0;

 // Useful delays
always @ (posedge clk or negedge rst_n) begin
   if (~rst_n) begin
      dout_out_1dly <= 0;
      lut_out_ref_1prev <= 0;
      lut_out_ref <= 0;
      lut_out_ena_1dly <= 0;
   end
   else begin
      lut_out_ena_1dly <= lut_out_ena;
      dout_out_1dly <= sop ? dout_out : dout_out_1dly;
      lut_out_ref_1prev <= lut_out_ref_2prev;
      lut_out_ref <= lut_out_ref_1prev;
   end
end



endmodule
