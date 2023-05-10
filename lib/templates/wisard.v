//`define WISARD_PARALLEL_INPUT 1
//`define WISARD_DEPLOYMENT 1

module wisard
#(parameter ADDRESS_WIDTH = __ADDRESS_WIDTH__, INDEX_WIDTH=__INDEX_WIDTH__, N_RAMS=__N_RAMS__, N_CLASSES = __N_CLASSES__, CLASS_WIDTH = __CLASS_WIDTH__)
(
`ifndef WISARD_DEPLOYMENT
 output reg [N_CLASSES-1:0] lut_out_t,
 input cnt_mux_sel,
 input [N_CLASSES-1:0] cnt_mux_ext,
 output source_valid_r,
 output [CLASS_WIDTH-1:0] predicted_class_r,
 output clk_div4,
 `endif

 input clk,
 input rst_n,
 input sop,
 input sink_valid,
 `ifdef WISARD_PARALLEL_INPUT
 input [ADDRESS_WIDTH-1:0] addr,
 `else
 input addr,
 `endif
 output source_valid,
 output [CLASS_WIDTH-1:0] predicted_class
 );

wire  [N_CLASSES-1:0] lut_out;
wire [INDEX_WIDTH-1:0] index;
wire eop;
wire sop_buf, sink_valid_buf;
wire [ADDRESS_WIDTH-1:0] addr_buf;

wire [N_CLASSES-1:0] cnt_mux_out;

`ifndef WISARD_DEPLOYMENT
// Test structures for observability and controlability

always @ (posedge clk or negedge rst_n) begin
   if (~rst_n)
      lut_out_t <= 0;
   else if (sink_valid_buf)
      lut_out_t <= lut_out;
end
//assign lut_out_t = lut_out;

assign cnt_mux_out = cnt_mux_sel ? cnt_mux_ext : lut_out;

// Divide clock by 4
reg [1:0] cnt_div;
always @ (posedge clk) begin
   cnt_div <= cnt_div + 2'd1;
end
assign clk_div4 = cnt_div[1];
`else
// When there is no test mux
assign cnt_mux_out = lut_out;
`endif


`ifdef WISARD_PARALLEL_INPUT

// Input buffering
wisard_inbuf #(.ADDRESS_WIDTH(ADDRESS_WIDTH))
wisard_inbuf_u0
(. clk(clk),
.rst_n(rst_n),
 .sop(sop),
 .sink_valid(sink_valid),
 .addr(addr),
 .sop_buf(sop_buf),
 .sink_valid_buf(sink_valid_buf),
 .addr_buf(addr_buf)
 );

`else  // For serial input

wisard_parallel2serial #(.ADDRESS_WIDTH(ADDRESS_WIDTH))
wisard_parallel2serial
(. clk(clk),
.rst_n(rst_n),
 .sop(sop),
 .sink_valid(sink_valid),
 .addr(addr),
 .sop_buf(sop_buf),
 .sink_valid_buf(sink_valid_buf),
 .addr_buf(addr_buf)
 );

`endif

// Index and eop generation
wisard_index #(.INDEX_WIDTH(INDEX_WIDTH),.N_RAMS(N_RAMS))
wisard_index_u0 (
 .clk(clk),
 .rst_n(rst_n),
 .sop(sop_buf),
 .sink_valid(sink_valid_buf),
 .index(index),
 .eop(eop));

// LUT Instantiation
wisard_lut #(.ADDR_WIDTH(ADDRESS_WIDTH), .INDEX_WIDTH(INDEX_WIDTH), .O_WIDTH(N_CLASSES))
wisard_lut_u0
(.addr(addr_buf),.index(index), .out(lut_out));

// Wisard control instantiation
wisard_ctrl
#(.ADDRESS_WIDTH(ADDRESS_WIDTH), .INDEX_WIDTH(INDEX_WIDTH), .N_CLASSES(N_CLASSES), .CLASS_WIDTH(CLASS_WIDTH))
wisard_ctrl_u0
(. clk(clk),
.rst_n(rst_n),
 .sop(sop_buf),
 .sink_valid(sink_valid_buf),
 .eop(eop),
 .lut_out(cnt_mux_out),
 .source_valid(source_valid),
 .class_result(predicted_class));

`ifndef WISARD_DEPLOYMENT
// Redundant instance of control
wisard_ctrl
#(.ADDRESS_WIDTH(ADDRESS_WIDTH), .INDEX_WIDTH(INDEX_WIDTH), .N_CLASSES(N_CLASSES), .CLASS_WIDTH(CLASS_WIDTH))
wisard_ctrl_u1
(. clk(clk),
.rst_n(rst_n),
 .sop(sop_buf),
 .sink_valid(sink_valid_buf),
 .eop(eop),
 .lut_out(cnt_mux_out),
 .source_valid(source_valid_r),
 .class_result(predicted_class_r));
`endif

endmodule

