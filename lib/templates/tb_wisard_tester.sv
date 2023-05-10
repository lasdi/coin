 `timescale 10us/100ns

module tb_wisard;

/////////////////// EDIT ME /////////////////////
localparam CLASS_WIDTH = 1;
localparam N_CLASSES = 2;
localparam CNT_MUX_SEL_SW = 1'b0;
localparam CLK_DIV_WITH = 1;
/////////////////////////////////////////////////

// TB signal and variables
reg clk_125MHz;
wire clk;
reg rst;
wire led_running, led_success, led_prediction_error, led_lut_error;
integer i;
wire [N_CLASSES-1:0] lut_out_t, cnt_mux_ext;
wire cnt_mux_sel;

// Clock generation
initial begin
   clk_125MHz = 0;
   while (1) begin
      clk_125MHz = ~clk_125MHz;
      #10;
   end
end


 initial begin
   wisard_tester_u0.clk_cnt = 0;
   rst = 1;
   repeat (10) @(negedge clk_125MHz);
   rst = 0;


   while (led_running) @(negedge clk_125MHz);
   repeat (100) @(negedge clk_125MHz);
   if (led_success)
      $display("VERIFICATION PASSED SUCCESFULY.");
   else begin
      if (led_prediction_error)
         $display("VERIFICATION FAILED AT PREDICTION CHECKING.");
      if (led_lut_error)
         $display("VERIFICATION FAILED AT LUT CHECKING.");
   end
   $finish;
end

wire tuple_bit, tuple_valid, sop;
wire prediction_valid, prediction_valid_r;
wire [CLASS_WIDTH-1:0] predicted_class, predicted_class_r;
wire clk_div4;

wisard_tester
#(.CLK_DIV_WITH(CLK_DIV_WITH))
wisard_tester_u0
( .clk_125MHz(clk_125MHz),
 .rst(rst),
 .clk_out(clk),
 .tuple_bit(tuple_bit),
 .tuple_valid(tuple_valid),
 .sop(sop),
 .prediction_valid(prediction_valid),
 .predicted_class(predicted_class),
 .lut_out_t(lut_out_t),
 .cnt_mux_sel_sw(CNT_MUX_SEL_SW),
 .cnt_mux_sel(cnt_mux_sel),
 .cnt_mux_ext(cnt_mux_ext),
 .led_running(led_running),
 .led_success(led_success),
 .led_prediction_error(led_prediction_error),
 .led_lut_error(led_lut_error)
 );


// Wisard instantiation
wisard
//#(.ADDRESS_WIDTH(ADDRESS_WIDTH), .INDEX_WIDTH(INDEX_WIDTH), .N_RAMS(N_RAMS), .N_CLASSES(N_CLASSES), .CLASS_WIDTH(CLASS_WIDTH))
wisard_u0
(
.lut_out_t(lut_out_t),
.cnt_mux_sel(cnt_mux_sel),
.cnt_mux_ext(cnt_mux_ext),
.source_valid_r(prediction_valid_r),
.predicted_class_r(predicted_class_r),
.clk_div4(clk_div4),

.clk(clk),
.rst_n(~rst),
.sop(sop),
.sink_valid(tuple_valid),
.addr(tuple_bit),
.source_valid(prediction_valid),
.predicted_class(predicted_class)
);


endmodule
