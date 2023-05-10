module wisard_inbuf
#(parameter ADDRESS_WIDTH=5)
(
input clk,
input rst_n,
input sop,
input sink_valid,
input [ADDRESS_WIDTH-1:0] addr,
output reg sop_buf,
output reg sink_valid_buf,
output reg [ADDRESS_WIDTH-1:0] addr_buf
);

always @ (posedge clk or negedge rst_n) begin
   if (~rst_n) begin
      sop_buf <= 1'b0;
      sink_valid_buf <= 1'b0;
      addr_buf <= 0;
   end
   else begin
      sop_buf <= sop;
      sink_valid_buf <= sink_valid;
      addr_buf <= addr;
   end
end

endmodule
