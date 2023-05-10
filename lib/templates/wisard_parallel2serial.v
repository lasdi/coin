module wisard_parallel2serial
#(parameter ADDRESS_WIDTH=5)
(
input clk,
input rst_n,
input sop,
input sink_valid,
input addr,
output reg sop_buf,
output reg sink_valid_buf,
output reg [ADDRESS_WIDTH-1:0] addr_buf);

localparam [ADDRESS_WIDTH-1:0] ADDR_ZERO = {ADDRESS_WIDTH{1'b0}};

// It covers until ADDRESS_WIDTH=32
reg [4:0] cnt;
wire frame_valid, sink_valid_buf_prev;
reg sop_buf_prev;

always @ (posedge clk or negedge rst_n) begin
   if (~rst_n) begin
      addr_buf <= ADDR_ZERO;
   end
   else begin
       if (sink_valid)
          addr_buf <= {addr, ADDR_ZERO[ADDRESS_WIDTH-2:0]};
       else if (frame_valid)
          addr_buf <= {addr, addr_buf[ADDRESS_WIDTH-1:1]};
   end
end

always @ (posedge clk or negedge rst_n) begin
   if (~rst_n) begin
      cnt <= 0;
   end
   else begin
       if (sink_valid)
          cnt <= 1;
       else if (cnt>0 && cnt<ADDRESS_WIDTH-1)
          cnt <= cnt + 1;
       else
          cnt <= 0;
   end
end

assign frame_valid = cnt>0 && cnt<=ADDRESS_WIDTH-1 ? 1'b1 : 1'b0;
assign sink_valid_buf_prev = cnt==ADDRESS_WIDTH-1;

always @ (posedge clk or negedge rst_n) begin
   if (~rst_n) begin
      sink_valid_buf <= 1'b0;
      sop_buf_prev <= 1'b0;
      sop_buf <= 1'b0;
   end
   else begin
      sink_valid_buf <= sink_valid_buf_prev;
      sop_buf_prev <= sop ? 1'b1 : sop_buf_prev & sink_valid_buf_prev ? 1'b0 : sop_buf_prev;
      sop_buf <= sop_buf_prev & sink_valid_buf_prev;
   end
end

endmodule
