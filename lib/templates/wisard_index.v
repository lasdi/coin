module wisard_index
#(parameter INDEX_WIDTH=5, N_RAMS=27)
(
input clk,
input rst_n,
input sink_valid,
input sop,
output reg [INDEX_WIDTH-1:0] index,
output eop
);


always @ (posedge clk or negedge rst_n) begin
   if (~rst_n) begin
      index <= 0;
   end
   else if(sink_valid) begin
       if (eop)
          index <= 0;
       else if (sop)
          index <= 1;
       else
          index <= index + 1;
   end
end

assign eop = index==N_RAMS-1 && sink_valid? 1'b1 : 1'b0;

endmodule
