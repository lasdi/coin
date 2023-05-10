module rom_tester_out #(parameter A_WID=10, NUM_POS=1024, D_WID=32)
(
 input clk,
 input [A_WID-1:0] addr,
 output reg [D_WID-1:0] dout
);

reg [D_WID-1:0] mem [NUM_POS-1:0];

assign dout = mem[addr];

initial $readmemh("./data/y_pred_sw.txt", mem);

endmodule
