
module wisard_ctrl
#(parameter ADDRESS_WIDTH = 14, INDEX_WIDTH=5, N_CLASSES = 2, CLASS_WIDTH = 1)
(input clk,
 input rst_n,
 input sop,
 input sink_valid,
 input eop,
 input [N_CLASSES-1:0] lut_out,
 output reg source_valid,
 output reg [CLASS_WIDTH-1:0] class_result);


integer i, j;
reg signed [INDEX_WIDTH:0] class_count [0:N_CLASSES-1];
reg signed [INDEX_WIDTH:0] class_count_buf [0:N_CLASSES-1];
wire eop_valid;
reg eop_1dly, eop_2dly;
reg eop_valid_1dly;
reg [CLASS_WIDTH-1:0] eop_cnt;
reg [CLASS_WIDTH-1:0] class_result_prev;
reg signed [INDEX_WIDTH:0] max_val;
wire source_valid_prev;
wire present;

// Automatically generated parameters
__TAU_PARAMETERS__


// Indicates when the current address is present in the LUT
assign present = |lut_out;

// Counts the hits of each class
always @ (posedge clk or negedge rst_n) begin
  if(!rst_n) begin
     for (i=0; i<N_CLASSES; i=i+1)
       class_count[i] <= {(INDEX_WIDTH+1){1'b0}};
  end
  else if (sink_valid) begin
     for (j=0; j<N_CLASSES; j=j+1) begin
        if (sop)
          class_count[j] <= lut_out[j] ? TAU_PLUS1[j] : present ? TAU_MINUS1[j] : TAU[j];
        else if (lut_out[j])
          class_count[j] <= class_count[j] + 1;
        else if (present)
          class_count[j] <= class_count[j] - 1;
     end
  end
end

// Signaling for highest score search
assign eop_valid = eop_2dly | eop_cnt!={CLASS_WIDTH{1'b0}};

always @ (posedge clk or negedge rst_n) begin
  if(!rst_n) begin
    eop_1dly <= 1'b0;
    eop_2dly <= 1'b0;
    eop_valid_1dly <= 1'b0;
  end
  else begin
    eop_1dly <= eop;
    eop_2dly <= eop_1dly;
    eop_valid_1dly <= eop_valid;
  end
end

always @ (posedge clk or negedge rst_n) begin
   if(!rst_n) begin
     for (i=0; i<N_CLASSES; i=i+1)
       class_count_buf[i] <= {(INDEX_WIDTH+1){1'b0}};
  end
  else if(eop_1dly) begin
     for (i=0; i<N_CLASSES; i=i+1)
       class_count_buf[i] <= class_count[i];
  end
end

always @ (posedge clk or negedge rst_n) begin
  if(!rst_n) begin
    eop_cnt <= {CLASS_WIDTH{1'b0}};
  end
  else if (eop_valid) begin
     if (eop_cnt<N_CLASSES-1)
        eop_cnt <= eop_cnt + 1;
     else
        eop_cnt <= {CLASS_WIDTH{1'b0}};
  end
end

// Highest score search
always @ (posedge clk or negedge rst_n) begin
  if(!rst_n) begin
    class_result_prev <= {CLASS_WIDTH{1'b0}};
    max_val <= {(INDEX_WIDTH+1){1'b0}};
  end
  else if (eop_valid) begin
     if (eop_2dly | class_count_buf[eop_cnt]>max_val) begin
       max_val <= class_count_buf[eop_cnt];
       class_result_prev <= eop_cnt;
     end
  end
end

// Updates the output and rises flag
assign source_valid_prev = ~eop_valid & eop_valid_1dly;

always @ (posedge clk or negedge rst_n) begin
  if(!rst_n)
    source_valid <= 1'b0;
  else
    source_valid <= source_valid_prev;
end

always @ (posedge clk or negedge rst_n) begin
  if(!rst_n)
    class_result <= {CLASS_WIDTH{1'b0}};
  else if(source_valid_prev)
    class_result <= class_result_prev;
end

endmodule
