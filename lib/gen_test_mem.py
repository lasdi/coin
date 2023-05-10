


def gen_rom_tester_in (N):
    txt = "module rom_tester_in #(parameter NUM_RAMS=4, INDEX_WID=2, A_WID=10, NUM_POS=1024, D_WID=32)\n"
    txt += "( input clk, input [INDEX_WID-1:0] index, input [A_WID-1:0] addr, output reg [D_WID-1:0] dout);\n\n"
    txt += "wire [D_WID-1:0] mem [NUM_RAMS-1:0][NUM_POS-1:0];\n\n"
    txt += "assign dout = mem[index][addr];\n\n"
    
    for i in range (N):
        txt += "reg [D_WID-1:0] mem_"+str(i)+" [NUM_POS-1:0];\n"
    txt += "\n"
    for i in range (N):
        txt += "assign mem["+str(i)+"] = mem_"+str(i)+";\n"
    txt += "\n"
    for i in range (N):
        txt += 'initial $readmemh("./data/in'+str(i)+'.txt",mem_'+str(i)+');\n'    
    
    txt += "\nendmodule\n"
    
    text_file = open('../rom_tester_in.sv', "w")
    text_file.write(txt)
    text_file.close()

def gen_rom_tester_lo (N):
    
    txt = "module rom_tester_lo #(parameter NUM_RAMS=4, INDEX_WID=2, A_WID=10, NUM_POS=1024, D_WID=32)\n"
    txt += "( input clk, input [INDEX_WID-1:0] index, input [A_WID-1:0] addr, output reg [D_WID-1:0] dout);\n\n"
    txt += "wire [D_WID-1:0] mem [NUM_RAMS-1:0][NUM_POS-1:0];\n\n"
    txt += "assign dout = mem[index][addr];\n\n"
    
    for i in range (N):
        txt += "reg [D_WID-1:0] mem_"+str(i)+" [NUM_POS-1:0];\n"
    txt += "\n"
    for i in range (N):
        txt += "assign mem["+str(i)+"] = mem_"+str(i)+";\n"
    txt += "\n"
    for i in range (N):
        txt += 'initial $readmemh("./data/lo'+str(i)+'.txt",mem_'+str(i)+');\n'    
    
    txt += "\nendmodule\n"
    
        
    text_file = open('../rom_tester_lo.sv', "w")
    text_file.write(txt)
    text_file.close()
    

if __name__ == "__main__":
    
    N = 1024


    gen_rom_tester_in (N)
    gen_rom_tester_lo (N)
