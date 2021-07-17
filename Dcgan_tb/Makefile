all: mktarget app.exe top.xclbin

mktarget:
	@echo "---Creating Target Directory---"
	mkdir ./${Target}

app.exe: ./host_x_slr.cpp
	@echo "---Compiling Host Program---"
	g++ -Wall -g -std=c++11 ./host_x_slr.cpp -o ./${Target}/app.exe \
		-I${XILINX_XRT}/include/ \
		-I/tools/Xilinx/Vivado/2020.2/include/ \
		-L/tools/Xilinx/Vivado/2020.2/lib/ \
		-L${XILINX_VIVADO}/lib/ \
		-lOpenCL -lpthread -lrt -lstdc++
	
top.xo: ./top_entry_perfimprove_newstore.cpp
	@echo "---Synthesizing Kernel Program---"
	v++ -c -t ${Target} --config ./u200_x_slr.cfg -k top -I./ ./top_entry_perfimprove_newstore.cpp -o ./${Target}/top.xo 

top.xclbin: ./${Target}/top.xo
	@echo "---Kernel Place & Route, Generating Bitstream---"
	v++ -l -t ${Target} --config ./u200_x_slr.cfg ./${Target}/top.xo -o ./${Target}/top.xclbin

clean:
	rm -rf vadd* app.exe *json *csv *log *summary _x .run .Xil .ipcache *.jou ./${Target}

# Unless specified, use the current directory name as the v++ build target
# Target ?= $(notdir $(CURDIR))
Target ?= $(foo)
foo = sw_emu