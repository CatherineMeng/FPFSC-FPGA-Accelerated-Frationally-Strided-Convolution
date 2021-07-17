# FPGA-Accelerated Fractionally-Strided Convolution

## Introduction
Fractionally-Strided Convolution (FSC) is a type of upsampling convolution (CONV) operation widely used in various applications (CNN training - back propagation; Auto encoderes + decoders; Generative CNN; etc). A basic arithmetic is to insert and pad zeros into(around) the small input feature map and conduct normal convolution to obtain a larger output feature map:

_N.B.: Blue maps are inputs, and cyan maps are outputs._

<table style="width:100%; table-layout:fixed;">
  <tr>
    <td><img width="100px" src="images/down.gif"></td>
    <td><img width="100px" src="images/up.gif"></td>
  </tr>
  <tr>
    <td>Down-sampling</td>
    <td>Up-sampling</td>
  </tr>
</table>

This repository describes a algorithm-hardware co-designs methodology to avoid the zero computations as shown in the right box of the table, and the accelerator is described in High-Level Synthesis (HLS) for simulation, emulation and implementation on FPGAs.

Specifically, **we break down each kernel into 1x1 sub-kernels and conduct 1x1 CONV for each of them before aggregating together to get the final output feature maps. This way all zero computation can be naturally avoided while producing the correct result.** The high-level accelerator pipeline is shown below:

<!-- ![Image of arch](https://github.com/Anonymous-Author-A/FPGA-FSC/blob/main/images/arch.png) -->
<img src="https://github.com/CatherineMeng/FPFSC-FPGA-Accelerated-Frationally-Strided-Convolution/blob/main/images/arch.png" alt="drawing" width="500"/>

## Software and Hardware Dependencies

[VITIS 2020 Ubuntu](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis.html):
```
./Xilinx_Unified_2020.1_0602_1208_Lin64.bin --noexec --target installer

./installer/xsetup --agree XilinxEULA,3rdPartyEULA,WebTalkTerms --batch Install --config ${HOME}/.Xilinx/install_config.txt
```
[Xilinx XRT](https://www.xilinx.com/products/design-tools/vitis/xrt.html#gettingstarted):
```
./xrt_202110.2.11.634_18.04-amd64-xrt.deb
```
FPGA board and corresponding device support installed on a host CPU

Example using Avleo u200:
```
./xilinx-u200-gen3x16-xdma-all_1-3209015.deb
```

## Usage

### Step 1
To customize your accelerator given a specific FSC layer metadata, start with src_template.cpp and src_tmpl.h. In the header file, the following variables in line 16-25 are relevant:
```
const int K = kernel dimension
const int S = stride size
#define H = dimension of the input(CONV)/output(FSC) feature maps
const int O2 = number of pixels in the output(CONV)/input(FSC) feature maps
const int Cout = number of output(CONV)/input(FSC) feature maps
const int Cin = number of input(CONV)/output(FSC) feature maps
```

### Step 2
Pick the architectural parameters that defines the accelerator setting. We provide a dse script - dse.py - for you to enter FSC metadata and device constraint, so you can check the theoretically optimal design point. Note that larger accelerator may lead to significantly longer synthesis time.

Follow the instructions in the comments and run
```
python dse.py
```

### Step 3
Finish by cusomizing your host code (required for software/hardware simulation and actual deeployment) and configuration file (required for synthesis, place & route).

1. In the header file line 39-52, populate the accelerator parameters gained at step 2:
```
#define P 16
#define T 16
#define n 2 
#define L 32
#define Pa 32
#define Ta 16
```

2. Given the value of n (number of 1x1 CONV Cores), in line 76-77, change the weight interface to the same number. For example, if n=3:
```
void sstage1_3(blockvec_In_P *A, blockvec_W_T *B1, blockvec_W_T *B2, blockvec_W_T *B3, int it2,hls::stream<blockvec_Out_P> &outpipe);
void top(blockvec_In_P *A, blockvec_W_T *B1, blockvec_W_T *B2, blockvec_W_T *B3,blockvec_Out_P *C);
```
3. Finish the configuration file for DDR connection, profiling settings, etc. (example: u200_x_slr.cfg)

Now the program can be synthesized, placed and routed, and used for generating bitstream (relevant commands ->.xo,.xclbin in an example MakeFile)

4. Customize the (testbench) and/or host code

Now the program can be simulated, emulated (with waveform) and run on the device.

## Notes

Example test files for FSC layer in three CNN benchmarks aree included - check the instructions in each folder to get started!


