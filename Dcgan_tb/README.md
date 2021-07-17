# DCGAN
Demonstrating the kn2row acceleration of the generative DCGAN CONV Layers

There are four upsampling convolution layers with uniform stride of 2 in the generative network of DCGAN.

![Image of generative cnn - dcgan](https://github.com/CatherineMeng/FPFSC-FPGA-Accelerated-Frationally-Strided-Convolution/blob/main/images/dcgan.png)

## Software emulation:
```
make all Target=sw_emu
cd sw_emu
export XCL_EMULATION_MODE=sw_emu
./app.exe
```
## Hardware emulation:
```
make all Target=hw_emu
cd hw_emu
export XCL_EMULATION_MODE=hw_emu
./app.exe
```
To enable hardware emulation with real-time waveform inspection and more accurate performance estimation in cycles with approximate DDR access models, add "-g" flag to both of the v++ compilation commands after the string "v++", and add a xrt.ini file in the hw_emu target directory with the following content:
```
[Emulation]
debug_mode=gui
```
## Hardware:
Make sure that the target device is detectable from your machine
```
make all Target=hw
./hw/app.exe
```

## Cleanup and delete all generated directories/files:
```
make clean
```
