#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <vector>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <CL/cl2.hpp>
#include "./block_newstore.h"


#define In_single_SIZE 704 //c_out*(O^2/size(blockvec_In_P))
#define In_SIZE (In_single_SIZE*BSIZE)
#define W_SIZE 192 //c_out*(c_in/blockvec_W_T)*(K^2/n)
#define OUT_single_SIZE 1152 //c_in* (H*H/size(blockvec_Out_P))
#define OUT_SIZE (OUT_single_SIZE*BSIZE)

// Forward declaration of utility functions included at the end of this file
std::vector<cl::Device> get_xilinx_devices();
char *read_binary_file(const std::string &xclbin_file_name, unsigned &nb);

// HBM Pseudo-channel(PC) requirements
#define MAX_HBM_PC_COUNT 16
#define PC_NAME(n) n | XCL_MEM_TOPOLOGY
const int pc[MAX_HBM_PC_COUNT] = {
    PC_NAME(0),  PC_NAME(1),  PC_NAME(2),  PC_NAME(3),  PC_NAME(4),  PC_NAME(5),  PC_NAME(6),  PC_NAME(7),
    PC_NAME(8),  PC_NAME(9),  PC_NAME(10), PC_NAME(11), PC_NAME(12), PC_NAME(13), PC_NAME(14), PC_NAME(15)};

template <typename T1>
struct aligned_allocator
{
  using value_type = T1;
  T1* allocate(std::size_t num)
  {
    void* ptr = nullptr;
    if (posix_memalign(&ptr,4096,num*sizeof(T1)))
      throw std::bad_alloc();
    return reinterpret_cast<T1*>(ptr);
  }
  void deallocate(T1* p, std::size_t num)
  {
    free(p);
  }
};

// ------------------------------------------------------------------------------------
// Main program
// ------------------------------------------------------------------------------------
int main(int argc, char **argv)
{
    // ------------------------------------------------------------------------------------
    // Step 1: Initialize the OpenCL environment
    // ------------------------------------------------------------------------------------
    cl_int err;
    std::string binaryFile = (argc != 2) ? "top.xclbin" : argv[1];
    unsigned fileBufSize;
    std::vector<cl::Device> devices = get_xilinx_devices();
    devices.resize(1);
    cl::Device device = devices[0];
    cl::Context context(device, NULL, NULL, NULL, &err);
    char *fileBuf = read_binary_file(binaryFile, fileBufSize);
    cl::Program::Binaries bins{{fileBuf, fileBufSize}};
    cl::Program program(context, devices, bins, NULL, &err);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    //cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    // cl::Kernel krnl_k2r(program, "top", &err); //comment out===========
    std::vector<cl::Kernel> krnl_k2r(2);
    for (int k = 0; k< 2; k++) {krnl_k2r[k] = cl::Kernel(program,  "top", &err);}

    // ------------------------------------------------------------------------------------
    // Step 2: Create buffers and initialize test values
    // ------------------------------------------------------------------------------------
    //DATA_SIZE=BATCH SIZE
    // The following is for 2 SLRs, 3 weight ports. copy/uncomment/delete as needed
    std::vector<blockvec_In_P, aligned_allocator<blockvec_In_P>> In_rows [2];
    for (int k = 0; k< 2; k++) {In_rows[k].resize(In_SIZE);}
    std::vector<blockvec_W_T, aligned_allocator<blockvec_W_T>> W_cols1 [2];
    for (int k = 0; k< 2; k++) {W_cols1[k].resize(W_SIZE);}
    std::vector<blockvec_W_T, aligned_allocator<blockvec_W_T>> W_cols2 [2];
    for (int k = 0; k< 2; k++) {W_cols2[k].resize(W_SIZE);}
    std::vector<blockvec_W_T, aligned_allocator<blockvec_W_T>> W_cols3 [2];
    for (int k = 0; k< 2; k++) {W_cols3[k].resize(W_SIZE);}
    // std::vector<blockvec_W_T, aligned_allocator<blockvec_W_T>> W_cols4 [2];
    // for (int k = 0; k< 2; k++) {W_cols4[k].resize(W_SIZE);}
    // std::vector<blockvec_W_T, aligned_allocator<blockvec_W_T>> W_cols5 [2];
    // for (int k = 0; k< 2; k++) {W_cols5[k].resize(W_SIZE);}
    // std::vector<blockvec_W_T, aligned_allocator<blockvec_W_T>> W_cols6 [2];
    // for (int k = 0; k< 2; k++) {W_cols6[k].resize(W_SIZE);}
    // std::vector<blockvec_W_T, aligned_allocator<blockvec_W_T>> W_cols7 [2];
    // for (int k = 0; k< 2; k++) {W_cols7[k].resize(W_SIZE);}
    // std::vector<blockvec_W_T, aligned_allocator<blockvec_W_T>> W_cols8 [2];
    // for (int k = 0; k< 2; k++) {W_cols8[k].resize(W_SIZE);}
    // std::vector<blockvec_W_T, aligned_allocator<blockvec_W_T>> W_cols9 [2];
    // for (int k = 0; k< 2; k++) {W_cols9[k].resize(W_SIZE);}
    // std::vector<blockvec_In_P> C_rows(OUT_SIZE);
    std::vector<blockvec_In_P,aligned_allocator<blockvec_In_P>> C_rows [2];
    for (int k = 0; k< 2; k++) {C_rows[k].resize(OUT_SIZE);}

    
    int i, j, k;
    std::cout << "init In_ini matrix." << std::endl;
    // Testbench place holder. customize based on dataset

    // Custom data initialization & check 
    FILE *fp1;
    fp1=fopen("./In_rows.dat","w");
    for (i = 0; i < In_SIZE; i++) {
        for (j = 0; j < Pa; j++) {
            fprintf(fp1,"%f ",In_rows[0][i].a[j]);
        }
        fprintf(fp1,"\n");
    }
    fclose(fp1);
    
    std::cout << "init W_ini matrix." << std::endl;
    // Testbench place holder. customize based on load model

    FILE *fp2;
    fp2=fopen("./W_rows.dat","w");

    for (i = 0; i < W_SIZE; i++) { //W_SIZE=Cout*Cin/Ta
        for (j = 0; j < Ta; j++) {
            fprintf(fp2,"%f ",W_cols1[0][i].a[j]);
        }
        fprintf(fp2,"\n");
    }
    fclose(fp2);

    std::cout << "init Crows." << std::endl;
    for (k = 0; k < 2; k++) {
    for (i = 0; i < OUT_SIZE; i++) { //OUT_SIZE=Cin*(H*H/Pa) H is original H, not declared H (which is bigger)!
        for (j = 0; j < Pa; j++) {
            C_rows[k][i].a[j] = 0;
        }
    }
    }
  printf("inied\n");
    
    cl_mem_ext_ptr_t InrExt [2];
    cl_mem_ext_ptr_t InrExt1 [2];
    cl_mem_ext_ptr_t InrExt2 [2];
    cl_mem_ext_ptr_t InrExt3 [2];
    cl_mem_ext_ptr_t CrExt [2];
    

    InrExt[0].obj = In_rows[0].data();
    InrExt[0].param = 0;
    InrExt[0].flags = 0|XCL_MEM_TOPOLOGY;

    CrExt[0].obj = C_rows[0].data();
    CrExt[0].param = 0;
    CrExt[0].flags = 0|XCL_MEM_TOPOLOGY;

    InrExt1[0].obj = W_cols1[0].data();
    InrExt1[0].param = 0;
    InrExt1[0].flags = 0|XCL_MEM_TOPOLOGY;

    InrExt2[0].obj = W_cols2[0].data();
    InrExt2[0].param = 0;
    InrExt2[0].flags = 0|XCL_MEM_TOPOLOGY;

    InrExt3[0].obj = W_cols3[0].data();
    InrExt3[0].param = 0;
    InrExt3[0].flags = 0|XCL_MEM_TOPOLOGY;


    // Create the buffers and allocate memory
    std::vector<cl::Buffer> in1_buf(2);
    std::vector<cl::Buffer> w1_buf(2);
    std::vector<cl::Buffer> w2_buf(2);
    std::vector<cl::Buffer> w3_buf(2)
    std::vector<cl::Buffer> out_buf(2);
    for (k = 0; k < 2; k++) {
    in1_buf[k] = cl::Buffer(context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(blockvec_In_P) * In_SIZE, &InrExt[k], &err);
    // printf("buf1\n");
    w1_buf[k] = cl::Buffer(context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(blockvec_W_T) * W_SIZE, &InrExt1[k], &err);
    // printf("buf2\n");
    w2_buf[k] = cl::Buffer(context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(blockvec_W_T) * W_SIZE, &InrExt2[k], &err);
    // printf("buf3\n");
    w3_buf[k] = cl::Buffer(context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(blockvec_W_T) * W_SIZE, &InrExt3[k], &err);
    // printf("buf10\n");
    out_buf[k]=cl::Buffer(context, CL_MEM_USE_HOST_PTR|CL_MEM_WRITE_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(blockvec_In_P) * OUT_SIZE, &CrExt[k], &err);
      // printf("hi\n");
    }
    
    // printf("setArg finished\n");
    // ------------------------------------------------------------------------------------
    // Step 3: Run the kernel
    // ------------------------------------------------------------------------------------
    for (k = 0; k < 2; k++) {
    krnl_k2r[k].setArg(0, in1_buf[k]);
    krnl_k2r[k].setArg(1, w1_buf[k]);
    krnl_k2r[k].setArg(2, w2_buf[k]);
    krnl_k2r[k].setArg(3, w3_buf[k]);
    krnl_k2r[k].setArg(4, out_buf[k]);
    }
    printf("setArg\n");
    // Schedule transfer of inputs to device memory, execution of kernel, and transfer of outputs back to host memory
    for (k = 0; k < 2; k++) {
    q.enqueueMigrateMemObjects({in1_buf[k]}, 0 /* 0 means from host*/);
    q.enqueueMigrateMemObjects({w1_buf[k]}, 0 /* 0 means from host*/);
    q.enqueueMigrateMemObjects({w2_buf[k]}, 0 /* 0 means from host*/);
    q.enqueueMigrateMemObjects({w3_buf[k]}, 0 /* 0 means from host*/);
    }
    // Wait for all scheduled operations to finish
    for (k = 0; k < 2; k++) {
        q.enqueueTask(krnl_k2r[k]);
    }
    q.finish();
    for (k = 0; k < 2; k++) {
        q.enqueueMigrateMemObjects({out_buf[k]}, CL_MIGRATE_MEM_OBJECT_HOST);
    }
    q.finish();
    printf("q.finish\n");
    // ------------------------------------------------------------------------------------
    // Step 4: Check Results and Release Allocated Resources
    // ------------------------------------------------------------------------------------
    bool match = true;
    printf("hi, printing Crows into file now\n");
    FILE *fp;
    fp=fopen("./Crows.dat","w");
    printf("hi\n");
    for (i = 0; i < Cin*(H*H/Pa); i++) {
        for (j = 0; j < Ta; j++) {
            fprintf(fp, "%f ", C_rows[0][j].a[i]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
 
    delete[] fileBuf;

    std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;
    return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}

// ------------------------------------------------------------------------------------
// Utility functions
// ------------------------------------------------------------------------------------
std::vector<cl::Device> get_xilinx_devices()
{
    size_t i;
    cl_int err;
    std::vector<cl::Platform> platforms;
    err = cl::Platform::get(&platforms);
    cl::Platform platform;
    for (i = 0; i < platforms.size(); i++)
    {
        platform = platforms[i];
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>(&err);
        if (platformName == "Xilinx")
        {
            std::cout << "INFO: Found Xilinx Platform" << std::endl;
            break;
        }
    }
    if (i == platforms.size())
    {
        std::cout << "ERROR: Failed to find Xilinx platform" << std::endl;
        exit(EXIT_FAILURE);
    }

    //Getting ACCELERATOR Devices and selecting 1st such device
    std::vector<cl::Device> devices;
    err = platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
    return devices;
}

char *read_binary_file(const std::string &xclbin_file_name, unsigned &nb)
{
    if (access(xclbin_file_name.c_str(), R_OK) != 0)
    {
        printf("ERROR: %s xclbin not available please build\n", xclbin_file_name.c_str());
        exit(EXIT_FAILURE);
    }
    //Loading XCL Bin into char buffer
    std::cout << "INFO: Loading '" << xclbin_file_name << "'\n";
    std::ifstream bin_file(xclbin_file_name.c_str(), std::ifstream::binary);
    bin_file.seekg(0, bin_file.end);
    nb = bin_file.tellg();
    bin_file.seekg(0, bin_file.beg);
    char *buf = new char[nb];
    bin_file.read(buf, nb);
    return buf;
}