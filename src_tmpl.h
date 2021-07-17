
#include "hls_stream.h"
#include "ap_int.h"
// #include "hls_math.h"
#include <iostream>
#include <iomanip>
#include <vector>


extern "C"{
using namespace std;
typedef ap_uint<8> dinA_t;
typedef ap_int<16> dout_t;

//MM:In(o^2,Cout)*W(Cout,Cin)** K^2
#define BSIZE 1
#define cci 64
#define ccin (BSIZE*cci)
const int K = 5;
const int S = 2;
#define H 20 //fix it as the largest H needed to define scratchpad size[H=(i)+K-1]
const int O2 = 64; //make it multiple of Pa ((H-K)/S+1)^2 9*9=81->88, 49->56, 1->4(FC)
const int O = 8;
const int Cout =256;
const int Cin = ccin; //if multiple SLRs, divide by the number of SLRs

//Fill in any subsequent layers
//layer n:
// #define cci 64
// #define ccin (BSIZE*cci)
// const int K = 3;
// const int S = 1;
// #define H 12;
// const int O2 = 52; //((H-K)/S+1)^2 49->52, 1->4(FC)
// const int Cout =32;
// const int Cin = ccin;


//hardware parameters
// # PxT SAs
#define P 16
#define T 16
#define n 2 
// # adders in pad-acc
#define L 32

const int k_bound=(K*K % n) ? K*K/n + 1 : K*K/n; 

//Fifo Aggregation size (for float: typically needs to be larger than P,T for best intereval hiding)
//(for int: Pa, Ta = P,T)
#define Pa 32 
#define Ta 16



typedef struct {
	dout_t a[Pa];
} blockvec_In_P;
// same for out

typedef struct {
	int a[Pa];
} blockvec_Out_P;

typedef struct {
	dout_t a[Ta];
} blockvec_W_T;

void loadIn(blockvec_In_P In[],  hls::stream<blockvec_In_P> &Inrows,const int o2,const int co, int it1);
void InBroadcast(hls::stream<blockvec_In_P> &Inrows,hls::stream<blockvec_In_P> Outrows[n],const int co);
void loadW(blockvec_W_T W[], blockvec_W_T Wcols[], const int ci,const int co,int it2,int itk);
void matmulcore(hls::stream<blockvec_In_P> &Inrows, blockvec_W_T Wcols[], hls::stream<blockvec_Out_P> &Crows,const int co);
void padacc(hls::stream<blockvec_Out_P> Inrows[n],const int S,const int K,const int O,int it1,int it2,int itk,hls::stream<blockvec_Out_P> &outpipe);
void storeDDR(blockvec_Out_P C[], hls::stream<blockvec_Out_P> &outpipe, int it1,int it2);
// B_n are the weight interfaces. Customize based on n
void sstage1_3(blockvec_In_P *A, blockvec_W_T *B1, blockvec_W_T *B2, int it2,hls::stream<blockvec_Out_P> &outpipe);
void top(blockvec_In_P *A, blockvec_W_T *B1, blockvec_W_T *B2, blockvec_Out_P *C);

}
