
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
//model metadata A(BSIZE,LL)*B(LL,LN)
//batchsize:BSIZE last-layer dimension:LL next-layer dimension:LN
//**MM cores:In(o^2,Cout)*W(Cout,Cin)** K^2 cores
//layer 2: convtranspose2d(256,128,k=5,s=2)H->O 16->8
#define BSIZE 1
#define cci 64
#define ccin (BSIZE*cci)
const int K = 5;
const int S = 2;
#define H 20 //fix it as the largest H to define scratchpad size[need to make it=(actual H)+K-1=16+5-1=20]
const int O2 = 64; //mkae it multiple of Pa ((H-K)/S+1)^2 9*9=81->88, 49->56, 1->4(FC)
const int O = 8;
const int Cout =256;
// const int Cin = 32; //before using 2 slrs
const int Cin = ccin; //is Cin* BSIZE 16*4=64


//layer 3:
// const int BSIZE = 1;
// const int K = 3;
// const int S = 1;
// const int H = 9;
// const int O2 = 52; //((H-K)/S+1)^2 49->52, 1->4(FC)
// const int Cout =32;
// const int Cin = 64;


//hardware parameters
//PE array dimensions
#define P 16
//#define T 16 //before using 2 slrs
#define T 16
// # PxT SAs
#define n 2 //before using 9
// # adders in pad-acc
//#define L 128

const int k_bound=(K*K % n) ? K*K/n + 1 : K*K/n; //how many rounds into n matmulcores

//Aggregation size (for float: typically need to be larger than P,T)
//(for int: can = P,T)
#define Pa 16
// #define Ta 32 //before using 2 slrs
#define Ta 16

// #define P2 4
// #define T2 2
//const int BLOCK_SIZE = 16;
// PE array sizes


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
// typedef struct {
// 	float a[L3];
// } w3blockvec;
//typedef struct {
//	float a[L4];
//} w3blockvec;
//typedef struct {
//	float out[BLOCK_SIZE][BLOCK_SIZE];
//} blockmat;

void loadIn(blockvec_In_P In[],  hls::stream<blockvec_In_P> &Inrows,const int o2,const int co, int it1);
void InBroadcast(hls::stream<blockvec_In_P> &Inrows,hls::stream<blockvec_In_P> Outrows[n],const int co);
void loadW(blockvec_W_T W[], blockvec_W_T Wcols[], const int ci,const int co,int it2,int itk);
void matmulcore(hls::stream<blockvec_In_P> &Inrows, blockvec_W_T Wcols[], hls::stream<blockvec_In_P> &Crows,const int co);
void padacc(hls::stream<blockvec_In_P> Inrows[n],const int S,const int K,const int O,int it1,int it2,int itk,hls::stream<blockvec_In_P> &outpipe);
void storeDDR(blockvec_In_P C[], hls::stream<blockvec_In_P> &outpipe, int it1,int it2);
void sstage1_3(blockvec_In_P *A, blockvec_W_T *B1, blockvec_W_T *B2, int it2,hls::stream<blockvec_In_P> &outpipe);

void top(blockvec_In_P *A, blockvec_W_T *B1, blockvec_W_T *B2, blockvec_In_P *C);
}
