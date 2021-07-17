
#include "hls_stream.h"
// #include "hls_math.h"
#include <iostream>
#include <iomanip>
#include <vector>

extern "C"{
using namespace std;


//model metadata A(BSIZE,LL)*B(LL,LN)
//batchsize:BSIZE last-layer dimension:LL next-layer dimension:LN
//**MM cores:In(o^2,Cout)*W(Cout,Cin)** K^2 cores
//layer 2:
#define BSIZE 16
#define cci 16
#define ccin (BSIZE*cci)
const int K = 4;
const int S = 2;
#define H 24 //fix it as the largest H to define scratchpad size[need to make it=(actual H)+K-1=20+4-1=23]
const int O2 = 88; //mkae it multiple of Pa ((H-K)/S+1)^2 9*9=81->88, 49->56, 1->4(FC)
const int O = 9;
const int Cout =64;
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
#define P 8
//#define T 16 //before using 2 slrs
#define T 8
// # PxT SAs
#define n 6 //before using 9
// # adders in pad-acc
//#define L 128

//Aggregation size (for float: typically need to be larger than P,T)
//(for int: can = P,T)
#define Pa 8
// #define Ta 32 //before using 2 slrs
#define Ta 16

// #define P2 4
// #define T2 2
//const int BLOCK_SIZE = 16;
// PE array sizes


typedef struct {
	float a[Pa];
} blockvec_In_P;
// same for out

typedef struct {
	float a[Ta];
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
// void padacc(hls::stream<blockvec_In_P> Inrows[n], hls::stream<blockvec_In_P> &Outrows,scratchpad[H][H]);
// void storeDDR(blockvec_In_P C[],  hls::stream<blockvec_In_P> &Crows, int it1,int it2);
void padacc(hls::stream<blockvec_In_P> Inrows[n],float scratchpad[H][H][Ta],const int S,const int K,const int O,int it1,int it2,int itk);
void storeDDR(blockvec_In_P C[], float scratchpad[H][H][Ta], int it1,int it2);
// void top(blockvec_In_P *A, blockvec_W_T *B1, blockvec_W_T *B2, blockvec_W_T *B3,
	// blockvec_W_T *B4,blockvec_W_T *B5,blockvec_W_T *B6,blockvec_W_T *B7,
	// blockvec_W_T *B8,blockvec_W_T *B9,blockvec_In_P *C);
void top(blockvec_In_P *A, blockvec_W_T *B1, blockvec_W_T *B2, blockvec_W_T *B3, blockvec_W_T *B4,blockvec_W_T *B5,blockvec_W_T *B6,blockvec_In_P *C);
}
