#include "./src_tmpl.h"

extern "C"{
//input tile: cout*ceil(o^2/Pa) bvs in total, cout bvs per tile
// it1: in_tile_index (0=<it2<o^2/Pa-1)
// Assume In bvs in row major order
void loadIn(blockvec_In_P In[],  hls::stream<blockvec_In_P> &Inrows,const int o2,const int co, int it1){
	// int A_tile_index = int(it/(BSIZE/BLOCK_SIZE));
	#pragma HLS aggregate variable=In
	#pragma HLS aggregate variable=Inrows
	ReadInLoop:for (int i = 0; i < co; i++){
		#pragma HLS PIPELINE
		Inrows.write(In[it1*co+i]);
	}
}

void InBroadcast(hls::stream<blockvec_In_P> &Inrows,hls::stream<blockvec_In_P> Outrows[n],const int co)
{ 
    InBDcastLoop:for(int i = 0; i < co; i++){
    	#pragma HLS PIPELINE
    	blockvec_In_P tmp=Inrows.read();
    	for(int j = 0; j < n; j++){
        	Outrows[j].write(tmp);
    	}
    }
}

//weight tile: cout*ceil(cin/Ta) bvs in total, cout bvs per tile
// it2: w_tile_index (0=<it2<cin/Ta-1)
// Assume W bvs in col major order
void loadW(blockvec_W_T W[], blockvec_W_T Wcols[], const int ci,const int co,int it2,int itk){
//	int B_tile_index = it%(BSIZE/BLOCK_SIZE);
	#pragma HLS aggregate variable=W
	#pragma HLS aggregate variable=Wcols
	ReadWLoop:for (int i = 0; i < co; i++){
		#pragma HLS PIPELINE
		// Wcols.write(W[it2*co+i]);
		// Wcols[i]=(W[it2*co+i]);
		Wcols[i]=(W[itk*co*ci/Ta+it2*co+i]); //itk used to see which round (<K^2/n)
	}
}

//Inrows: co blockvecs (each size Pa)
//Wcols: co wblockvecs (each size Ta)
//Crows: Ta blockvecs (each size Pa)
//input fmap: [o^2,Cout] broadcast
//weights: [Cout,Cin]
void matmulcore(hls::stream<blockvec_In_P> &Inrows, blockvec_W_T Wcols[], hls::stream<blockvec_Out_P> &Crows,const int co) {
#pragma HLS aggregate variable=Inrows
#pragma HLS aggregate variable=Wcols
#pragma HLS aggregate variable=Crows
	int C[Pa/P][32/T][P][T]; 
	#pragma HLS bind_storage variable=C type=RAM_2P impl=LUTRAM
	#pragma HLS ARRAY_PARTITION variable=C dim=3 complete
	#pragma HLS ARRAY_PARTITION variable=C dim=4 complete

	blockvec_In_P tempA;
	blockvec_W_T tempB;
	int i=0;
	int j=0;
	int k=0;
   	#pragma HLS aggregate variable=tempA
    #pragma HLS aggregate variable=tempB
	#pragma HLS dependence array variable=C inter false
	// for(int yy = 0; yy < LL*BSIZE/P*LN/T; yy++){
	for(int yy = 0; yy < co*Pa/P*Ta/T; yy++){ //outer i,j,k loops
	#pragma HLS PIPELINE II = 1
		#pragma HLS dependence array variable=C inter false
		j = yy%(Ta/T);
		i = (yy/(Ta/T))%(Pa/P);
		k = yy/(Ta/T*Pa/P);

		if( i == 0 && j==0){ //# occurrences = co (original k loop)
			tempA = Inrows.read();
			printf("MM - Inrows read %d \n",k);
			tempB = Wcols[k];
		}
			

		for(int ii = 0; ii < P; ii++) {
			// #pragma HLS UNROLL
			for(int jj = 0; jj < T; jj++) {
				// #pragma HLS UNROLL
				#pragma HLS dependence variable=C inter false
				dout_t c1;
				#pragma HLS BIND_OP variable=c1 op=mul impl=dsp latency=2
				c1=tempA.a[i*P+ii] * tempB.a[j*T+jj];
				int c1_int=c1;
				C[i][j][ii][jj] = (k==0)? 0: C[i][j][ii][jj] + c1_int;
			}
		}
	}

	//write out to stream
	for(int j = 0; j < Ta/T; j++) {
		for(int jj = 0; jj < T; jj++) {
   		#pragma HLS PIPELINE
			blockvec_Out_P tempC;
			#pragma HLS aggregate variable=tempC
			for(int i = 0; i < Pa/P; i++) {
				for(int ii = 0; ii < P; ii++) {
					int tmp_c=C[i][j][ii][jj];
					// =(tmp_c>0)?tmp_c:0;
					// tempC.a[i*P+ii]=C[i][j][ii][jj];
					//relu activation implemented

					tempC.a[i*P+ii]=(tmp_c>0)?tmp_c: 0;
				}
			}
			Crows.write(tempC);
		}
	}
}


//relu implemented
// void activation(hls::stream<blockvec_In_P> &Inrows, hls::stream<blockvec_In_P> &Outrows){
// 	for (int i = 0; i < Ta; i++){
// 		#pragma HLS PIPELINE
// 		blockvec_In_P temp = Inrows.read();
// 		blockvec_In_P temp_out;
// 		for (int j = 0; j < Pa; j++){
// 			#pragma HLS UNROLL
// 			temp_out.a[j]=(temp.a[j]>0)?temp.a[j]:0;
// 			// temp.a[j]=tmp;
// 		}
// 		Outrows.write(temp_out);
// 	}
// }

//pad-acc module
//Inrows: o^2*cin (need to output H^2*cin) use (H)^2 BRAM banks H is largest H in the network? may use uram
//each Inrow stream has Ta bvs
//o^2:81,49,etc
//kernel 0 output(O^2 pixel-bars(each bar size Ta)): bar[ii][jj] goes to scratchpad[K-1+ii*S][K-1+jj*S]
//kernel 1 output(O^2 pixel-bars(each bar size Ta)): bar[ii][jj] goes to scratchpad[K-2+ii*S][K-1+jj*S]
// ...
//kernel y=(a*K+b) {we can index by a,b where a=y/K,b=y%k} 
// kernel y output bar[ii][jj] goes to scratchpad[K-1-a+ii*S][K-1-b+jj*S] 
// a,b ranges:0~K-1
//0<=ii,jj<O, cut by Pa & identified by Pa&it1. ii*O+jj=it1*Pa+kk
void padacc(hls::stream<blockvec_Out_P> Inrows[n],const int S,const int K,const int O,int it1,int it2,int itk,hls::stream<blockvec_Out_P> &outpipe){
#pragma HLS aggregate variable=Inrows
	blockvec_Out_P tempO2buffer[n][Ta];
	blockvec_Out_P tmp[n];
	#pragma HLS ARRAY_PARTITION variable=tempO2buffer dim=0 complete
	#pragma HLS ARRAY_PARTITION variable=tmp dim=0 complete
	#pragma HLS aggregate variable=tempO2buffer
	#pragma HLS aggregate variable=tmp
	int scratchpad[H][H][Ta]; //Cin=some number times Ta
	#pragma HLS bind_storage variable=scratchpad type=RAM_2P impl=bram
 // #pragma HLS RESOURCE variable=scratchpad core=XPM_MEMORY uram
	//#pragma HLS ARRAY_PARTITION variable=scratchpad dim=1 complete
	#pragma HLS ARRAY_PARTITION variable=scratchpad dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=scratchpad dim=3 complete
	// #pragma HLS ARRAY_PARTITION variable=tempO2buffer dim=2 complete
	int y=0; //y: which 1x1 kernel output is this, out of all K^2=itk*n
	int yy=0; //yy: which pixel-bar is this, out of all O^2 bars in this 1x1 kernel output. yy->ii,jj
	int a=0;
	int b=0;
	int ii=0;
	int jj=0;
	PadAccLoadLoop:for (int j = 0; j < Ta; j++){
		#pragma HLS PIPELINE
		for (int i = 0; i < n; i++){
			#pragma HLS UNROLL
			tmp[i]=Inrows[i].read();
			tempO2buffer[i][j] = tmp[i];
		}
	}
	PadAccNLoop:for (int i = 0; i < n; i++){ //each 1by1 kernel should find corresponding (different but may overlapping) index in sctratchpad 

		a=(itk*n+i)/K;
		b=(itk*n+i)%K;
		// #pragma HLS PIPELINE
		// #pragma HLS dependence array variable=scratchpad intra false
		PadAccPaLoop: for (int kk = 0; kk < Pa; kk++){ //each output pixel should find corresponding (different) index in sctratchpad
			// int y=itk*8+it1*Pa+kk; //y: which 1x1 kernel is this, out of all K^2=itk*n
			#pragma HLS PIPELINE II=1
			ii=(it1*Pa+kk)/O;
			jj=(it1*Pa+kk)%O;
			// #pragma HLS PIPELINE 
			#pragma HLS dependence variable=scratchpad inter false
			// #pragma HLS dependence array variable=scratchpad intra false
			PadAccUnrollLoop:for (int j = 0; j < Ta; j++){ //Ta->which Ta in Cin
				#pragma HLS dependence array variable=scratchpad inter false
				int ind1=K-1-a+ii*S;
				int ind2=K-1-b+jj*S;
				scratchpad[ind1][ind2][j] = (it1==0 && itk==0) ? 0:scratchpad[ind1][ind2][j]+tempO2buffer[i][j].a[kk];
			}
		}
	}
	if (itk==k_bound-1 && it1==O2/Pa-1){
		// storeDDR(C, scratchpad, 0,it2); //it1 not useful
		outPipeH1Loop:for (int j = 0; j < H; j++){
			#pragma HLS pipeline II=1
			blockvec_Out_P tmp [Ta*H/Pa]; //total size=(H/Pa)*Pa*Ta=H*Ta
			#pragma HLS ARRAY_PARTITION variable=tmp dim=1 complete
			#pragma HLS bind_storage variable=tmp type=RAM_1P impl=uram
			for (int i = 0; i < Ta; i++){
				for (int k = 0; k < H; k++){
					// #pragma HLS UNROLL
					int i1=k/Pa;
					int i2=k%Pa;
					tmp[i*H/Pa+i1].a[i2]=scratchpad[j][k][i];
				}
			}
			outPipeLoop:for (int i = 0; i < Ta*H/Pa; i++){
				#pragma HLS pipeline II=1
				outpipe.write(tmp[i]);
			}
		}
	}
}

void storeDDR(blockvec_Out_P C[], hls::stream<blockvec_Out_P> &outpipe, int it1,int it2){
#pragma HLS aggregate variable=C
	int hpal=(H/Pa<1)?1:H/Pa;
	StoreDDRTaLoop:for (int i = 0; i < Ta; i++){
		for (int j = 0; j < H; j++){
			for (int kk = 0; kk < hpal; kk++){
				#pragma HLS PIPELINE
				C[(j*H/Pa+kk)*Cin+it2*Ta+i] = outpipe.read();
			}
		}
	}
}

void sstage1_3(blockvec_In_P *A, blockvec_W_T *B1, blockvec_W_T *B2, int it2,hls::stream<blockvec_Out_P> &outpipe){
	hls::stream<blockvec_In_P> inpipe;	
	hls::stream<blockvec_In_P> inpipe_bd[n];
	
	blockvec_W_T wpipes[n][Cout]; //n=9 now 3
	#pragma HLS ARRAY_PARTITION variable=wpipes dim=1 complete
	#pragma HLS bind_storage variable=wpipes type=RAM_1P impl=uram

	hls::stream<blockvec_Out_P> crowspipe[n];
	// hls::stream<blockvec_In_P> outpipe;

	#pragma HLS STREAM variable=crowspipe depth=64
	#pragma HLS bind_storage variable=crowspipe type=FIFO impl=bram
	#pragma HLS STREAM variable=inpipe depth=32 
	#pragma HLS bind_storage variable=inpipe type=FIFO impl=bram
	#pragma HLS STREAM variable=inpipe_bd depth=32
	#pragma HLS bind_storage variable=inpipe_bd type=FIFO impl=bram

	KLoop: for (int itk =0; itk<k_bound;itk++){ //k_bound rounds into n matmulcores
		#pragma HLS DATAFLOW
		loadW(B1, wpipes[0], Cin, Cout,it2,itk);
		loadW(B2, wpipes[1], Cin, Cout,it2,itk);
		// ...
		#ifndef __SYNTHESIS__
		printf("in itk=%d\n",itk);
		#endif
		int o2lc=(O2/Pa<1)?1:O2/Pa;
		O2Loop:for (int it1=0;it1<o2lc;it1++){ //output-channel-wise parallelism
			#pragma HLS DATAFLOW
			loadIn(A, inpipe,O2,Cout,it1); //repeat Cin/Pa times, usually small
			InBroadcast(inpipe,inpipe_bd,Cout);
			nUnrollLoop:for (int i=0; i<n; i++){ 
				#pragma HLS UNROLL
				matmulcore(inpipe_bd[i], wpipes[i], crowspipe[i],Cout); 
			}
			padacc(crowspipe,S,K,O,it1,it2,itk,outpipe); //to be called k^2*o^2/Pa times
		}
	}
}

void top(blockvec_In_P *A, blockvec_W_T *B1, blockvec_W_T *B2, blockvec_Out_P *C){

	#pragma HLS INTERFACE m_axi port=A bundle=gmem0 offset=slave
	#pragma HLS INTERFACE m_axi port=B1 bundle=gmem1 offset=slave
	#pragma HLS INTERFACE m_axi port=B2 bundle=gmem2 offset=slave
	// ...
	#pragma HLS INTERFACE s_axilite port=A bundle=control
	#pragma HLS INTERFACE s_axilite port=B1 bundle=control
	#pragma HLS INTERFACE s_axilite port=B2 bundle=control
	// ...
	#pragma HLS INTERFACE s_axilite port=C bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control

	hls::stream<blockvec_Out_P> outpipe;
	#pragma HLS STREAM variable=outpipe depth=96 //Ta <-Ta*H*H/Pa
	#pragma HLS bind_storage variable=outpipe type=FIFO impl=bram

	// it1:o^2/Pa it2:cin/Ta
	#ifndef __SYNTHESIS__
	printf("declared.We got in kernel\n");
	#endif
	
	#ifndef __SYNTHESIS__
	printf("k_bond=%d\n",k_bound);
	#endif
	CinBatchLoop: for (int it2=0;it2<Cin/Ta;it2++) { //output-channel-wise parallelism Cin/Ta:layer1=1->it2 can only be 0
		
		#pragma HLS DATAFLOW

		sstage1_3(A, B1, B2,it2,outpipe);
		#ifndef __SYNTHESIS__
		printf("loaded W: it2=%d\n",it2);
		#endif
		//================================End sstage1_3==========================================
		//now padacc is called k^2*o^2/Pa times, filled h^2*Ta slots in the scratchpad, need to store back & empty it
		//storeDDR called Cin/Ta times to fill all h^2*Cin output(input in FW) fmaps
		storeDDR(C, outpipe, 0,it2); //it1 not useful
		#ifndef __SYNTHESIS__
		printf("finished StoreDDR\n");
		#endif
	}
}
}



