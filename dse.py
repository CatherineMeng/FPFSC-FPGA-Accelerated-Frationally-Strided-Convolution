import math
import numpy as np
from tqdm import tqdm

# This is an initial for deciding Dataflow Acc Engine
# Customize device constraint
freq=250*10**3 #put your target frequency: cycles/ms
DDR_band=8*10**6 #bytes/ms 
dataWidth=4 #bytes/word
batch=1
DSP_cstr=2265
BRAM_cstr=638
lowerbd=900 #prune away low-utilization samples for faster exploration
# calculate by hand the max possible PE array dimension: #DSPs/DSP-per-PE or sqrt(DSP/DSP-per-PE)
# Square-shaped arrays usually requires fewer IOs
max_PE=64 
max_l=129 #optional
max_n=4 
# Keep the top ql choices (leave flexibility given the capability of the design automation tool)
queue_top=set()
queue_choices={}
ql=20

def DDR_loadin(pa,cout):
    return dataWidth*(cout*pa)/DDR_band

def DDR_loadw(ta,cout):
    return dataWidth*(cout*ta)/DDR_band

def Matmul(pa,ta,p,t,cout):
    return math.ceil(pa/p)*math.ceil(ta/t)*cout/freq

def Pad_Acc(n,h2,cin,ta,l,mode):
    if (mode==1): #default parallelism
        return n*h2*ta/(ta*n*freq) 
    else: #optional: custom parallelism
        return n*h2*ta/(l*freq) 

def DDR_W(h2,ta):
    return dataWidth*h2*ta/DDR_band

def obj_layer_conv(h2,o2,k2,cin,cout,n,pa,ta,p,t,l):
#     print("max phase:",np.argmax([DDR_loadin(pa,cout),Matmul(pa,ta,p,t,cout),Pad_Acc(n,h2,cin,ta)]))
    return (math.ceil(k2/n)*max(DDR_loadw(ta,cout)*n,
                               o2/pa*max(DDR_loadin(pa,cout),Matmul(pa,ta,p,t,cout),Pad_Acc(n,h2,cin,ta,l,1)),
                               DDR_W(h2,ta)))*math.ceil(cin/ta)

# Uncomment the following if you also consider FC layers using the same dataflow acc engine
# def DDR_RFC(a,b,n):
#   return dataWidth*(a*batch+a*b)/DDR_band

# def MatmulFC(a,b,n,p1,p2):
#   return math.ceil(batch/p1)*math.ceil(b/p2)*a/freq

# def DDR_WFC(b):
#   return dataWidth*b*batch/DDR_band

# def obj_layer_FC(a,b,p1,p2,n):
#     return DDR_RFC(a,b,n)+MatmulFC(a,b,n,p1,p2)+DDR_WFC(b)


# def Log2(x):
#     return (math.log10(x) / 
#             math.log10(2));

def isPowerOfTwo(n):
    return math.log(n, 2).is_integer()

l=32# If l is custom, sepecfy l
argmin_p1=0
argmin_p2=0
argmin_n=0
argmin_l=0
sum1m=0
sum_min=100000000
for p in tqdm(range(1,max_PE)):
    for t in range(1,max_PE):
        for n in range(1,max_n):
            sum1m=0
            # assuming num-DSP-per-PE=1. change to 5 if supporting float MAC
            # assuming Pa=p & Ta=t. change accordingly otherwise
            if (isPowerOfTwo(p) and isPowerOfTwo(t) and p*t*n*1+t*1<DSP_cstr and p*t*n*1+t*1>lowerbd and (n+n)*p+n*t+1+1+t<BRAM_cstr):
                # p*t*n*1+Ta*1<DSP_cstr and p*t*n*1+Ta*1>lowerbd and (n+n)*Pa+n*Ta+1+1+Ta<BRAM_cstr):
#                 (h2,o2,k2,cin,cout,n,pa,ta,p,t)
# cout is the dimension being accumulated
# for integers, ta,pa=p,t; for floats, set pa=f*p ot ta=f*t for the best interval-hiding effect
# where f is a reasonable aggregation factor that can be supported be the device
                sum1m+=obj_layer_conv(64,16,25,512,1024,n,p,t,p,t,l)
                sum1m+=obj_layer_conv(256,64,25,256,512,n,p,t,p,t,l)
                sum1m+=obj_layer_conv(256,256,25,128,256,n,p,t,p,t,l)
                # ......Add layers as needed
#                 print("theoretical latency:",sum1m,p,t,n)
#                 print("theoretical bram:",(n+n)*Pa+n*Ta+1+1+Ta,((n+n)*Pa+n*Ta+1+1+Ta)*100/638,"%")
#                 print("theoretical dsp:",p*t*n*3+Ta*3,(p*t*n*3+Ta*3)*100/2265,"%")
#                 print("\n")

            if (sum1m!=0 and sum_min>sum1m):
                # flag=0
                if (len(queue_top)==ql):
                    # flag=1
                    dels=max(queue_top)
                    queue_top.remove(max(queue_top))
                    queue_choices.pop(dels)
                queue_top.add(sum1m) 
                queue_choices[sum1m]=(p,t,n,l)
                sum_min=sum1m
                argmin_p1=p
                argmin_p2=t
                argmin_n=n
                argmin_l=l

print("[My Model] DSE Theretical Opt:","p=",argmin_p1,"t=",argmin_p2,"n=",argmin_n)
print("Theretically lowest latency:",sum_min)
print("Top 20 choices (p,t,n,l):")
for key, value in queue_choices.items():
    print(key, ' : ', value)
print("Top (up to) 20 latencies:",queue_top)

# If the actual synthesis result differs from the estimation (i.e. out of i.o. constraint, etc),
# Pick the next-best choices! The actual performance do not vary too much
