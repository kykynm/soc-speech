#get_ipython().magic(u'matplotlib inline')
import sys
import numpy as np
import scipy as sp
import scipy.fftpack
import scipy.io.wavfile as wave
import matplotlib.pyplot as plt
import math
from numpy import dot
from lxml import etree
import sklearn.metrics.pairwise as dist
import htk

def framing(a, window, shift=1):
    shape = ((a.shape[0] - window) / shift + 1, window) + a.shape[1:]
    strides = (a.strides[0]*shift,a.strides[0]) + a.strides[1:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def mel_inv(x):
    return (np.exp(x/1127.)-1.)*700.

def mel(x):
    return 1127.*np.log(1. + x/700.)

def preemphasis(x, coef=0.97):
    return x - np.c_[x[..., :1], x[..., :-1]] * coef

def mel_fbank_mx(winlen_nfft, fs, NUMCHANS=20, LOFREQ=0.0, HIFREQ=None,  warp_fn=mel, inv_warp_fn=mel_inv):
    """Returns mel filterbank as an array (NFFT/2+1 x NUMCHANS)
    winlen_nfft - Typically the window length as used in mfcc_htk() call. It is
                  used to determine number of samples for FFT computation (NFFT).
                  If positive, the value (window lenght) is rounded up to the
                  next higher power of two to obtain HTK-compatible NFFT.
                  If negative, NFFT is set to -winlen_nfft. In such case, the
                  parameter nfft in mfcc_htk() call should be set likewise.
    fs          - sampling frequency (Hz, i.e. 1e7/SOURCERATE)
    NUMCHANS    - number of filter bank bands
    LOFREQ      - frequency (Hz) where the first filter strats
    HIFREQ      - frequency (Hz) where the last  filter ends (default fs/2)
    warp_fn     - function for frequency warping and its inverse
    inv_warp_fn - inverse function to warp_fn
    """

    if not HIFREQ: HIFREQ = 0.5 * fs
    nfft = 2**int(np.ceil(np.log2(winlen_nfft))) if winlen_nfft > 0 else -int(winlen_nfft)

    fbin_mel = warp_fn(np.arange(nfft / 2 + 1, dtype=float) * fs / nfft)
    cbin_mel = np.linspace(warp_fn(LOFREQ), warp_fn(HIFREQ), NUMCHANS + 2)
    cind = np.floor(inv_warp_fn(cbin_mel) / fs * nfft).astype(int) + 1
    mfb = np.zeros((len(fbin_mel), NUMCHANS))
    for i in xrange(NUMCHANS):
        mfb[cind[i]  :cind[i+1], i] = (cbin_mel[i]  -fbin_mel[cind[i]  :cind[i+1]]) / (cbin_mel[i]  -cbin_mel[i+1])
        mfb[cind[i+1]:cind[i+2], i] = (cbin_mel[i+2]-fbin_mel[cind[i+1]:cind[i+2]]) / (cbin_mel[i+2]-cbin_mel[i+1])
    if LOFREQ > 0.0 and float(LOFREQ)/fs*nfft+0.5 > cind[0]: mfb[cind[0],:] = 0.0 # Just to be HTK compatible
    return mfb

def fbank_htk(x, window, noverlap, fbank_mx, nfft=None, _E=None,
             USEPOWER=False, RAWENERGY=True, PREEMCOEF=0.97, ZMEANSOURCE=False,
             ENORMALISE=True, ESCALE=0.1, SILFLOOR=50.0, USEHAMMING=True):
    """Mel log Mel-filter bank channel outputs
    Returns NUMCHANS-by-M matrix of log Mel-filter bank outputs extracted form
    signal x, where M is the number of extracted frames, which can be computed
    as floor((length(x)-noverlap)/(window-noverlap)). Remaining parameters
    have the following meaning:
    x         - input signal
    window    - frame window lentgth (in samples, i.e. WINDOWSIZE/SOURCERATE)
                or vector of widow weights override default windowing function
                (see option USEHAMMING)
    noverlap  - overlapping between frames (in samples, i.e window-TARGETRATE/SOURCERATE)
    fbank_mx  - array with (Mel) filter bank (as returned by function mel_fbank_mx()).
                Note that this must be compatible with the parameter 'nfft'.
    nfft      - number of samples for FFT computation. By default, it is  set in the
                HTK-compatible way to the window length rounded up to the next higher
                pover of two.
    _E        - include energy as the "first" or the "last" coefficient of each
                feature vector. The possible values are: "first", "last", None.

    Remaining options have exactly the same meaning as in HTK.

    See also:
      mel_fbank_mx:
          to obtain the matrix for the parameter fbank_mx
      add_deriv:
          for adding delta, double delta, ... coefficients
      add_dither:
          for adding dithering in HTK-like fashion
    """
    from time import time
    tm = time()
    if type(USEPOWER) == bool:
        USEPOWER += 1
    if np.isscalar(window):
        window = np.hamming(window) if USEHAMMING else np.ones(window)
    if nfft is None:
        nfft = 2**int(np.ceil(np.log2(window.size)))
    x = framing(x.astype("float"), window.size, window.size-noverlap).copy()
    if ZMEANSOURCE:
        x -= x.mean(axis=1)[:,np.newaxis]
    if _E is not None and RAWENERGY:
        energy = np.log((x**2).sum(axis=1))
    if PREEMCOEF is not None:
        x = preemphasis(x, PREEMCOEF)
    x *= window
    if _E is not None and not RAWENERGY:
        energy = np.log((x**2).sum(axis=1))
    #x = np.abs(scipy.fftpack.fft(x, nfft))
    #x = x[:,:x.shape[1]/2+1]
    x = np.fft.rfft(x, nfft)
    #x = np.abs(x)
    x = x.real**2 + x.imag**2
    if USEPOWER != 2:
        x **= 0.5 * USEPOWER
    x = np.log(np.maximum(1.0, np.dot(x, fbank_mx)))
    if _E is not None and ENORMALISE:
        energy = (energy - energy.max())       * ESCALE + 1.0
        min_val  = -np.log(10**(SILFLOOR/10.)) * ESCALE + 1.0
        energy[energy < min_val] = min_val

    return np.hstack(([energy[:,np.newaxis]] if _E == "first" else []) + [x] +
                     ([energy[:,np.newaxis]] if (_E in ["last", True])  else []))

def norm_data(name_wav):
    wav = wave.read(name_wav, 'r')[1];   
    fbanks=mel_fbank_mx(256,8000,24,64,3800);
    wav_melbanks=fbank_htk(wav,200,120,fbanks,USEPOWER=2,RAWENERGY=False,PREEMCOEF=0.00)
    sha= np.shape(wav_melbanks)    
    norms=wav_melbanks.sum(axis=0, keepdims=True)
    mean = norms/sha[0]
    wav_melbanks=wav_melbanks-mean;
    return wav_melbanks

def exp_context(context, wav_melbanks):
    fr=0
    frame_len=np.shape(wav_melbanks)[1];
    frame_len_stac=(context*2+1)*frame_len;
    wav_melbanks_stacked=np.zeros((np.shape(wav_melbanks)[0],frame_len_stac));
    frame=np.hstack((np.tile(wav_melbanks[fr,:],context).T, np.reshape(wav_melbanks[fr:(context+1),:],((context+1)*frame_len)).T))
    wav_melbanks_stacked[fr]=np.reshape(np.reshape(frame.T,[context*2+1,frame_len]),-1,order='F')
    for fr in range(1,np.shape(wav_melbanks)[0]):
        frame=np.roll(frame,-frame_len,axis=0)
        pom=fr+context
        if pom >= np.shape(wav_melbanks)[0]:
            pom=np.shape(wav_melbanks)[0]-1
        frame[frame_len_stac-frame_len:frame_len_stac]=wav_melbanks[pom,:]
        wav_melbanks_stacked[fr]=np.reshape(np.reshape(frame.T,[context*2+1,frame_len]),-1,order='F')
        #print fr, pom, frame_len_stac-frame_len ;
    return wav_melbanks_stacked


#upload nn
#filename="NNBN_15fb24_silMax31_cmn_FSNRoabc500h_NN2+0HL300BN30_120_BN.complete_transform_1"
def upload_NN (filename):
    fr=open(filename,'r');
    st=0;
    NN={}    
    arr = (fr.readline()).split();
    #nactu maximum
    if( st == 0 ):
        version_maj = int(arr[0]);
        st = 1;
        
    arr = (fr.readline()).split();
    #nactu minimum
    if( st == 1 ):
        version_min = int(arr[0]);
        st = 2;

    arr = (fr.readline()).split();
    #nactu pocet slozek
    if( st == 2 ):
        num_comp = int(arr[0]);
        st = 3;

    for comp in range(1, num_comp+1):
        arr = (fr.readline()).split();
        #nactu nazvy a popis casti
        if( st == 3 ):
            NNLayer={};
            #v 360 0 NN_mean 1
            comp_type_str = arr[0]; comp_type = 0;
            if( comp_type_str == "v"): comp_type = 1
            if( comp_type_str == "m"): comp_type = 2
            if( comp_type == 0 ): print("ERROR: unknown component type ", comp_type_str);
            comp_len = int(arr[1]);
            comp_size = int(arr[2]);
            comp_name_str = arr[3]; comp_name = 0;
            comp_layer = int(arr[4]);
            if( comp_name_str == "NN_mean"): comp_name = 1
            if( comp_name_str == "NN_var"): comp_name = 2
            if( comp_name_str == "NN_weights"): comp_name = 11
            if( comp_name_str == "NN_bias"): comp_name = 12
            if( comp_name == 0 ): print("ERROR: unknown component name ", comp_name_str);
            NNLayer={"Type": comp_type, "Length": comp_len, "Size": comp_size, "Name": comp_name_str, "Name_ID": comp_name, "Layer": comp_layer}
            st = 4;
        
        if( st == 4 ):
            #nactu casti s jednim radkem
            if( comp_type == 1 ):
                arr = (fr.readline()).split();
                #print st, arr
                data_v = np.zeros(shape=(comp_len,1))
                if(comp_len != len(arr)): print("ERROR: different size of data ", comp_name_str, comp_len, len(arr) );            
                for i in range(0,comp_len):
                    data_v[i]=float(arr[i])                
                NNLayer.update({"Data": data_v})
            #nactu casti s vice radky
            if( comp_type == 2 ):
                data_m = np.zeros(shape=(comp_len,comp_size))
                for i in range(0,comp_len):
                    arr = (fr.readline()).split();
                    if(comp_size != len(arr)): print("ERROR: different size of data ", comp_name_str, comp_len, comp_size, len(arr) );
                    for j in range(0,comp_size):
                        data_m[i,j]=float(arr[j])
                NNLayer.update({"Data": data_m})
            NN.update({comp:NNLayer})
        
        st=3;
    return NN

def sigmoid(x):
      return 1 / (1 + np.exp(-x))
    
def fwd_pass_NN(data, NN):  
    # normalizace -> dato + mean * var
    data = data + NN[1]["Data"].T
    normData = data * NN[2]["Data"].T
    # layer 1
    layer1Data = np.matmul(normData, NN[3]["Data"].T)
    #bias 1.
    layer1Data = layer1Data + NN[4]["Data"].T
    #sigmoid.
    layer1Data = sigmoid(layer1Data)

    # layer 2
    layer2Data = np.matmul(layer1Data, NN[5]["Data"].T)
    #bias 2    
    layer2Data = layer2Data + NN[6]["Data"].T
    #sigmoid
    layer2Data = sigmoid(layer2Data)

    # layer 3
    layer3Data = np.matmul(layer2Data, NN[7]["Data"].T)
    #bias 3.
    layer3Data = layer3Data + NN[8]["Data"].T
    return layer3Data

def upload_xml(filename):
    tree = etree.parse(filename)
    root = tree.getroot()
    wordTimesTo = []
    wordTimesFrom = []
    words = []
    phonemeTimesTo = []
    phonemeTimesFrom = []
    phonemes = []
    o = 0
    p = 0
    for child in root:
        for grandchild in child:
            for greatgrandchild in grandchild:
                segmentTimeFrom = greatgrandchild.get('start') 
                segmentTimeFrom = int(float(segmentTimeFrom)*100)

                segmentTimeTo = greatgrandchild.get('end') 
                segmentTimeTo = int(float(segmentTimeTo)*100)
            
                for goal in greatgrandchild:
                    f = goal.get('from') 
                    f = int(float(f)*100)
                    wordTimesFrom.append(f)
                        
                    t = goal.get('to') 
                    t = int(float(t)*100)
                    wordTimesTo.append(t)
                    
                    w = goal.text.strip()
                    words.append(w)                    
                    o=o+1
                    for nextGoal in goal:
                        m = nextGoal.get('from') 
                        m  = int(float(m)*100)
                        phonemeTimesFrom.append(m)
                            
                        l = nextGoal.get('to') 
                        l = int(float(l)*100)
                        phonemeTimesTo.append(l)
                        
                        ph = nextGoal.text
                        phonemes.append(ph)
                        p = p+1

    d,e,f,g = ([] for i in range(4))    
    for i in xrange (o):
        d.append(-1)
        e.append(-1)
    for j in xrange (p):
        f.append(-1)
        g.append(-1)
    timesXml = [words, wordTimesFrom, wordTimesTo, d, e,
                phonemes, phonemeTimesFrom, phonemeTimesTo, f, g]
    return timesXml

filenameX = sys.argv[1]
filenameY = sys.argv[2]
filenameXml =sys.argv[3]
#reference
x = fwd_pass_NN(exp_context(7, norm_data(filenameX)),upload_NN ("/mnt/matylda6/szoke/REPLAYWELL/GoLearn/DATA/NNBN_15fb24_silMax31_cmn_FSNRoabc500h_NN2+0HL300BN30_120_BN.complete_transform_1"))
#example
y = fwd_pass_NN(exp_context(7, norm_data(filenameY)),upload_NN ("/mnt/matylda6/szoke/REPLAYWELL/GoLearn/DATA/NNBN_15fb24_silMax31_cmn_FSNRoabc500h_NN2+0HL300BN30_120_BN.complete_transform_1"))    
shaX = np.shape(x)
shaY = np.shape(y)
sizeX = shaX[0]
sizeY = shaY[0]
#init matrix
dtwM = np.full((sizeX+1, sizeY+1), np.inf)
dtwM[0,0] = 0 
distMatrix= np.zeros(shape = (sizeX, sizeY))
pathMatrix= np.full((sizeX, sizeY), np.inf)
distPathM = np.full((sizeX+1, sizeY+1), 1)
distMatrix = dist.cosine_similarity(x,y)*(-1) +1
#forward
for i in xrange (sizeX):
    for j in xrange (sizeY):
        kontr = 0        
        div1 = dtwM[i,j]/distPathM[i,j]
        div2 = dtwM[i, j+1]/distPathM[i,j+1]
        div3 = dtwM[i+1,j]/distPathM[i+1,j]
        minim = min (div1, div2, div3)        
        if minim == div1:
            dtwM[i+1,j+1] = dtwM[i,j]+ distMatrix[i,j]
            pathMatrix[i,j] = 2 
            distPathM[i+1,j+1] = distPathM[i,j]+1
            kontr = 1
        if minim == div2:
            dtwM[i+1,j+1] = dtwM[i,j+1]+ distMatrix[i,j]
            pathMatrix[i,j] = 1
            distPathM[i+1,j+1] = distPathM[i,j+1]+1
            kontr = 2
        if minim == div3:
            dtwM[i+1,j+1] = dtwM[i+1,j]+ distMatrix[i,j]
            pathMatrix[i,j] = 0
            distPathM[i+1,j+1] = distPathM[i+1,j]+1
            kontr = 3
        if kontr == 0:
            print "chyba"              
a = sizeX-1 #reference
b = sizeY-1 #example
xmlTimes = upload_xml(filenameXml)
#np.savetxt("dist_m.txt",distMatrix,'%.3f')
#backward
for x in xrange (len(xmlTimes[0])):
    if xmlTimes[1][x] > (sizeX-1):
        xmlTimes[1][x] = (sizeX-1)
        xmlTimes[3][x] = (sizeY-1)
    if xmlTimes[2][x] > (sizeX-1):
        xmlTimes[2][x] = (sizeX-1)
        xmlTimes[4][x] = (sizeY-1)
for x in xrange (len(xmlTimes[5])):
    if xmlTimes[6][x] > (sizeX-1):
        xmlTimes[6][x] = (sizeX-1)
        xmlTimes[8][x] = (sizeY-1)
    if xmlTimes[7][x] > (sizeX-1):
        xmlTimes[7][x] = (sizeX-1)
        xmlTimes[9][x] = (sizeY-1)

while (a>=0) and (b>=0):
    for x in xrange (len(xmlTimes[0])):
        if xmlTimes[1][x] == a:
            xmlTimes[3][x] = b
        if xmlTimes[2][x] == a:
            xmlTimes[4][x] = b
    for j in xrange (len(xmlTimes[5])):
        if xmlTimes[6][j] == a:
            xmlTimes[8][j] = b
        if xmlTimes[7][j] == a:
            xmlTimes[9][j] = b
    if pathMatrix[a,b] == 2:
        #pathMatrix[a, b] = 20
        a = a-1
        b = b-1
    else:
        if pathMatrix[a,b] == 1:
            #pathMatrix[a, b] = 20
            a = a-1
        else:
            if pathMatrix[a,b] == 0:
                #pathMatrix[a, b] = 20
                b = b-1
rawScoreW, rawScoreP, pathLenW, pathLenP, scoreW, scoreP = ([] for i in range(6))
for x in xrange (len(xmlTimes[0])): 
    rawScoreW.append(dtwM[xmlTimes[2][x]+1, xmlTimes[4][x]+1] - dtwM[xmlTimes[1][x]+1, xmlTimes[3][x]+1])
    pathLenW.append(distPathM[xmlTimes[2][x], xmlTimes[4][x]] - distPathM[xmlTimes[1][x], xmlTimes[3][x]])
    scoreW.append(rawScoreW[x]/pathLenW[x])
for x in xrange (len(xmlTimes[5])): 
    rawScoreP.append(dtwM[xmlTimes[7][x]+1, xmlTimes[9][x]+1] - dtwM[xmlTimes[6][x]+1, xmlTimes[8][x]+1])
    pathLenP.append(distPathM[xmlTimes[7][x], xmlTimes[9][x]] - distPathM[xmlTimes[6][x], xmlTimes[8][x]])
    scoreP.append(rawScoreP[x]/pathLenP[x])

textFile = open(sys.argv[4], "w")
y = 0
for x in xrange (len(xmlTimes[0])):
    a="w "+xmlTimes[0][x]+" "+sys.argv[5]+" "+str(float(xmlTimes[1][x])/100)+" "+str(float(xmlTimes[2][x])/100)+" "+sys.argv[6] +" "+str(float(xmlTimes[3][x])/100)+" "+str(float(xmlTimes[4][x])/100)+" "+str(rawScoreW[x])+" "+str(pathLenW[x])+"\n"
    textFile.write(a)    
    while (y<len(xmlTimes[5])) and (xmlTimes[7][y]<=xmlTimes[2][x]):
            a = "p "+xmlTimes[5][y]+" "+sys.argv[5]+" "+str(float(xmlTimes[6][y])/100)+" "+str(float(xmlTimes[7][y])/100)+" "+sys.argv[6] +" "+str(float(xmlTimes[8][y])/100)+" "+str(float(xmlTimes[9][y])/100)+" "+str(rawScoreP[y])+" "+str(pathLenP[y])+"\n"
            textFile.write(a) 
            y = y+1
textFile.close()


