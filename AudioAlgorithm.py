
import numpy as np 

def dtw(s, t, window):
    n, m = len(s), len(t)
    w = np.max([window, abs(n-m)])
    dtw_matrix = np.zeros((n+1, m+1))
    
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(np.max([1, i-w]), np.min([m, i+w])+1):
            dtw_matrix[i, j] = 0
    
    for i in range(1, n+1):
        for j in range(np.max([1, i-w]), np.min([m, i+w])+1):
            cost = abs(s[i-1] - t[j-1])
            # take last min from a square box
            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix



def percentageAudio(a):

    #Audio import
    import librosa
    x, sr1 = librosa.load('AudioFIle/IntradaB.wav')

    y, sr2 = librosa.load(a)



    #Print audio length and sample rate
    #print(x.shape)
    #print(sr1)

    #print(y.shape)
    #print(sr2)


    #sliding Windows definition function
    import statistics 
    def slidingWindows(arr, k):
        # length of the array
        n = len(arr)

    
        # n must be greater than k
        if n < k:
            print("Invalid")
            return -1
    
        #define array
        median_array = []
        
        #50% overlap factor k/2
        for i in range(0, n-k+1, int(k/2)):
            median_array.append(statistics.mean(arr[i:i+k]))
    
        return median_array



    #MFCC comparison

    mfcc1 = librosa.feature.mfcc(x, sr1)
    mfcc2 = librosa.feature.mfcc(y, sr2)

    mfccMean = []

    for i in range(0, len(mfcc1)) :
        mfcc1a = slidingWindows(mfcc1[i], 100)
        mfcc2a = slidingWindows(mfcc2[i], 100)

        mfccMean.append(dtw(mfcc1a,mfcc2a, 650)[len(mfcc1a)][len(mfcc2a)])

    mfccMeasure=statistics.mean(mfccMean)
    #print(mfccMeasure)


    #Chroma Feature Comparison
    """
    chroma1 = librosa.feature.chroma_stft(x, sr1)
    chroma2 = librosa.feature.chroma_stft(y, sr2)

    chromaMean = []

    for i in range (0, len(chroma1)) :
        chroma1a = slidingWindows(chroma1[i], 10)
        chroma2a = slidingWindows(chroma2[i], 10)
        
        chromaMean.append(dtw(chroma1a,chroma2a, 650)[len(chroma1a)][len(chroma2a)])


    chromaMeasure = statistics.mean(chromaMean)
    #print(chromaMeasure)
    """
    #scale function
    def scale(OldValue, OldMin, OldMax, NewMin, NewMax ):
        NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
        return NewValue

    #final print
    #print("MFCC: " + str(mfccMeasure))
    #print("ChromaFeature: " + str(chromaMeasure))

    #return [scale(mfccMeasure, 500, 3000, 100, 0), scale(chromaMeasure, 5, 42, 100, 0) ]
    print("Algo1", mfccMeasure)
    return scale(mfccMeasure*10, 500, 3000, 100, 0)