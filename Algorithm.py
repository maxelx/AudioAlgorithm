


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


def gmm_js(gmm_p, gmm_q, n_samples=5000):
    X = gmm_p.sample(n_samples)
    Y = gmm_q.sample(n_samples)

    log_p_X = gmm_p.score_samples(X[0])
    log_q_X = gmm_q.score_samples(X[0])
    log_mix_X = log_p_X.mean() - log_q_X.mean()
    
    log_p_Y = gmm_p.score_samples(Y[0])
    log_q_Y = gmm_q.score_samples(Y[0])
    log_mix_Y = log_p_Y.mean() - log_q_Y.mean()

    return log_mix_X.mean() - log_mix_Y.mean()


def scale(OldValue, OldMin, OldMax, NewMin, NewMax ):
    NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
    return NewValue


#MFCC comparison with  Kullback-Leibler divergence
def algorithm(audio):

    import librosa
    x, sr1 = librosa.load('AudioFIle/IntradaB.wav')

    y, sr2 = librosa.load(audio)

    leng = 5000

    import math

    from sklearn.mixture import GaussianMixture

    import speechpy

    mfcc1A= speechpy.feature.mfcc(x, sr1)
    mfcc2B= speechpy.feature.mfcc(y, sr2)


    #Normalization  Global Cepstral Mean and Variance Normalization 
    mfcc1A = speechpy.processing.cmvnw(mfcc1A, win_size=651, variance_normalization=True)
    mfcc2A = speechpy.processing.cmvnw(mfcc2B, win_size=651, variance_normalization=True)

    #Extraction of the 2 gm models
    gm = GaussianMixture(n_components=64, random_state=0).fit(mfcc1A)
    gm2 = GaussianMixture(n_components=64, random_state=0).fit(mfcc2B)

    #print the KL-divergence between the 2 distributions using Monte-Carlo method
    valor = gmm_js(gm, gm2)
    valor = math.log(valor)
    print("Result", scale(valor, 5.82, 16.56 , 100, 0 ))

    return scale(valor, 5.82, 16.56 , 100, 0 )