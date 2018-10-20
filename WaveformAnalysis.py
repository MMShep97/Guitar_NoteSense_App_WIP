import numpy as np
import scipy.signal as signal
import numpy
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft

import plotly.plotly as py
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF

import numpy as np
import pandas as pd
import scipy
import peakutils


# myAudio = "GuitarMiddleC.wav"
myAudio = "594pure.wav"

#Read file and get sampling freq [ usually 44100 Hz ]  and sound object
samplingFreq, mySound = wavfile.read(myAudio)
print("sampling freq: " + str(samplingFreq))

#Check if wave file is 16bit or 32 bit. 24bit is not supported
mySoundDataType = mySound.dtype

#We can convert our sound array to floating point values ranging from -1 to 1 as follows

mySound = mySound / (2.**15)

#Check sample points and sound channel for duel channel(5060, 2) or  (5060, ) for mono channel

mySoundShape = mySound.shape
samplePoints = float(mySound.shape[0])

print(mySoundShape)
print("(number of sample points, 1=mono 2=stereo)")

#Get duration of sound file
signalDuration =  mySound.shape[0] / samplingFreq
print("signal duration in seconds: " + str(signalDuration))

#If two channels, then select only one channel
# mySoundOneChannel = mySound[:,0]
mySoundOneChannel = mySound

#Plotting the tone

# We can represent sound by plotting the pressure values against time axis.
#Create an array of sample point in one dimension
timeArray = numpy.arange(0, samplePoints, 1)


#
timeArray = timeArray / samplingFreq

#Scale to milliSeconds
timeArray = timeArray * 1000

#Plot the tone
plt.plot(timeArray, mySoundOneChannel, color='G')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.show()

#--------------------------------------------------------------------------------------------

#Plot frequency content
#We can get frquency from amplitude and time using FFT , Fast Fourier Transform algorithm

#Get length of mySound object array
mySoundLength = len(mySound)
print("data points (duration in seconds x sampling freq): " + str(mySoundLength))

#Take the Fourier transformation on given sample point
#fftArray = fft(mySound)
fftArray = fft(mySoundOneChannel)
print(fftArray)

numUniquePoints = int(numpy.ceil((mySoundLength + 1) / 2.0))
print("numUniquePoints: " + str(numUniquePoints))
fftArray = fftArray[0:int(numUniquePoints)]

#FFT contains both magnitude and phase and given in complex numbers in real + imaginary parts (a + ib) format.
#By taking absolute value , we get only real part

fftArray = abs(fftArray)

#Scale the fft array by length of sample points so that magnitude does not depend on
#the length of the signal or on its sampling frequency

fftArray = fftArray / float(mySoundLength)

#FFT has both positive and negative information. Square to get positive only
fftArray = fftArray **2

#Multiply by two (research why?)
#Odd NFFT excludes Nyquist point
if mySoundLength % 2 > 0: #we've got odd number of points in fft
    fftArray[1:len(fftArray)] = fftArray[1:len(fftArray)] * 2

else: #We've got even number of points in fft
    fftArray[1:len(fftArray) -1] = fftArray[1:len(fftArray) -1] * 2

freqArray = numpy.arange(0, numUniquePoints, 1.0) * (samplingFreq / mySoundLength);
print("freqArray/1000 length: " + str(len(freqArray/1000)))

#--------------------------------------------------------------------------------





# time series treated as the x


# cb = np.array(freqArray/1000)
# indices = peakutils.indexes(cb, thres=0.678, min_dist=1) #0.1
#
# trace = go.Scatter(
#     x=[j for j in range(len(freqArray/1000))],
#     y=10 * numpy.log10 (fftArray),
#     mode='lines',
#     name='Original Plot'
# )
#
# trace2 = go.Scatter(
#     x=indices,
#     y=[(10 * numpy.log10 (fftArray))[j] for j in indices],
#     mode='markers',
#     marker=dict(
#         size=8,
#         color='rgb(255,0,0)',
#         symbol='cross'
#     ),
#     name='Detected Peaks'
# )


# data = [trace, trace2]
# data = trace
# py.iplot(data, filename='milk-production-plot-with-higher-peaks')




x = (freqArray/1000)#[:1000]
#print('x')
#print(x)
y = 10 * numpy.log10 (fftArray)
ysort = 10 * numpy.log10 (fftArray)
#print('y')
#print(y)

print('100 largest amplitudes of y')
ysort.sort()
largest100y = ysort[::-1][0:100]
print(largest100y)

print('loop')
for i in range(0,10):
    index = np.argmax(y)
    print(y[index])
    print(x[index])


#xforlargest100y = []
#for val in largest100y:
    #print(val)

    #index = y.index(val)
    #print(index)
    #xforlargest100y.append(x[index])

#print(xforlargest100y)






# window = signal.general_gaussian(51, p=0.5, sig=20)
# filtered = signal.fftconvolve(window, data)
# filtered = (np.average(data) / np.average(filtered)) * filtered
# filtered = np.roll(filtered, -25)

# peaksX = signal.find_peaks_cwt(data, np.arange(100,200))
# peaksY = [data[index] for index in peaksX]
# print(peaksX)
# print(peaksY)
#---------------------------------------------------------------------------------


#Plot the frequency
plt.plot((freqArray/1000), (10 * numpy.log10 (fftArray)), color='B')
plt.xlabel('Frequency (Khz)')
plt.ylabel('Power (dB)')
# plt.plot(peaksX, peaksY, color='R')
plt.show()

#Get List of element in frequency array
#print freqArray.dtype.type
freqArrayLength = len(freqArray)
print ("freqArrayLength =", freqArrayLength)
numpy.savetxt("freqData.txt", freqArray, fmt='%6.2f')

#Print FFtarray information
print ("fftArray length =", len(fftArray))
numpy.savetxt("fftData.txt", fftArray)
