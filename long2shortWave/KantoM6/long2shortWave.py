# -*- coding: utf-8 -*-
import os
import numpy as np
import pdb
import matplotlib.pylab as plt
import pickle
import pywt
import glob
import pandas as pd

visualPath = "visualization"

#------------------------------------
# 長周期と短周期の波のプロット
def plotWaves(longWv,shortWv,dataName):
	fig, figInds = plt.subplots(nrows=longWv.shape[1]-1, ncols=2, sharex=True)
	
	for figInd in np.arange(longWv.shape[1]-1):
		figInds[figInd][0].plot(longWv[:,0], longWv[:,figInd+1])
		figInds[figInd][1].plot(shortWv[:,0], shortWv[:,figInd+1])
		
	fig.suptitle(dataName) 
	
	path = os.path.join(visualPath,dataName)
	plt.savefig(path)
#------------------------------------
	
############## MAIN #####################
if __name__ == "__main__":
	
	rootPaths = ["20000603175400"]
	prefName = "*"
	isWindows = True
	maxLenWave = 10000
	lenWind = 1000
	
	_X = []
	_Y = []
	for rootPath in rootPaths:
		#--------------------
		# long/short_wave.datの読み込み
		if isWindows:
			longFiles = glob.glob('{}\\{}*_long_wave.dat'.format(rootPath,prefName))
			shortFiles = glob.glob('{}\\{}*_short_wave.dat'.format(rootPath,prefName))
		else:
			longFiles = glob.glob('{}/{}*_long_wave.dat'.format(rootPath,prefName))
			shortFiles = glob.glob('{}/{}*_short_wave.dat'.format(rootPath,prefName))

	
		for fID in np.arange(len(longFiles)):
			print('reading',longFiles[fID])
			print('reading',shortFiles[fID])

			longDF = pd.read_csv(longFiles[fID],header=None,sep='\t')
			shortDF = pd.read_csv(shortFiles[fID],header=None,sep='\t')
		
			if isWindows:
				dataName = longFiles[fID].split('_')[0].replace('\\','_')
			else:
				dataName = longFiles[fID].split('_')[0].replace('/','_')

			# プロット
			plotWaves(longDF.values,shortDF.values, dataName)
	
			_X.append(longDF.values)
			_Y.append(shortDF.values)
			
			pdb.set_trace()
		#--------------------
	
		#--------------------
		# reshape for training
		X=np.transpose(np.array([np.reshape(_X[i][:maxLenWave,0],[-1,lenWind]) for i in np.arange(0,len(_X))]),[1,0,2])
		Y=np.transpose(np.array([np.reshape(_Y[i][:maxLenWave,0],[-1,lenWind]) for i in np.arange(0,len(_Y))]),[1,0,2])
		#--------------------
	

	pdb.set_trace()
#########################################