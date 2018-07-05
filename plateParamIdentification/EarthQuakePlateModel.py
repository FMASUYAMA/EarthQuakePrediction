# -*- coding: utf-8 -*-
import os
import numpy as np
import pdb
import tensorflow as tf
from tensorflow.python.ops import nn_ops
import matplotlib.pylab as plt
import pickle
import pywt
import glob


#########################################
class EarthQuakePlateModel:
	logPath = 'logs'
	dataPath = 'data'
	visualPath = 'visualization'

	#--------------------------
	def __init__(self,logName,nCell,nYear):
		self.logName = logName
		self.logFullPath = os.path.join(self.logPath,logName)

		self.nCell = nCell
		self.nYear = nYear
		self.yInd = 1
		self.vInds = [2,3,4,5,6,7,8,9]
		self.yV = np.zeros([nYear,nCell])
	#--------------------------

	#--------------------------
	#データの読み込み
	def loadABLV(self):
		self.data = open(self.logFullPath).readlines()
		
		# A, B, Lの取得
		self.A = np.zeros(self.nCell)
		self.B = np.zeros(self.nCell)
		self.L = np.zeros(self.nCell)
		
		for i in np.arange(1,self.nCell+1):
			tmp = np.array(self.data[i].strip().split(",")).astype(np.float32)
			self.A[i-1] = tmp[0]
			self.B[i-1] = tmp[1]
			self.L[i-1] = tmp[4]
			
		
		# Vの開始行取得
		isRTOL = [True if self.data[i].count('value of RTOL')==1 else False for i in np.arange(len(self.data))]
		vInd = np.where(isRTOL)[0][0]+1
		
		# Vの値の取得（vInd行から最終行まで）
		flag = False
		for i in np.arange(vInd,len(self.data)):
			tmp = np.array(self.data[i].strip().split(",")).astype(np.float32)
			
			if not flag:
				self.V = tmp
				flag = True
			else:
				self.V = np.vstack([self.V,tmp])

	#--------------------------
	
	#--------------------------
	# Vを年単位のデータに変換
	def convV2YearlyData(self):
		
		for year in np.arange(self.nYear):
			if np.sum(np.floor(self.V[:,self.yInd])==year):
				self.yV[year,:] = np.mean(self.V[np.floor(self.V[:,self.yInd])==year,self.vInds[0]:],axis=0)
	#--------------------------
		
	#--------------------------
	# Vの生データのプロット
	def plotV(self,isPlotShow=False,isYearly=False):
		plt.close()
		fig, figInds = plt.subplots(nrows=8, sharex=True)
	
		for figInd in np.arange(len(figInds)):
			figInds[figInd].plot(self.V[:,self.yInd],self.V[:,self.vInds[0]+figInd])
			if isYearly:
				figInds[figInd].plot(np.arange(self.nYear), self.yV[:,figInd])

		fullPath = os.path.join(self.visualPath,"{}_data.png".format(self.logName))
		plt.savefig(fullPath)

		if isPlotShow:
			fig.show()
			plt.show()
	#--------------------------
	
	#--------------------------
	# Wavelet画像に変換し保存
	def convV2Wavelet(self,width=30,isPlotShow=False,isSavePkl=False):
	
		plt.close()
		
		fig, figInds = plt.subplots(nrows=self.nCell, sharex=True)
	
		# 係数マップ画像領域の確保
		images = np.zeros([self.nCell,width,self.nYear])
		
		# 周波数
		widths = np.arange(1,width+1)
		
		for cellInd in np.arange(self.nCell):
			# continuous waveletをかけて、係数マップと周波数を出力
			images[cellInd], freqs = pywt.cwt(self.yV[:,cellInd], widths, 'mexh')
			
			# 係数マップのプロット
			figInds[cellInd].imshow(images[cellInd], extent=[1, self.nYear, widths[-1], widths[0]], cmap='gray',aspect='auto',vmax=abs(images[cellInd]).max(), vmin=-abs(images[cellInd]).max())
		
		# プロット画像の保存
		fullPath = os.path.join(self.visualPath,"{}_cwt.png".format(self.logName))
		plt.savefig(fullPath)

		if isPlotShow:
			fig.show()
			plt.show()

		# pklデータの保存
		fullPath = os.path.join(self.dataPath,"{}.pkl".format(self.logName))
		with open(fullPath,'wb') as fp:
			pickle.dump(images,fp)
			pickle.dump(log.B,fp)
			
		return images
	#--------------------------
		
	#--------------------------
	# Wavelet画像のプロット（縦：周波数、横：シフト）
	def plotVWavelet(self,width=30):
		plt.close()
		fig, figInds = plt.subplots(nrows=self.nCell, sharex=True)
		
		# 周波数
		widths = np.arange(1,width+1)
		
		for figInd in np.arange(len(figInds)):
			# continuous waveletをかけて、係数マップと周波数を出力
			cwtmatr, freqs = pywt.cwt(self.yV[:,figInd], widths, 'mexh')
			
			# 係数マップのプロット
			figInds[figInd].imshow(cwtmatr, extent=[1, self.nYear, widths[-1], widths[0]], cmap='gray', aspect='auto',vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
			
		plt.hold(True)
				
		fullPath = os.path.join(self.visualPath,"{}_cwt.png".format(self.logName))
		plt.savefig(fullPath)

		fig.show()
		plt.show()
    #--------------------------

#########################################

#########################################
class Data:
	logPath = 'logs'
	dataPath = 'data'
	visualPath = 'visualization'

	#--------------------------
	def __init__(self,fname="log_25*",trainRatio=0.8,nCell=8,nFreqs=30, sYear=1000, eYear=2000, bInd=0, isTensorflow=True,isWindows=False):

		# pklファイルの一覧
		fullPath = os.path.join(self.dataPath,fname)
		files = glob.glob(fullPath)
		
		# データの領域確保
		self.nData = len(files)
		
		nYear = eYear-sYear
		self.X = np.zeros([self.nData, nFreqs, nYear, nCell])
		self.Y = np.zeros([self.nData])
		
		# データの読み込み
		for fID in np.arange(self.nData):
			
			if isWindows:
				file = files[fID].split('\\')[1]
			else:
				file = files[fID].split('/')[1]
				
			fullPath = os.path.join(self.dataPath,file)
			
			with open(fullPath,'rb') as fp:
				tmpX = pickle.load(fp)
				tmpY = pickle.load(fp)
				
			# データの切り抜き
			tmpX = tmpX[:,:,sYear:eYear]
		
			# tensorflow用に変形
			if isTensorflow:
				self.X[fID] = np.transpose(tmpX,(1,2,0))
				self.Y[fID] = tmpY[bInd]
			else:
				self.X[fID] = tmpX
				self.Y = tmpY

		#
		pdb.set_trace()
		# 平均画像の作成
		self.meanImg = np.mean(self.X, axis=0)

		# XとYの正規化(Yが小さすぎるため）
		self.minY = np.min(self.Y)
		self.maxY = np.max(self.Y)
		self.Y = (self.Y - self.minY)/(self.maxY-self.minY)

		# 平均画像を引く
		self.X = self.X - self.meanImg

		#self.minX = np.min(self.X)
		#self.maxX = np.max(self.X)
		#self.X = (self.X - self.minX)/(self.maxX-self.minX)

		# 学習データとテストデータ数
		self.nTrain = np.floor(self.nData * trainRatio).astype(int)
		self.nTest = self.nData - self.nTrain
		
		# ランダムにインデックスをシャッフル
		self.randInd = np.random.permutation(self.nData)

		# 学習データ
		self.xTrain = self.X[self.randInd[0:self.nTrain]]
		self.yTrain = self.Y[self.randInd[0:self.nTrain]]

		# 評価データ
		self.xTest = self.X[self.randInd[self.nTrain:]]
		self.yTest = self.Y[self.randInd[self.nTrain:]]
		
		# ミニバッチの初期化
		self.batchCnt = 0
		self.batchRandInd = np.random.permutation(self.nTrain)

	#------------------------------------
	# ミニバッチの取り出し
	def nextBatch(self,batchSize,isTensorflow=True):

		sInd = batchSize * self.batchCnt
		eInd = sInd + batchSize

		batchX = self.xTrain[self.batchRandInd[sInd:eInd]]
		batchY = self.yTrain[self.batchRandInd[sInd:eInd]]
		
		if eInd+batchSize > self.nTrain:
			self.batchCnt = 0
		else:
			self.batchCnt += 1

		return batchX, batchY
	#------------------------------------
	
#########################################

############## MAIN #####################
if __name__ == "__main__":
	
	isWindows = False
	nYear = 10000
	nCell = 8
	nFreqs = 30
	
	#Reading load log.txt
	if isWindows:
		files = glob.glob('logs\\*.txt')
	else:
		files = glob.glob('logs/log*.txt')

	for fID in np.arange(len(files)):
		print('reading',files[fID])

		if isWindows:
			file = files[fID].split('\\')[1]
		else:
			file = files[fID].split('/')[1]
		
		# 地震プレートモデル用のオブジェクト
		log = EarthQuakePlateModel(file,nCell=nCell,nYear=nYear)
		log.loadABLV()
		log.convV2YearlyData()
		log.plotV(isPlotShow=False,isYearly=False)
		
		# 入力（waveletの係数マップ画像）と出力（b）の取得
		log.convV2Wavelet(width=nFreqs,isPlotShow=False,isSavePkl=True)
#########################################
