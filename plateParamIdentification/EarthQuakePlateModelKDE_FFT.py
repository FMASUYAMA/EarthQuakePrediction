# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 17:14:14 2018

@author: yu
"""
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

import pandas as pd
import seaborn as sns
from scipy import stats

import math
import scipy.optimize

#########################################
class EarthQuakePlateModel:
	logPath = 'logs'
	dataPath = 'data'
	visualPath = 'visualization'

	#--------------------------
	def __init__(self,logName,nCell=8,nYear=10000):
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
		
		self.yV = self.yV.T
		
	#--------------------------
		
	#--------------------------
	# Vの生データのプロット
	def plotV(self,isPlotShow=False,isYearly=True,prefix='yV'):
		if isPlotShow:
			plt.close()
			fig, figInds = plt.subplots(nrows=8, sharex=True)
	
			for figInd in np.arange(len(figInds)):
				figInds[figInd].plot(self.V[:,self.yInd],self.V[:,self.vInds[0]+figInd])
				if isYearly:
					figInds[figInd].plot(np.arange(self.nYear), self.yV[:,figInd])

			fullPath = os.path.join(self.visualPath,"{}_data.png".format(self.logName))
			plt.savefig(fullPath)
			fig.show()
			plt.show()			
		
		# pklデータの保存
		fullPath = os.path.join(self.dataPath,"{}{}.pkl".format(prefix,self.logName))
		with open(fullPath,'wb') as fp:
			pickle.dump(self.yV,fp)
			pickle.dump(log.B,fp)
			pickle.dump(self.yVkde,fp)
			pickle.dump(self.yVfft,fp)
			pickle.dump(self.X,fp)
	#--------------------------
	

	#------------------------------------
	# イベントデータ（デルタ関数）を、KDEで滑らかにつなげる
	def KDE(self, v_divid = 10.0, bw = 0.01):

		flag = False
		for cellInd in np.arange(self.nCell):

			#　速度vから地震が何回起きたか相対的に取得
			v_width = self.yV[cellInd,:].max() / v_divid
			eqp_num = np.floor(self.yV[cellInd,:] / v_width)
				
			# list(float) -> list(int) -> array
			eqp_tmp = list(map(int,eqp_num))
			eqp_num = np.array(eqp_tmp)

			# 年数を任意の回数増やした(0回のデータは消える)
			eqp_v = np.repeat(np.arange(0,self.nYear),eqp_num)
			
			# KDE
			x_grid = np.arange(0,self.nYear)
			kde_model = stats.gaussian_kde(eqp_v,bw_method=bw)
			kde_model = kde_model(x_grid)
			kde_model = kde_model[np.newaxis,:]

			if not flag:
				self.yVkde = kde_model
				flag = True
			else:
				self.yVkde = np.concatenate((self.yVkde,kde_model),axis=0)
	#--------------------------

	#--------------------------
	# 周波数特徴量の抽出
	def FFT(self,widthWindow=25,eFrq=250, sYear=2000, eYear=10000):

		# FFTの計算
		self.yVfft = np.abs(np.fft.fft(self.yVkde[:,sYear:eYear]))

		#----------------------
		# スペクトラムをスライディングウィンドウごとに平均をとった特徴量の抽出
		flag = False
		for cnt in np.arange(int(eFrq/widthWindow)):
			sInd = widthWindow * cnt + 1
			eInd = sInd + widthWindow
			
			# ウィンドウのスペクトラムの平均
			X = np.mean(self.yVfft[:,sInd:eInd],axis=1)
			X = X[np.newaxis]

			if not flag:
				self.X = X
				flag = True
			else:
				self.X = np.concatenate((self.X,X),axis=0)

		self.X = self.X.T
	#--------------------------
	
#########################################

#########################################
class Data:
	logPath = 'logs'
	dataPath = 'data'
	visualPath = 'visualization'

	#--------------------------
	def __init__(self,fname="kde_fft_log_10*", trainRatio=0.8, nCell=8, sYear=2000, eYear=10000, bInd=0, isWindows=True):
				
		# pklファイルの一覧
		fullPath = os.path.join(self.dataPath,fname)
		files = glob.glob(fullPath)
		
		# データの領域確保
		self.nData = len(files)

		flag = False
		# データの読み込み
		for fID in np.arange(self.nData):
			
			if isWindows:
				file = files[fID].split('\\')[1]
			else:
				file = files[fID].split('/')[1]
				
			fullPath = os.path.join(self.dataPath,file)
			
			with open(fullPath,'rb') as fp:
				tmpyV = pickle.load(fp)
				tmpY = pickle.load(fp)
				tmpyVkde = pickle.load(fp)
				tmpyVfft = pickle.load(fp)
				tmpX = pickle.load(fp)


			if not flag:
				self.X = tmpX[np.newaxis]
				self.Y = tmpY[bInd]
				flag = True
			else:
				self.X = np.concatenate((self.X,tmpX[np.newaxis]),axis=0)
				self.Y = np.append(self.Y, tmpY[bInd])
			
		# XとYの正規化(Yが小さすぎるため,そのもののbを可視化したいときはYの正規化だけはずす）
		self.minY = np.min(self.Y)
		self.maxY = np.max(self.Y)
		self.Y = (self.Y - self.minY)/(self.maxY-self.minY)
		
		self.minX = np.min(self.X)
		self.maxX = np.max(self.X)
		self.X = (self.X - self.minX)/(self.maxX-self.minX)
		
		self.nTrain = np.floor(self.nData * trainRatio).astype(int)
		self.nTest = self.nData - self.nTrain
		
		# ミニバッチの初期化
		self.batchCnt = 0
		self.batchRandInd = np.random.permutation(self.nTrain)
		
		
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
	#------------------------------------
	
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
	
	#Reading load log.txt
	if isWindows:
		files = glob.glob('logs\\log_10*.txt')
	else:
		files = glob.glob('logs/log_10*.txt')

	for fID in np.arange(len(files)):
		print('reading',files[fID])

		if isWindows:
			file = files[fID].split('\\')[1]
		else:
			file = files[fID].split('/')[1]
		
		# 地震プレートモデル用のオブジェクト
		log = EarthQuakePlateModel(file,nCell=8,nYear=10000)
		log.loadABLV()
		log.convV2YearlyData()
		
		# KDE
		log.KDE()
		log.FFT(widthWindow=25,eFrq=250)

		# 保存
		log.plotV(isPlotShow=False,isYearly=False,prefix='kde_fft_')
	
	
	#実行する意味がない
	#data = Data(fname="yVlog_10*",trainRatio=0.8,nCell=8,sYear=2000, eYear=10000, bInd=0, isTensorflow=False,isWindows=True)
	#data.KDE(nYear=8000)
	
	#data.Wavelet(width=100)
	#data.FFT(widthWindow=25,eFrq=250,nYear=8000,nCell=8,trainRatio=0.8,isTensorflow=False)
	
############################################

