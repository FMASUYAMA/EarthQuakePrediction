# -*- coding: utf-8 -*-
import os
import numpy as np
import pdb
import matplotlib.pylab as plt
import pickle
import pandas as pd
import pandas.tseries.offsets as offsets

#########################################
class Data:
	
	#--------------------------
	# データの読み込み
	def __init__(self, sTrain, eTrain, sTest, eTest, dataPath='data'):
		self.sTrain = sTrain
		self.eTrain = eTrain
		self.sTest = sTest
		self.eTest = eTest
		self.dataPath = dataPath
        
		fullPath = os.path.join(self.dataPath,'atr.dat')
		self.data = pd.read_csv(fullPath,sep='\t',index_col='date', parse_dates=['date'])
		
		# 学習データ
		self.dataTrain = self.data[sTrain:eTrain]
		
		# テストデータ
		self.dataTest = self.data[sTest:eTest]
		
	#--------------------------

	#--------------------------
	# グリッド内のデータ取り出し
	# sLat: 開始緯度
	# sLon: 開始経度
	# eLat: 終了緯度
	# eLon: 終了経度
	# dataType: all, train, test
	def getDataInGrid(self, sLat, sLon, eLat, eLon, dataType='train'):
		if dataType=='all':
			data = self.data
		elif dataType=='train':
			data = self.dataTrain
		elif dataType=='test':
			data = self.dataTest
			
		tmpData = data[(data['latitude'] >= sLat) & (data['latitude'] < eLat) &
		 (data['longitude'] >= sLon)  & (data['longitude'] < eLon)]
		 
		return tmpData
	#--------------------------
	
	#--------------------------
	# sliding windowでデータを分割
	# winIn: 入力用のウィンドウ幅（単位：月）
	# winOut: 出力用のウィンドウ幅（単位：月）	
	# stride: ずらし幅（単位：月）
	def splitData2Slice(self, winIn=120, winOut=3, stride=1):
	
		# ウィンドウ幅と、ずらし幅のoffset
		winInOffset = offsets.DateOffset(months=winIn, days=-1)
		winOutOffset = offsets.DateOffset(months=winOut, days=-1)
		strideOffset = offsets.DateOffset(months=stride)
		
		# 学習データの開始・終了のdatetime
		sTrainDT = pd.to_datetime(self.sTrain)
		eTrainDT = pd.to_datetime(self.eTrain)
		
		#---------------
		# 各ウィンドウのdataframeを取得
		self.dfX = []
		self.dfY = []
		
		# 現在の日時
		currentDT = sTrainDT
		endDTList = [] # Saito temporarily added (7/9)
		while currentDT + winInOffset + winOutOffset <= eTrainDT:
			endDTList.append(currentDT+winInOffset) # Saito temporarily added (7/9)
		
			# 現在の日時からwinInOffset分を抽出
			self.dfX.append(self.dataTrain[currentDT:currentDT+winInOffset])

			# 現在の日時からwinInOffset分を抽出
			self.dfY.append(self.dataTrain[currentDT+winInOffset:currentDT+winInOffset+winOutOffset])
			
			# 現在の日時をstrideOffset分ずらす
			currentDT = currentDT + strideOffset
		#---------------
        
		return self.dfX, self.dfY, endDTList, # Saito temporarily added (7/9)
	#--------------------------

	#--------------------------
	# pointCNN用のデータ作成
	def makePointCNNData(self, trainRatio=0.8):
		# 学習データとテストデータ数
		self.nData = len(self.dfX)
		self.nTrain = np.floor(self.nData * trainRatio).astype(int)
		self.nTest = self.nData - self.nTrain
		
		# ランダムにインデックスをシャッフル
		self.randInd = np.random.permutation(self.nData)
		
		'''
		# 学習データ
		self.xTrain = self.dfX[self.randInd[0:self.nTrain]]
		self.yTrain = self.dfY[self.randInd[0:self.nTrain]]

		# 評価データ
		self.xTest = self.dfX[self.randInd[self.nTrain:]]
		self.yTest = self.dfY[self.randInd[self.nTrain:]]
		'''
		
		# ミニバッチの初期化
		self.batchCnt = 0
		self.batchRandInd = np.random.permutation(self.nTrain)
		#--------------------		
	#--------------------------
	
	#------------------------------------
	# pointCNN用のミニバッチの取り出し
	def nextPointCNNBatch(self,batchSize):

		sInd = batchSize * self.batchCnt
		eInd = sInd + batchSize

		batchX = []
		batchY = []
		'''
		batchX = self.xTrain[self.batchRandInd[sInd:eInd]]
		batchY = self.yTrain[self.batchRandInd[sInd:eInd]]
		'''
		
		if eInd+batchSize > self.nTrain:
			self.batchCnt = 0
		else:
			self.batchCnt += 1

		return batchX, batchY
	#------------------------------------
		
	#--------------------------
	# ヒュベニの公式を用いた緯度・経度座標系の2点間の距離(km)
	# https://qiita.com/chiyoyo/items/b10bd3864f3ce5c56291
	# を参考にして作成
	# lat1: 1点目の緯度
	# lon1: 1点目の経度
	# lat2: 2点目の緯度
	# lon2: 2点目の経度	
	# mode: 測地系の切り替え
	def deg2dis(self, lat1, lon1, lat2, lon2, mode=True):
		#lat2 = data['latitude'].values
		#lon2 = data['longitude'].values
		
		# 緯度経度をラジアンに変換
		radLat1 = lat1/180*np.pi # 緯度１
		radLon1 = lon1/180*np.pi # 経度１
		radLat2 = lat2/180*np.pi # 緯度２
		radLon2 = lon2/180*np.pi # 経度２
		
		# 緯度差
		radLatDiff = radLat1 - radLat2

		# 経度差算
		radLonDiff = radLon1 - radLon2;

		# 平均緯度
		radLatAve = (radLat1 + radLat2) / 2.0

		# 測地系による値の違い
		a = [6378137.0 if mode else 6377397.155][0]						# 赤道半径
		b = [6356752.314140356 if mode else 6356078.963][0]				# 極半径
		e2 = [0.00669438002301188 if mode else 0.00667436061028297][0]	# 第一離心率^2
		a1e2 = [6335439.32708317 if mode else 6334832.10663254][0]		# 赤道上の子午線曲率半径

		sinLat = np.sin(radLatAve)
		W2 = 1.0 - e2 * (sinLat**2)
		M = a1e2 / (np.sqrt(W2)*W2)		# 子午線曲率半径M
		N = a / np.sqrt(W2)				# 卯酉線曲率半径

		t1 = M * radLatDiff;
		t2 = N * np.cos(radLatAve) * radLonDiff
		dist = np.sqrt((t1 * t1) + (t2 * t2))

		return dist/1000
	#--------------------------
#########################################
