# -*- coding: utf-8 -*-
import os
import numpy as np
import pdb
import matplotlib.pylab as plt
import pickle
import pandas as pd

#########################################
class Data:
	dataPath = 'data'

	#--------------------------
	# データの読み込み
	def __init__(self):
		fullPath = os.path.join(self.dataPath,'atr.dat')
		self.data = pd.read_csv(fullPath,sep='\t',index_col='date', parse_dates=['date'])
		self.nData = len(self.data)
	#--------------------------
	
	#--------------------------
	# 期間による絞り込み
	def limitDate(self, sDate, eDate):
		data = self.data[sDate:eDate]
		return data
	#--------------------------

	#--------------------------
	# グリッド内のデータ
	def getDataInGrid(self, sLat, sLon, eLat, eLon, data):
		tmpData = data[(data['latitude'] >= sLat) & (data['latitude'] < eLat) &
		 (data['longitude'] >= sLon)  & (data['longitude'] < eLon)]
		return tmpData
	#--------------------------
	
	#--------------------------
	# ヒュベニの公式を用いた緯度・経度座標系の2点間の距離(km)
	# https://qiita.com/chiyoyo/items/b10bd3864f3ce5c56291
	# を参考にして作成
	# lat1: 1点目の緯度
	# lon1: 1点目の経度
	# lat2: 2点目の緯度
	# lon2: 2点目の経度	
	# data: 中心座標から距離を測りたいデータ
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
