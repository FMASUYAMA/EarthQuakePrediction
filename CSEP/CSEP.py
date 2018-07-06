# -*- coding: utf-8 -*-
import os
import numpy as np
import pdb
import matplotlib.pylab as plt
import pickle
import pandas as pd

visualPath = 'visualization'

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
	# ヒュベニの公式を用いた緯度・経度座標系の2点間の距離(km)
	# https://qiita.com/chiyoyo/items/b10bd3864f3ce5c56291
	# を参考にして作成
	# lat1: 中心座標の緯度
	# lon1: 中心座標の緯度
	# inds: データのインデックス
	# mode: 測地系の切り替え
	def deg2dis(self, lat1, lon1, data, mode=True):
		lat2 = data['latitude'].values
		lon2 = data['longitude'].values
		
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

############## MAIN #####################
if __name__ == "__main__":
	myCSEP = Data()
	
	#---------------------
	# RI10k法：10kmの円に入る各マグニチュードの回数をカウント
	
	# 2000-01-01～2017-12-31の期間
	data = myCSEP.limitDate('2000-01-01','2017-12-31')
	
	# CSEP関東領域グリッド（lon:138.475-141.525, lat:34.475-37.025）
	lats = np.arange(34.475, 37.025, 0.05)
	lons = np.arange(138.475, 141.525, 0.05)
	
	# マグニチュードの発生回数
	nums3_4 = np.zeros([len(lats), len(lons)])
	nums4_5 = np.zeros([len(lats), len(lons)])
	nums5_6 = np.zeros([len(lats), len(lons)])
	nums6_7 = np.zeros([len(lats), len(lons)])
	nums3_ = np.zeros([len(lats), len(lons)])
	nums4_ = np.zeros([len(lats), len(lons)])
	nums5_ = np.zeros([len(lats), len(lons)])
	nums6_ = np.zeros([len(lats), len(lons)])
	
	for i, lat in enumerate(lats):
		print("latitude:{}...".format(lat))
		for j, lon in enumerate(lons):
			dist = myCSEP.deg2dis(lat, lon, data)
	
			# 10km以内のマグニチュードの抽出
			mags = data['magnitude'][dist<=10].values
	
			# 各マグニチュードの回数
			nums3_4[i,j] = np.sum((mags >= 3) & (mags < 4))
			nums4_5[i,j] = np.sum((mags >= 4) & (mags < 5))
			nums5_6[i,j] = np.sum((mags >= 5) & (mags < 6))
			nums6_7[i,j] = np.sum((mags >= 6) & (mags < 7))
			nums3_[i,j] = np.sum(mags >= 3)
			nums4_[i,j] = np.sum(mags >= 4)
			nums5_[i,j] = np.sum(mags >= 5)
			nums6_[i,j] = np.sum(mags >= 6)
	#---------------------

	#---------------------
	# プロット
	fig, figInds = plt.subplots(nrows=2, ncols=4)
	
	figInds[0,0].imshow(nums3_4,cmap="bwr")
	figInds[0,0].set_title('3_4')
	figInds[0,1].imshow(nums4_5,cmap="bwr")
	figInds[0,1].set_title('4_5')
	figInds[0,2].imshow(nums5_6,cmap="bwr")
	figInds[0,2].set_title('5_6')
	figInds[0,3].imshow(nums6_7,cmap="bwr")
	figInds[0,3].set_title('6_7')
	
	figInds[1,0].imshow(nums3_,cmap="bwr")
	figInds[1,0].set_title('3_')
	figInds[1,1].imshow(nums4_,cmap="bwr")
	figInds[1,1].set_title('4_')	
	figInds[1,2].imshow(nums5_,cmap="bwr")
	figInds[1,2].set_title('5_')	
	figInds[1,3].imshow(nums6_,cmap="bwr")
	figInds[1,3].set_title('6_')
	
	fullPath = os.path.join(visualPath,'RI10k.png')
	plt.savefig(fullPath)
	plt.show()	
	#---------------------

	pdb.set_trace()
	
#########################################
