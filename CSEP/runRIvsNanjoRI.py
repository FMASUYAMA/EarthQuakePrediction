# -*- coding: utf-8 -*-
import os
import numpy as np
import pdb
import matplotlib.pylab as plt
import pickle
import pandas as pd
import CSEP

visualPath = 'visualization'

############## MAIN #####################
# RI法：グリッドごとに地震回数をカウント
# NanjoRI法：グリッドごとの地震回数を半径Sの円に入るグリッド数で平滑化

if __name__ == "__main__":
	myCSEP = CSEP.Data()
	cellSize = 0.05				# セルの大きさ（°）
	mL = 2.5					# 最小マグニチュード
	Ss = [10, 30,50,100]		# 平滑化パラメータ（グリッド中心からの距離[km]）
	sTrainDay = '2000-01-01'	# 学習の開始日
	eTrainDay = '2017-12-31'	# 学習の終了日

	# 2000-01-01～2017-12-31の期間
	data = myCSEP.limitDate(sTrainDay, eTrainDay)
	
	# CSEP関東領域グリッド（lon:138.475-141.525, lat:34.475-37.025）
	lats = np.arange(34.475, 37.025, cellSize)
	lons = np.arange(138.475, 141.525, cellSize)
	
	# マグニチュードの発生回数
	numsRI = np.zeros([len(lats), len(lons)])			# RI法
	numsNanjoRI = np.zeros([len(lats), len(lons)])		# NanjoRI法

	# セル中心
	cellCsFlat = np.array([[lat + cellSize/2, lon + cellSize/2] for lat in lats for lon in lons])
	cellCs = np.reshape(cellCsFlat, [len(lats), len(lons), 2])
	
	for S in Ss:
		print("S:{}km".format(S))
		for i, lat in enumerate(lats):
			print("latitude:{}...".format(lat))
			for j, lon in enumerate(lons):
				tmpData = myCSEP.getDataInGrid(lat, lon, lat+cellSize, lon+cellSize, data)

				#-------------
				# RI法：各セルのマグニチュードmL以上の地震をカウント
				numsRI[i,j] = np.sum(tmpData['magnitude'].values > mL)
				#-------------
			
				#-------------
				# Nanjo RI法：各セルのマグニチュードmL以上の地震をカウントし、
				# 距離S以内のセル全てに(Nsb+1)^-1をカウント
				
				# セル中心からS以内にセル中心があるセルのインデックスとセル数Nsbを取得
				dists = myCSEP.deg2dis(cellCs[i,j,0],cellCs[i,j,1], cellCsFlat[:,0],cellCsFlat[:,1])
				latInds,lonInds = np.where(np.reshape(dists,[len(lats),len(lons)])<S)	# インデックス
				Nsb = len(latInds)	# セル数
			
				# (Nsb + 1)^-1を割り当てる
				for k, l in zip(latInds, lonInds):
					numsNanjoRI[k,l] += numsRI[i,j]*1/(Nsb + 1)
				#-------------
			
		#---------------------	
		# 正規化
		numsRI = numsRI / np.sum(numsRI)
		numsNanjoRI = numsNanjoRI / np.sum(numsNanjoRI)
		#---------------------	
	
		#---------------------
		# プロット
		fig, figInds = plt.subplots(ncols=2)
		figInds[0].imshow(numsRI,cmap="bwr")
		figInds[0].set_title('Relative Intensity')

		figInds[1].imshow(numsNanjoRI,cmap="bwr")
		figInds[1].set_title('Nanjo Relative Intensity')

		fullPath = os.path.join(visualPath,'RIvsNanjoRI_{}km.png'.format(S))
		plt.savefig(fullPath)
		plt.show()	
		#---------------------
#########################################
