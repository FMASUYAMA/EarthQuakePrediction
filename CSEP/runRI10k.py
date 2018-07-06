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
if __name__ == "__main__":
	myCSEP = CSEP.Data()
	
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

#########################################
