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
# pointCNN法

if __name__ == "__main__":
	cellSize = 0.05				# セルの大きさ（°）
	mL = 2.5					# 最小マグニチュード
	Ss = [10, 30,50,100]		# 平滑化パラメータ（グリッド中心からの距離[km]）
	sTrainDay = '1950-01-01'	# 学習の開始日
	eTrainDay = '2016-12-31'	# 学習の終了日
	sTestDay = '2017-01-01'		# 評価の開始日
	eTestDay = '2017-12-31'		# 評価の終了日

	# CSEPのデータクラス
	myCSEP = CSEP.Data(sTrainDay, eTrainDay, sTestDay, eTestDay)
	
	# データをスライディングウィンドウで分割
	myCSEP.splitData2Slice(winIn=120, winOut=3, stride=1)
	
	# データの作成
	myCSEP.makePointCNNData(trainRatio=0.8)

#########################################
