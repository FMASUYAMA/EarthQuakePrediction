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
	# �f�[�^�̓ǂݍ���
	def __init__(self):
		fullPath = os.path.join(self.dataPath,'atr.dat')
		self.data = pd.read_csv(fullPath,sep='\t',index_col='date', parse_dates=['date'])
		self.nData = len(self.data)
	#--------------------------
	
	#--------------------------
	# ���Ԃɂ��i�荞��
	def limitDate(self, sDate, eDate):
		data = self.data[sDate:eDate]
		return data
	#--------------------------

	#--------------------------
	# �O���b�h���̃f�[�^
	def getDataInGrid(self, sLat, sLon, eLat, eLon, data):
		tmpData = data[(data['latitude'] >= sLat) & (data['latitude'] < eLat) &
		 (data['longitude'] >= sLon)  & (data['longitude'] < eLon)]
		return tmpData
	#--------------------------
	
	#--------------------------
	# �q���x�j�̌�����p�����ܓx�E�o�x���W�n��2�_�Ԃ̋���(km)
	# https://qiita.com/chiyoyo/items/b10bd3864f3ce5c56291
	# ���Q�l�ɂ��č쐬
	# lat1: 1�_�ڂ̈ܓx
	# lon1: 1�_�ڂ̌o�x
	# lat2: 2�_�ڂ̈ܓx
	# lon2: 2�_�ڂ̌o�x	
	# data: ���S���W���狗���𑪂肽���f�[�^
	# mode: ���n�n�̐؂�ւ�
	def deg2dis(self, lat1, lon1, lat2, lon2, mode=True):
		#lat2 = data['latitude'].values
		#lon2 = data['longitude'].values
		
		# �ܓx�o�x�����W�A���ɕϊ�
		radLat1 = lat1/180*np.pi # �ܓx�P
		radLon1 = lon1/180*np.pi # �o�x�P
		radLat2 = lat2/180*np.pi # �ܓx�Q
		radLon2 = lon2/180*np.pi # �o�x�Q
		
		# �ܓx��
		radLatDiff = radLat1 - radLat2

		# �o�x���Z
		radLonDiff = radLon1 - radLon2;

		# ���ψܓx
		radLatAve = (radLat1 + radLat2) / 2.0

		# ���n�n�ɂ��l�̈Ⴂ
		a = [6378137.0 if mode else 6377397.155][0]						# �ԓ����a
		b = [6356752.314140356 if mode else 6356078.963][0]				# �ɔ��a
		e2 = [0.00669438002301188 if mode else 0.00667436061028297][0]	# ��ꗣ�S��^2
		a1e2 = [6335439.32708317 if mode else 6334832.10663254][0]		# �ԓ���̎q�ߐ��ȗ����a

		sinLat = np.sin(radLatAve)
		W2 = 1.0 - e2 * (sinLat**2)
		M = a1e2 / (np.sqrt(W2)*W2)		# �q�ߐ��ȗ����aM
		N = a / np.sqrt(W2)				# �K�ѐ��ȗ����a

		t1 = M * radLatDiff;
		t2 = N * np.cos(radLatAve) * radLonDiff
		dist = np.sqrt((t1 * t1) + (t2 * t2))

		return dist/1000
	#--------------------------
#########################################
