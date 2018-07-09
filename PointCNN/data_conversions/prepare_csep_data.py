#!/usr/bin/python3
'''Convert MNIST to points.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import h5py
import random
import argparse
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath('..')))
import data_utils
import CSEP.CSEP as CSEP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', '-f', help='Path to data folder')
    parser.add_argument('--point_num', '-p', help='Point number for each sample', type=int, default=256)
    parser.add_argument('--save_ply', '-s', help='Convert .pts to .ply', action='store_true')
    args = parser.parse_args()
    print(args)

    batch_size = 256

    folder_csep = args.folder if args.folder else '../../CSEP/data'
    folder_pts = os.path.join(os.path.dirname(folder_csep), 'pts')

    m_l = 2.5 # 最小マグニチュード
    m_m = 8.0 # 最大マグニチュード?
    m_th = 4.0 # binary label作成の閾値

	# CSEP関東領域（lon:138.475-141.525, lat:34.475-37.025）
    s_lat = 34.475
    e_lat = 37.025
    s_lon = 138.475
    e_lon = 141.525

    # Training and validation
    s_train_day = '1980-01-01'    # 学習の開始日
    e_train_day = '2016-12-31'    # 学習の終了日
    s_test_day = '2017-01-01'    # 評価の開始日
    e_test_day = '2017-12-31'    # 評価の終了日

    # CSEPのデータクラス
    csep_data = CSEP.Data(s_train_day, e_train_day, s_test_day, e_test_day, dataPath=folder_csep)

    # データ整形
    win_in = [120]
    win_out = [3]
    win_stride = [1]
    save_data = np.zeros((batch_size, args.point_num, 4)) # using points.append((longitude, random.random() * 1e-6, latitude))
    save_label = np.zeros((batch_size), dtype=np.int32)

    for tag in ['train', 'test']: # 仮の構成として学習データを評価データにする
        idx_h5 = 0
        filename_filelist_h5 = os.path.join(folder_csep, '%s_files.txt' % tag)
        with open(filename_filelist_h5, 'w') as filelist_h5:
            for (w_in, w_out, w_stride) in zip(win_in, win_out, win_stride):
                point_num_total = 0
                data_list, label_list, dt_list = csep_data.splitData2Slice(w_in, w_out, w_stride)
                for idx_img, (data, label, dt) in enumerate(zip(data_list, label_list, dt_list)): # '画像'に相当するループ
                    year_end = int(dt.year)
                    data = data[(data['latitude'] >= s_lat) & (data['latitude'] < e_lat) &
            		    (data['longitude'] >= s_lon)  & (data['longitude'] < e_lon)]
                    label = label[(label['latitude'] >= s_lat) & (label['latitude'] < e_lat) &
            		    (label['longitude'] >= s_lon)  & (label['longitude'] < e_lon)]
        
                    points = []
                    pixels = []
                    max_magnitude = 0
                    for idx_pixel, (point_data, point_label) in enumerate(zip(data.iterrows(), label.iterrows())): # '画素'に相当するループ
                        magnitude = point_data[1].magnitude
                        if magnitude <= m_l:
                            continue
                        date_abs = point_data[0]
                        year_abs = int(str(date_abs.year))
                        year_relative = year_abs - year_end # Sliding Windowの終端からの時間差（負の数）
                        longitude = point_data[1].longitude
                        latitude = point_data[1].latitude
                        depth = point_data[1].depth
                        points.append((longitude, random.random() * 1e-6, latitude))
                        #points.append((longitude, random.random() * 1e-6, latitude, depth, year_relative))
                        pixels.append(magnitude)
                        if point_label[1].magnitude > max_magnitude:
                            max_magnitude = point_label[1].magnitude
                    label = max_magnitude >= m_th
                    point_num_total = point_num_total + len(points)
                    pixels_sum = sum(pixels)
                    probs = [pixel / pixels_sum for pixel in pixels]
                    indices = np.random.choice(list(range(len(points))), size=args.point_num,
                                   replace=(len(points) < args.point_num), p=probs)
                    points_array = np.array(points)[indices]
                    pixels_array_1d = (np.array(pixels)[indices].astype(np.float32) / m_m) - 0.5
                    pixels_array = np.expand_dims(pixels_array_1d, axis=-1)
        
                    points_min = np.amin(points_array, axis=0)
                    points_max = np.amax(points_array, axis=0)
                    points_center = (points_min + points_max) / 2
                    scale = np.amax(points_max - points_min) / 2
                    points_array = (points_array - points_center) * (0.8 / scale)
                
                    if args.save_ply:
                        filename_pts = os.path.join(folder_pts, tag, '{:06d}.ply'.format(idx_img))
                        data_utils.save_ply(points_array, filename_pts, colors=np.tile(pixels_array, (1, 3)) + 0.5)
        
                    idx_in_batch = idx_img % batch_size
                    save_data[idx_in_batch, ...] = np.concatenate((points_array, pixels_array), axis=-1)
                    save_label[idx_in_batch] = label
                    if ((idx_img + 1) % batch_size == 0) or idx_img == len(data_list) - 1:
                        item_num = idx_in_batch + 1
                        filename_h5 = os.path.join(folder_csep, '%s_%d.h5' % (tag, idx_h5))
                        print('{}-Saving {}...'.format(datetime.now(), filename_h5))
                        filelist_h5.write('./%s_%d.h5\n' % (tag, idx_h5))
        
                        file = h5py.File(filename_h5, 'w')
                        file.create_dataset('data', data=save_data[0:item_num, ...])
                        file.create_dataset('label', data=save_label[0:item_num, ...])
                        file.close()
        
                        idx_h5 = idx_h5 + 1
                
    print('Average point number in each sample is : %f!' % (point_num_total / len(data_list)))
    
if __name__ == '__main__':
    main()
