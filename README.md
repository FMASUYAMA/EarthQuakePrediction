# EarthQuakePrediction

# How to use PointCNN
cd EarthQuakePrediction/PointCNN/data_conversions

python prepare_csep_data.py

cd EarthQuakePrediction/PointCNN/pointcnn_cls

./train_val_csep.sh -g 0 -x csep_x2_l4

See training log: 

EarthQuakePrediction/models/cls/pointcnn_cls_csep_x2_l4_2018-*/log.txt

Change setting:

EarthQuakePrediction/PointCNN/pointcnn_cls/csep_x2_l4.py

EarthQuakePrediction/PointCNN/pointcnn_cls/train_val_csep.sh

EarthQuakePrediction/PointCNN/data_conversions/prepare_csep_data.py

