# Configuration File

# Base directory for data formats
name = 'GURO_CELL'
data_base = '/home/mnt/datasets/'+name
aug_base = '/home/bumsoo/Data/split/'+name
# test_dir = '/home/bumsoo/Data/test/INBREAST'

# model option
batch_size = 16
num_epochs = 50
lr_decay_epoch=10
feature_size = 100

# data option
mean = [0.78137868728010351, 0.61806759968710656, 0.6235838367660721]
std = [0.17391331169026211, 0.25071588588807481, 0.22333351056788922]
# mean = [0.456, 0.456, 0.456]
# std = [0.224, 0.224, 0.224]
