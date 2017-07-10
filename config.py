# Configuration File

# Base directory for data formats
name = 'GURO_ALL'
data_base = '/home/mnt/datasets/'+name
aug_base = '/home/bumsoo/Data/split/'+name
test_dir = '/home/bumsoo/Data/test/INBREAST'

# model option
batch_size = 16
num_epochs = 50
lr_decay_epoch=10
feature_size = 100

# data option
mean = [0.456, 0.456, 0.456]
std = [0.224, 0.224, 0.224]
