dataset_path = '/Users/abdu/Desktop/Research/Shanghai/'

batch_size = 1
num_workers = 4
num_epochs = 400
learning_rate = 1e-7

train_json = 'json_files/part_A_train.json'
val_json = 'json_files/part_A_val.json'
test_json = 'json_files/part_A_test.json'

hparams = {
    "lr":1e-6,
    "batch_size":1,
    "momentum":0.95,
    "weight_decay":5*1e-4,
    "num_workers":4,
    "num_epochs":400
    }