import os

def get_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} does not exist.")
        return None

    ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]

    if len(ckpt_files) == 0:
        print("No checkpoint files found in the directory.")
        return None

    # Sort by modification time (newest first)
    ckpt_files = sorted(ckpt_files, key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)

    # Return the latest checkpoint
    latest_ckpt = os.path.join(checkpoint_dir, ckpt_files[0])
    return latest_ckpt



# Other configuration variables
dataset_path = 'D:\\Ashhad\\FYP\\Dataset\\Shanghai'
batch_size = 1
num_workers = 4
num_epochs = 400
learning_rate = 1e-7

train_json = 'json_files/part_A_train.json'
val_json = 'json_files/part_A_val.json'
test_json = 'json_files/part_A_test.json'
checkpoint_dir = 'CSRNet-Light/t36zoxvu/checkpoints'

hparams = {
    "lr": 1e-6,
    "batch_size": 1,
    "momentum": 0.95,
    "weight_decay": 5 * 1e-4,
    "num_workers": 4,
    "num_epochs": 400
}
