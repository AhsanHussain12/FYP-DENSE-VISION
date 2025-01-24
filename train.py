import torch
from torchvision import transforms

from model.model import CSRNet
import Config as cfg
import dataset
import json

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme

from argparse import ArgumentParser

def main():
    # Create an instance of the model
    model = CSRNet(learning_rate=cfg.learning_rate)

    # ================== Data ==================

    # Read json files containing image paths for training and validation
    with open (cfg.train_json) as f:
        train_list = json.load(f)

    with open (cfg.val_json) as f:
        val_list = json.load(f)

    with open(cfg.test_json) as f:
        test_list = json.load(f)

    # Create dataloaders
    train_dataset = dataset.listDataset(train_list,
                                        shuffle=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                        std=[0.229, 0.224, 0.225]),
                                        ]),
                                        train=True,
                                        batch_size=cfg.batch_size,
                                        num_workers=cfg.num_workers)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        # collate_fn=train_dataset.collate_fn,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True)
    
    val_dataset = dataset.listDataset(val_list,
                                        shuffle=False,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                        std=[0.229, 0.224, 0.225]),
                                        ]),
                                        train=False,
                                        batch_size=cfg.batch_size,
                                        num_workers=cfg.num_workers)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True)
    
    test_dataset = dataset.listDataset(test_list,
                                        shuffle=False,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                        std=[0.229, 0.224, 0.225]),
                                        ]),
                                        train=False,
                                        batch_size=cfg.batch_size,
                                        num_workers=cfg.num_workers)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True)
    
    
    print(f"Length of training dataset: {len(train_dataset)}")
    print(f"Length of validation dataset: {len(val_dataset)}")
    print(f"Length of testing dataset: {len(test_dataset)}")
    
    # ================== Training ==================
    
    # Callbacks for logging and checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint_dir,  # Set this to your desired checkpoint directory
        monitor='val_mae',
        save_top_k=1,
        mode='min'
    )
    
    wandb_logger = WandbLogger(project='CSRNet-Light')

    # A custom theme progress bar
    progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
        description="green_yellow",
        progress_bar="green1",
        progress_bar_finished="green1",
        progress_bar_pulse="#6206E0",
        batch_progress="green_yellow",
        time="grey82",
        processing_speed="grey82",
        metrics="grey82",
    ))
    
    #get latest checkpoint
    latest_ckpt = cfg.get_latest_checkpoint(cfg.checkpoint_dir)

    # Setup trainer without resume_from_checkpoint
    trainer = pl.Trainer(
        accelerator='auto',
        max_epochs=cfg.num_epochs,
        callbacks=[checkpoint_callback, progress_bar],
        logger=wandb_logger
    )

    # Train the model with checkpoint if available
    if latest_ckpt:
        print(f"Resuming training from checkpoint: {latest_ckpt}")
        trainer.fit(model, train_loader, val_loader, ckpt_path=latest_ckpt)  # Use ckpt_path here
    else:
        print("Starting training from scratch.")
        trainer.fit(model, train_loader, val_loader)


    print(f"Training completed.")

    trainer.test(model,dataloaders=test_loader)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--train_json', type=str, default='json_files/part_A_train.json')
    parser.add_argument('--val_json', type=str, default='json_files/part_A_val.json')
    parser.add_argument('--test_json', type=str, default='json_files/part_A_test.json')
    parser.add_argument('--num_epochs', type=int, default=400)
    args = parser.parse_args()
    return args   

if __name__ == '__main__':
    args = parse_args()
    cfg.train_json = args.train_json
    cfg.val_json = args.val_json
    cfg.test_json = args.test_json
    cfg.num_epochs = args.num_epochs

    main()
