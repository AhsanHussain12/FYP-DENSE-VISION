from typing import Any, Optional
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch


class CSRNet(pl.LightningModule):

    def __init__(self, learning_rate):
        super().__init__()

        # Define layer configuration
        # M stands for MaxPooling2D
        self.frontend_feats = [
            64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feats = [512, 512, 512, 256, 128, 64]

        # Define block of layers
        self.frontend = self.make_layers(self.frontend_feats)
        self.backend = self.make_layers(
            self.backend_feats, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        self.learning_rate = learning_rate

        # Load weights
        vgg16 = models.vgg16(weights = 'DEFAULT')
        self._initialize_weights()

        # Fetch part of pretrained model
        vgg16_dict = vgg16.state_dict()
        frontend_dict = self.frontend.state_dict()
        transfer_dict = {k: vgg16_dict['features.' + k] for k in frontend_dict}

        # Transfer weights
        self.frontend.load_state_dict(transfer_dict)

        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def make_layers(config, in_channels=3, batch_norm=False, dilation=False):
        d_rate = 2 if dilation else 1
        layers = []

        for filters in config:
            if filters == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                continue

            conv2d = nn.Conv2d(in_channels, filters,
                            kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(filters), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]

            in_channels = filters
        return nn.Sequential(*layers)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.95, weight_decay=5*1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss)
        return loss
    

        


