from typing import Any, Callable
from pytorch_lightning.core.optimizer import LightningOptimizer
import torch
from torch.optim.optimizer import Optimizer
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import copy
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import random_split

import warnings
warnings.filterwarnings("ignore")

def reproducibility(SEED):
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

#DataTransform class
class Augment:
    """
    A stochastic data augmentation module
    Transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """
    def __init__(self, img_size, s=1):
        color_jitter = T.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        blur = T.GaussianBlur((3, 3), (0.1, 2.0))

        self.train_transform = T.Compose([
            T.ToTensor(),
            T.RandomResizedCrop(size=img_size),
            T.RandomHorizontalFlip(p=0.5),  # with 0.5 probability
            T.RandomApply([color_jitter], p=0.8),
            T.RandomApply([blur], p=0.5),
            T.RandomGrayscale(p=0.2),
            # imagenet stats
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])            
        ])
    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x),

def loss_fn(x, y):
    # L2 normalization
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

def get_cifar10_dataloader(batch_size, train=True, transform=Augment(224)):
    dataset = CIFAR10(root="./", train=train, transform=transform, download=True)
    return DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4, drop_last=True)

#Weight Update Class for Target Network
class EMA():
   def __init__(self, alpha):
       super().__init__()
       self.alpha = alpha

   def update_average(self, old, new):
       if old is None:
           return new
       return old * self.alpha + (1 - self.alpha) * new


class BYOL(pl.LightningModule):
    def __init__(self,backbone,
                 lr,
                 layer_name="fc",
                 in_features=512,#Projection Input
                 hidden_size=2048,
                 embedding_size=256,#Output size of Projection
                 moving_average_decay=0.99,
                 batch_norm_mlp=True):
        super().__init__()
        #Online Netork
        self.online_representation = backbone
        # remove last layer 'fc' or 'classifier'
        setattr(self.online_representation, layer_name, nn.Identity())
        self.online_representation.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.online_representation.maxpool = torch.nn.Identity()
        self.online_projection = self.MLP_Layer(in_features,embedding_size,hidden_size,batch_norm_mlp)
        self.online_prediction = self.MLP_Layer(embedding_size,embedding_size,hidden_size,batch_norm_mlp)
        self.target_ema_updater = EMA(moving_average_decay)
        self.lr = lr
        self.save_hyperparameters()

        #Target Netork
        self.target_representation = copy.deepcopy(self.online_representation)
        self.target_projection = copy.deepcopy(self.online_projection)

        #Freezing the Target Network
        for param in self.target_representation.parameters():
            param.requires_grad = False
        for param in self.target_projection.parameters():
            param.requires_grad = False
            

    def MLP_Layer(self,dim,embedding_size,hidden_size,batch_norm_mlp):
        norm = nn.BatchNorm1d(hidden_size) if batch_norm_mlp else nn.Identity()   

        return nn.Sequential(
            nn.Linear(dim, hidden_size),
            norm,
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, embedding_size)
        )
    @torch.no_grad()
    def update_moving_average(self):
        assert self.target_representation is not None, 'target encoder has not been created yet'

        for online_representation_params, target_representation_params in zip(self.online_representation.parameters(), self.target_representation.parameters()):
          old_weight, up_weight = target_representation_params.data, online_representation_params.data
          target_representation_params.data = self.target_ema_updater.update_average(old_weight, up_weight)

        for online_projection_params, target_projection_params in zip(self.online_projection.parameters(), self.target_projection.parameters()):
          old_weight, up_weight = target_projection_params.data, online_projection_params.data
          target_projection_params.data = self.target_ema_updater.update_average(old_weight, up_weight)

    def forward(self, image_one, image_two=None,return_embedding=False):
        if return_embedding or (image_two is None):
            return self.online_representation(image_one)

        online_forward = self.online_prediction(self.online_projection(self.online_representation(image_one)))
        target_forward = self.target_projection(self.target_representation(image_one))
        return online_forward,target_forward
    
    def training_step(self,batch,batch_idx):
        (view1,view2),_ = batch
        online_view1,target_view2 = self(view1,view2)
        online_view2,online_view1 = self(view2,view1)

        loss_one = loss_fn(online_view1, target_view2)
        loss_two = loss_fn(online_view2, target_view2)

        loss = (loss_one + loss_two).mean()

        self.log(
            "train_loss",loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss
    
    def optimizer_step(self, epoch: int, batch_idx: int, optimizer: Optimizer | LightningOptimizer, optimizer_closure: Callable[[], Any] | None = None) -> None:
        # Update Online params
        optimizer.step(closure=optimizer_closure)
        #update Target Params
        self.update_moving_average()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


# optimizer and loss
lr = 3e-4
weight_decay = 0.000001
BATCH_SIZE  = 256
epochs = 100


#DataLoaders
data_transform = Augment(32)
loader_train = get_cifar10_dataloader(batch_size=BATCH_SIZE, train=True, transform=data_transform)  

model_checkpoint_callback = ModelCheckpoint("Checkpoints/",filename="model-{epoch:02d}-{val_loss:.2f}",save_top_k=1,
                                            monitor="train_loss",mode="min",save_weights_only=True)

reproducibility(9999)

#Model
backbone = models.resnet18(pretrained=False)
model = BYOL(backbone=backbone,
             lr=lr,
             in_features=512,
             )

trainer = pl.Trainer(accelerator="gpu",
        devices=[0],
        min_epochs=1,
        max_epochs=epochs,
        callbacks=model_checkpoint_callback)
trainer.fit(model,loader_train)


#Linear Evaluation
class Linear_Model(pl.LightningModule):
    def __init__(self,backbone,in_features,num_classes,lr):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        self.in_features = in_features
        self.lr = lr
        self.fc = torch.nn.Linear(in_features,num_classes)

    def forward(self, x):
        features = self.backbone(x)
        scores = self.fc(features)
        probabilites = torch.softmax(scores,dim=1)
        # predictions = torch.argmax(probabilites,dim=1)
        return probabilites
    
    def training_step(self,batch,batch_idx):
        X,y = batch
        outputs = self(X)
        loss = F.cross_entropy(outputs,y) 
        self.log("train_Loss",loss,prog_bar=True,on_step=True,on_epoch=True)     
        return loss
    
    def validation_step(self,batch,batch_idx):
        X,y = batch
        outputs = self(X)
        loss = F.cross_entropy(outputs,y) 
        self.log("val_Loss",loss,prog_bar=True,on_step=True,on_epoch=True)     
        return loss
    
    def test_step(self,batch,batch_idx):
        X,y = batch
        outputs = self(X)
        preds = torch.argmax(outputs,dim=1)
        corrects = torch.sum(preds==y)
        accuracy = corrects.double() / len(y)
        loss = F.cross_entropy(outputs,y) 
        self.log_dict({
                        "test_Loss":loss,
                        "accuracy":accuracy
                       
                       },
                       prog_bar=True,on_step=True,on_epoch=True)     
        return loss
    
    def predict_step(self,batch,batch_idx):
        X,y = batch
        outputs = self(X)
        preds = torch.argmax(outputs,dim=1)
        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
lr = 0.2
in_features = 512
num_classes =10

for params in model.parameters():
    params.requires_grad = False

# Transformations and Dataset
train_transform = T.Compose([T.RandomResizedCrop(32), 
                                      T.RandomHorizontalFlip(p=0.5), 
                                      T.ToTensor()])
test_transform = T.ToTensor()

train_dataset = CIFAR10("./",train=True,transform=train_transform,download=True)
test_dataset = CIFAR10("./",train=False,transform=test_transform,download=True)
train_size = int(0.7*len(train_dataset))
val_size = len(train_dataset) - train_size

train_dataset,val_dataset = random_split(train_dataset,[train_size,val_size])


#Dataloaders
train_loader = DataLoader(train_dataset,batch_size=128,shuffle=True,num_workers=4)
val_loader = DataLoader(val_dataset,batch_size=128,shuffle=False,num_workers=4)
test_loader = DataLoader(test_dataset,batch_size=128,shuffle=False,num_workers=4)


eval_model = Linear_Model(model,
                          in_features=in_features,
                          num_classes=num_classes,
                          lr=lr)

trainer = pl.Trainer(accelerator="gpu",
                     devices=[0],
                     min_epochs=1,
                     max_epochs=50,
                     )
trainer.fit(eval_model,train_loader)
trainer.validate(eval_model,val_loader)
trainer.test(eval_model,test_loader)