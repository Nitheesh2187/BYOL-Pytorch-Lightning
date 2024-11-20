import torch
from torchvision import transforms as T
import torchvision.transforms.functional as TF
import random
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import copy
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import torch
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

import warnings

warnings.filterwarnings("ignore")

def reproducibility(SEED):
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

def loss_fn(x, y):
    # L2 normalization
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


class EMA():
   def __init__(self, alpha):
       super().__init__()
       self.alpha = alpha

   def update_average(self, old, new):
       if old is None:
           return new
       return old * self.alpha + (1 - self.alpha) * new

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


class AddProjHead(nn.Module):
    def __init__(self, model, in_features, layer_name, hidden_size=4096,
                 embedding_size=256, batch_norm_mlp=True):
        super(AddProjHead, self).__init__()
        self.backbone = model
        # remove last layer 'fc' or 'classifier'
        setattr(self.backbone, layer_name, nn.Identity())
        self.backbone.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = torch.nn.Identity()
        # add mlp projection head
        self.projection = MLP(in_features, embedding_size, hidden_size=hidden_size, batch_norm_mlp=batch_norm_mlp)

    def forward(self, x, return_embedding=False):
        embedding = self.backbone(x)
        if return_embedding:
            return embedding
        return self.projection(embedding)

class MLP(nn.Module):
    def __init__(self, dim, embedding_size=256, hidden_size=2048, batch_norm_mlp=False):
        super().__init__()
        norm = nn.BatchNorm1d(hidden_size) if batch_norm_mlp else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            norm,
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, embedding_size)
        )

    def forward(self, x):
        return self.net(x)
    
class BYOL(nn.Module):
    def __init__(
            self,
            net,
            batch_norm_mlp=True,
            layer_name='fc',
            in_features=512,
            projection_size=256,
            projection_hidden_size=2048,
            moving_average_decay=0.99,
            use_momentum=True):
        """
        Args:
            net: model to be trained
            batch_norm_mlp: whether to use batchnorm1d in the mlp predictor and projector
            in_features: the number features that are produced by the backbone net i.e. resnet
            projection_size: the size of the output vector of the two identical MLPs
            projection_hidden_size: the size of the hidden vector of the two identical MLPs
            augment_fn2: apply different augmentation the second view
            moving_average_decay: t hyperparameter to control the influence in the target network weight update
            use_momentum: whether to update the target network
        """
        super().__init__()
        self.net = net
        self.student_model = AddProjHead(model=net, in_features=in_features,
                                         layer_name=layer_name,
                                         embedding_size=projection_size,
                                         hidden_size=projection_hidden_size,
                                         batch_norm_mlp=batch_norm_mlp)
        self.use_momentum = use_momentum
        self.teacher_model = self._get_teacher()
        self.target_ema_updater = EMA(moving_average_decay)
        self.student_predictor = MLP(projection_size, projection_size, projection_hidden_size)
    
    @torch.no_grad()
    def _get_teacher(self):
        return copy.deepcopy(self.student_model)
    
    @torch.no_grad()
    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum ' \
                                  'for the target encoder '
        assert self.teacher_model is not None, 'target encoder has not been created yet'

        for student_params, teacher_params in zip(self.student_model.parameters(), self.teacher_model.parameters()):
          old_weight, up_weight = teacher_params.data, student_params.data
          teacher_params.data = self.target_ema_updater.update_average(old_weight, up_weight)

    def forward(
            self,
            image_one, image_two=None,
            return_embedding=False):
        if return_embedding or (image_two is None):
            return self.student_model(image_one, return_embedding=True)

        # student projections: backbone + MLP projection
        student_proj_one = self.student_model(image_one)
        student_proj_two = self.student_model(image_two)

        # additional student's MLP head called predictor
        student_pred_one = self.student_predictor(student_proj_one)
        student_pred_two = self.student_predictor(student_proj_two)

        with torch.no_grad():
            # teacher processes the images and makes projections: backbone + MLP
            teacher_proj_one = self.teacher_model(image_one).detach_()
            teacher_proj_two = self.teacher_model(image_two).detach_()
            
        loss_one = loss_fn(student_pred_one, teacher_proj_one)
        loss_two = loss_fn(student_pred_two, teacher_proj_two)        

        return (loss_one + loss_two).mean()



class KNN():
    def __init__(self, model, k, device):
        super(KNN, self).__init__()
        self.k = k
        self.device = device
        self.model = model.to(device)
        self.model.eval()

    def extract_features(self, loader):
        """
        Infer/Extract features from a trained model
        Args:
            loader: train or test loader
        Returns: 3 tensors of all:  input_images, features , labels
        """
        x_lst = []
        features = []
        label_lst = []

        with torch.no_grad():
            for input_tensor, label in loader:
                h = self.model(input_tensor.to(self.device))
                features.append(h)
                x_lst.append(input_tensor)
                label_lst.append(label)

            x_total = torch.stack(x_lst)
            h_total = torch.stack(features)
            label_total = torch.stack(label_lst)

            return x_total, h_total, label_total

    def knn(self, features, labels, k=1):
        """
        Evaluating knn accuracy in feature space.
        Calculates only top-1 accuracy (returns 0 for top-5)
        Args:
            features: [... , dataset_size, feat_dim]
            labels: [... , dataset_size]
            k: nearest neighbours
        Returns: train accuracy, or train and test acc
        """
        feature_dim = features.shape[-1]
        with torch.no_grad():
            features_np = features.cpu().view(-1, feature_dim).numpy()
            labels_np = labels.cpu().view(-1).numpy()
            # fit
            self.cls = KNeighborsClassifier(k, metric="cosine").fit(features_np, labels_np)
            acc = self.eval(features, labels)
            
        return acc
    
    def eval(self, features, labels):
      feature_dim = features.shape[-1]
      features = features.cpu().view(-1, feature_dim).numpy()
      labels = labels.cpu().view(-1).numpy()
      acc = 100 * np.mean(cross_val_score(self.cls, features, labels))
      return acc

    def _find_best_indices(self, h_query, h_ref):
        h_query = h_query / h_query.norm(dim=1).view(-1, 1)
        h_ref = h_ref / h_ref.norm(dim=1).view(-1, 1)
        scores = torch.matmul(h_query, h_ref.t())  # [query_bs, ref_bs]
        score, indices = scores.topk(1, dim=1)  # select top k best
        return score, indices

    def fit(self, train_loader, test_loader=None):
        with torch.no_grad():
            x_train, h_train, l_train = self.extract_features(train_loader)
            train_acc = self.knn(h_train, l_train, k=self.k)

            if test_loader is not None:
                x_test, h_test, l_test = self.extract_features(test_loader)
                test_acc = self.eval(h_test, l_test)
                return train_acc, test_acc

def get_cifar10_dataloader(batch_size, train=True, transform=Augment(224)):
    dataset = CIFAR10(root="./", train=train, transform=transform, download=True)
    return DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4, drop_last=True)
data_transform = Augment(32)
dataloader = get_cifar10_dataloader(batch_size=64, train=True, transform=data_transform)

def training_step(model, data):
    (view1, view2), _ = data
    loss = model(view1.cuda(), view2.cuda())
    return loss

def train_one_epoch(model, train_dataloader, optimizer):
    model.train()
    total_loss = 0.
    num_batches = len(train_dataloader)
    for data in train_dataloader:
        optimizer.zero_grad()
        loss = training_step(model, data)
        loss.backward()
        optimizer.step()
        # EMA update
        model.update_moving_average()

        total_loss += loss.item()
        
    
    return total_loss/num_batches

load = False
model = models.resnet18(pretrained=False)
model = BYOL(model, in_features=512, batch_norm_mlp=True)
model.cuda()

# optimizer and loss
lr = 3e-4
weight_decay = 0.000001
BATCH_SIZE  = 256

#param_groups = define_param_groups(model, weight_decay, 'lars')    
#optimizer = LARS(param_groups, lr=0.1, momentum=0.9)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# data
data_transform = Augment(32)

test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])


loader_train = get_cifar10_dataloader(batch_size=BATCH_SIZE, train=True, transform=data_transform)
loader_train_plain = get_cifar10_dataloader(batch_size=BATCH_SIZE, train=True, transform=test_transform)
loader_test = get_cifar10_dataloader(batch_size=BATCH_SIZE, train=False, transform=test_transform)


# general info
available_gpus = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])
print('available_gpus:',available_gpus)


reproducibility(9999)

epochs = 100
mean_losses = []
train_knns = []
val_knns = []

for i in range(epochs):
  mean_loss = train_one_epoch(model, loader_train, optimizer)
  mean_losses.append(mean_loss)
  print("EPOCH",i+1,"Mean_loss:",mean_loss)
  if (i%4)==0:
    # KNN evaluation
    ssl_evaluator = KNN(model=model, k=1, device='cuda')
    train_acc, val_acc = ssl_evaluator.fit(loader_train_plain, loader_test)
    print(f'\n Epoch {i}: loss:{mean_loss}')
    print(f"k-nn accuracy k= {ssl_evaluator.k} for train split: {train_acc}")
    print(f"k-nn accuracy k= {ssl_evaluator.k} for val split: {val_acc} \n")
    print('-----------------')
    train_knns.append(train_acc)
    val_knns.append(val_acc)