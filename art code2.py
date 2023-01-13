#!/usr/bin/env python
# coding: utf-8

# In[61]:


import random
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random_seed = 42

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


# In[62]:


model_name = "swsl_resnext50_32x4d"

epoch_size = 30
batch_size = 48

learning_rate = 1e-4
early_stop = 5
k_fold_num = 5


# In[63]:


import timm
import torch.nn as nn 
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from sklearn.metrics import f1_score


# In[64]:


def train(data_loader):
    model = timm.create_model(model_name, pretrained = True, num_classes = 7).to(device=device)
    
    
    class_num = [329, 205, 235, 134, 151, 245, 399]
    class_weight = torch.tensor(np.max(class_num)/class_num).to(device=device, dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    
    
    feature_extractor = [m for n, m in model.named_parameters() if "fc" not in n]
    classifier = [p for p in model.fc.parameters()]
    params = [
        {"params": feature_extractor, "lr": learning_rate * 0.5},
        {"params": classifier, "lr": learning_rate}
    ]
    optimizer = AdamW(params, lr=learning_rate)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min =0)
    
    result = {
        "train_loss" : [],
        "valid_loss" : [],
        "valid_acc" : [],
        "valid_f1" : [],
    }
    
    
    train_loader = data_loader["train_loader"]
    valid_loader = data_loader["valid_loader"]
    
    best_model_state = None
    best_f1 = 0
    early_stop_count = 0
    
    for epoch_idx in range(1, epoch_size + 1):
        model.train()
        
        
        iter_train_loss = []
        iter_valid_loss = []
        iter_valid_acc = []
        iter_valid_f1 = []
        
        for iter_idx, (train_imgs, train_labels) in enumerate(train_loader, 1):
            train_imgs, train_labels = train_imgs.to(device=device, dtype=torch.float), train_labels.to(device)
            
            
            optimizer.zero_grad()


            train_pred = model(train_imgs)
            train_loss = criterion(train_pred, train_labels)
            train_loss.backward()


            optimizer.step()
            iter_train_loss.append(train_loss.cpu().item())

            print(
                f"[Epoch {epoch_idx}/{epoch_size}] model training iteration {iter_idx}/{len(train_loader)} ",end="\r",            
            )
        
        with torch.no_grad():
            for iter_idx, (valid_imgs, valid_labels) in enumerate(valid_loader, 1):
                model.eval()

                valid_imgs, valid_labels = valid_imgs.to(device=device, dtype=torch.float), valid_labels.to(device)


                valid_pred = model(valid_imgs)
                valid_loss = criterion(valid_pred, valid_labels)

                iter_valid_loss.append(valid_loss.cpu().item())

                valid_pred_c = valid_pred.argmax(dim=-1)
                iter_valid_acc.extend((valid_pred_c == valid_labels).cpu().tolist())

                iter_f1_score = f1_score(y_true=valid_labels.cpu().numpy(), y_pred=valid_pred_c.cpu().numpy(), average="macro")
                iter_valid_f1.append(iter_f1_score)

                print(
                    f"[Epoch {epoch_idx}/{epoch_size}] model validation iteration{iter_idx}/{len(valid_loader)}", end="\r",
                )

        epoch_train_loss = np.mean(iter_train_loss)
        epoch_valid_loss = np.mean(iter_valid_loss)
        epoch_valid_acc = np.mean(iter_valid_acc) * 100
        epoch_valid_f1_score = np.mean(iter_valid_f1)

        result["train_loss"].append(epoch_train_loss)
        result["valid_loss"].append(epoch_valid_loss)
        result["valid_acc"].append(epoch_valid_acc)
        result["valid_f1"].append(epoch_valid_f1_score)

        scheduler.step()

        print(
            f"[Epoch {epoch_idx}/{epoch_size}] "
            f"train loss : {epoch_train_loss:.4f} | "
            f"valid loss : {epoch_valid_loss:.4f} | valid acc : {epoch_valid_acc:.2f}% | valid f1 score : {epoch_valid_f1_score:.4f}"
        )

        if epoch_valid_f1_score > best_f1:
            best_f1 = epoch_valid_f1_score
            best_model_state = model.state_dict()
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count == early_stop:
            print("early stoped." + " " * 30)
            break

    return result, best_model_state


# In[65]:


import os 
import cv2
import albumentations as A

from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import Dataset, DataLoader


# In[66]:


class_encoder = {
    'dog' : 0,
    'elephant' : 1,
    'giraffe' : 2,
    'guitar' : 3,
    'horse' : 4,
    'house' : 5,
    'person' : 6
}

def img_gather_(img_path):
    class_list = os.listdir(img_path)
    
    file_lists = []
    label_lists = []
    
    for class_name in class_list:
        file_list = os.listdir(os.path.join(img_path, class_name))
        file_list = list(map(lambda x: "/".join([img_path]+[class_name] + [x]), file_list))
        label_list = [class_encoder[class_name]] * len(file_list)
        
        file_lists.extend(file_list)
        label_lists.extend(label_list)
        
        
    file_lists = np.array(file_lists)
    label_lists = np.array(label_lists)
    
    return file_lists, label_lists



class TrainDataset(Dataset):
    def __init__(self, file_lists, label_lists, transforms=None):
        self.file_lists = file_lists.copy()
        self.label_lists = label_lists.copy()
        self.transforms = transforms
        
    def __getitem__(self, idx):
        img = cv2.imread(self.file_lists[idx], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            img = self.transforms(image=img)["image"]
            
        img = img.transpose(2,0,1)
        
        label = self.label_lists[idx]
        
        img = torch.tensor(img, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)
        
        return img, label
    
    def __len__(self):
        assert len(self.file_lists) == len(self.label_lists)
        return len(self.file_lists)
    
class TestDataset(Dataset):
    def __init__(self, file_lists, transforms=None):
        self.file_lists = file_lists.copy()
        self.transforms = transforms
        
    def __getitem__(self, idx):
        img = cv2.imread(self.file_lists[idx], cv2.IMREAD_COLOR)
        lmg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            img = self.transforms(image=img)["image"]
            
        img = img.transpose(2, 0, 1)
        
        img = torch.tensor(img, dtype=torch.float)
        
        return img
    
    def __len__(self):
        return len(self.file_lists)
    


# In[67]:


train_transforms = A.Compose([
    A.Rotate(),
    A.HorizontalFlip(),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    A.Normalize()
])

valid_transforms = A.Compose([
    A.Normalize()
])


# In[68]:


data_lists, data_labels = img_gather_("./train/train")

best_models = []

if k_fold_num == -1:
    train_lists, valid_lists, train_labels, valid_labels = train_test_split(data_lists, data_lables, train_size = 0.8, random_state = random_seed, stratify=data_labels)
    
    train_dataset = TrainDataset(file_lists=train_lists, label_lists=train_labels, transforms=train_transforms)
    valid_dataset = TrainDataset(file_lists=valid_lists, label_lists=valid_labels, transforms=valid_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    
    
    data_loader = {"train_loader":train_loader, "valid_loader":valid_loader}
    
    print("No fold training starts....")
    train_result, best_model = train(data_loader)
    
    best_models.append(best_model)
    
    
else:
    skf = StratifiedKFold(n_splits=k_fold_num, random_state=random_seed, shuffle=True)
    
    print(f"{k_fold_num} fold training starts...")
    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(data_lists, data_labels), 1):
        print(f"-{fold_idx} fold -")
        train_lists, train_labels = data_lists[train_idx], data_labels[train_idx]
        valid_lists, valid_labels = data_lists[valid_idx], data_labels[valid_idx]
        
        train_dataset = TrainDataset(file_lists=train_lists, label_lists=train_labels, transforms=train_transforms)
        valid_dataset = TrainDataset(file_lists=valid_lists, label_lists=valid_labels, transforms=valid_transforms)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
        
        data_loader = {"train_loader": train_loader, "valid_loader": valid_loader}
        
        train_result, best_model = train(data_loader)
        
        best_models.append(best_model)
    


# In[69]:


test_transforms_ = A.Compose([
    A.Normalize()
])


test_files = os.listdir("./test/test/0")
test_files = sorted(test_files)
test_files = list(map(lambda x: "/".join(["./test/test/0",x]), test_files))

test_dataset = TestDataset(file_lists=test_files, transforms=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# In[70]:


import pandas as pd

answer_logits = []

model = timm.create_model(model_name, pretrained = True, num_classes = 7).to(device=device)

for fold_idx, best_model in enumerate(best_models, 1):
    model.load_state_dict(best_model)
    model.eval()
    
    fold_logits = []
    
    with torch.no_grad():
        for iter_idx, test_imgs in enumerate(test_loader, 1):
            test_imgs = test_imgs.to(device)
            
            test_pred = model(test_imgs)
            fold_logits.extend(test_pred.cpu().tolist())
            
            print(f"[{fold_idx} fold] inference iteration {iter_idx}/{len(test_loader)}" + " " * 10, end = "\r")
            
    answer_logits.append(fold_logits)
        
answer_logits = np.mean(answer_logits, axis=0)
answer_value = np.argmax(answer_logits, axis=-1)

i=0
while True:
    if not os.path.isfile(os.path.join("submissions", f"submission_{i}.csv")):
        submission_path = os.path.join("submissions", f"submissions_{i}.csv")
        break
    i+=1

submission = pd.read_csv("test_answer_sample_.csv", index_col=False)
submission["answer value"] = answer_value
submission["answer value"].to_csv(submission_path)
print("\nAll done")


# In[ ]:




