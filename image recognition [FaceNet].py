#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import pandas as pd
from facenet_pytorch import InceptionResnetV1

class CSVDataset(Dataset):
    def __init__(self, csv_file, label_to_idx, transform=None):
        self.df = pd.read_csv(csv_file)
        self.label_to_idx = label_to_idx
        self.transform = transform
        self.mean = torch.tensor([0.5,0.5,0.5]).view(3,1,1)
        self.std  = torch.tensor([0.5,0.5,0.5]).view(3,1,1)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        img_path = str(self.df.loc[idx,"image"]).strip().strip('"')
        label_str = self.df.loc[idx,"label"].strip().lower()
        label = self.label_to_idx[label_str]
        img = Image.open(img_path).convert("RGB")
        if self.transform: img = self.transform(img)
        img = F.pil_to_tensor(img).float()/255.0
        img = (img - self.mean)/self.std
        return img, label


# In[2]:


train_tfms = transforms.Compose([
    transforms.Resize((160,160)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(12),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])
eval_tfms = transforms.Compose([transforms.Resize((160,160))])


df_all = pd.read_csv("image.csv")
labels_sorted = sorted(df_all["label"].unique())
label_to_idx = {lab:i for i,lab in enumerate(labels_sorted)}
idx_to_label = {i:lab for lab,i in label_to_idx.items()}


train_ds = CSVDataset("train.csv", label_to_idx, transform=train_tfms)
val_ds   = CSVDataset("val.csv",   label_to_idx, transform=eval_tfms)
test_ds  = CSVDataset("test.csv",  label_to_idx, transform=eval_tfms)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=8, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
facenet = InceptionResnetV1(pretrained='vggface2').to(device)


for param in facenet.parameters():
    param.requires_grad = False
for param in facenet.last_linear.parameters():
    param.requires_grad = True

num_classes = len(label_to_idx)
classifier = nn.Linear(512, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    list(facenet.last_linear.parameters()) + list(classifier.parameters()),
    lr=1e-4
)

def get_embeddings(x):
    return facenet(x)


@torch.no_grad()
def evaluate(loader):
    classifier.eval()
    facenet.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        emb = get_embeddings(x)
        logits = classifier(emb)
        loss = criterion(logits,y)
        loss_sum += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds==y).sum().item()
        total += y.size(0)
    return correct/total if total>0 else 0, loss_sum/len(loader)


# In[3]:


epochs = 20
best_val_acc = 0.0
for epoch in range(1,epochs+1):
    classifier.train()
    facenet.train()  
    running_loss = 0.0
    for x,y in train_loader:
        x,y = x.to(device), y.to(device)
        emb = get_embeddings(x)
        logits = classifier(emb)
        loss = criterion(logits,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    val_acc, val_loss = evaluate(val_loader)
    print(f"Epoch {epoch:02d} | train_loss={running_loss/len(train_loader):.4f} | val_acc={val_acc:.3f} | val_loss={val_loss:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "classifier": classifier.state_dict(),
            "facenet": facenet.state_dict()
        }, "best_facenet_classifier.pth")
        print("✅ Saved new best model")
print("Best Validation Accuracy:",best_val_acc)


test_acc, test_loss = evaluate(test_loader)
print(f"Final Test Accuracy: {test_acc:.3f} | Test Loss: {test_loss:.4f}")


# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from facenet_pytorch import InceptionResnetV1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("best_facenet_classifier.pth", map_location=device)

facenet = InceptionResnetV1(pretrained='vggface2').to(device)
classifier = nn.Linear(512, len(pd.read_csv("image.csv")["label"].unique())).to(device)

classifier.load_state_dict(checkpoint["classifier"])
facenet.load_state_dict(checkpoint["facenet"])
classifier.eval()
facenet.eval()


df_all = pd.read_csv("image.csv")
labels_unique = list(df_all["label"].unique())
label_to_idx = {lab.lower():i for i,lab in enumerate(labels_unique)}
idx_to_label = {i:lab for lab,i in label_to_idx.items()}


infer_tfms = transforms.Compose([transforms.Resize((160,160))])
mean = torch.tensor([0.5,0.5,0.5]).view(3,1,1)
std  = torch.tensor([0.5,0.5,0.5]).view(3,1,1)



# In[3]:


def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img_proc = infer_tfms(img)
    img_tensor = F.pil_to_tensor(img_proc).float()/255.0
    img_tensor = (img_tensor - mean)/std
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        emb = facenet(img_tensor)
        logits = classifier(emb)
        pred_idx = logits.argmax(dim=1).item()
        pred_label = idx_to_label[pred_idx]

    
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Predicted: {pred_label}")
    plt.show()

    print(f"Image Path: {img_path}")
    print(f"Predicted label: {pred_label}")
    return pred_label

img_path = input("Enter image path: ")
predict_image(img_path)


# In[ ]:




