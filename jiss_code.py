import warnings
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
import clip
import pandas as pd
import numpy as np
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
import clip
from PIL import Image
# !pip install git+https://github.com/openai/CLIP.git
device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model, compose = clip.load('ViT-L/14', device = device)
# text_model = text_model.cpu()
def process(idx_val,arr):
  if idx_val=='0':
    arr.append(0)
  else:
    arr.append(1)

import pandas as pd

# %%
data = pd.read_csv('/DATA/gitanjali_2021cs03/CLIP/work_on_bias_final/jiis/offensive_political.csv')

# %%
data.head(10)

# %%


# %%
# !pip install multilingual-clip

# %%
from multilingual_clip import pt_multilingual_clip
import transformers

# %%
from PIL import ImageFile
from PIL import Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

# %%
len(data)

from IPython.display import Image, display
import PIL.Image
import io
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score,precision_score
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
import os
from transformers import BertModel,ViTModel
import torch
import numpy as np
from transformers import VisualBertForQuestionAnswering, BertTokenizerFast
import pandas as pd
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# %%
from IPython.display import Image, display
import PIL.Image
import io
pred_e = 0
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score,precision_score
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score
import os

# %%
from transformers import AutoTokenizer, VisualBertModel
import torch

import pytorch_lightning as pl

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
class MFB(nn.Module):
    def __init__(self,img_feat_size, ques_feat_size, is_first, MFB_K, MFB_O, DROPOUT_R):
        super(MFB, self).__init__()
        #self.__C = __C
        self.MFB_K = MFB_K
        self.MFB_O = MFB_O
        self.DROPOUT_R = DROPOUT_R

        self.is_first = is_first
        self.proj_i = nn.Linear(img_feat_size, MFB_K * MFB_O)
        self.proj_q = nn.Linear(ques_feat_size, MFB_K * MFB_O)

        self.dropout = nn.Dropout(DROPOUT_R)
        self.pool = nn.AvgPool1d(MFB_K, stride = MFB_K)

    def forward(self, img_feat, ques_feat, exp_in=1):
        '''
            img_feat.size() -> (N, C, img_feat_size)    C = 1 or 100
            ques_feat.size() -> (N, 1, ques_feat_size)
            z.size() -> (N, C, MFB_O)
            exp_out.size() -> (N, C, K*O)
        '''
        batch_size = img_feat.shape[0]
        img_feat = self.proj_i(img_feat)                # (N, C, K*O)
        ques_feat = self.proj_q(ques_feat)              # (N, 1, K*O)

        exp_out = img_feat * ques_feat             # (N, C, K*O)
        exp_out = self.dropout(exp_out) if self.is_first else self.dropout(exp_out * exp_in)     # (N, C, K*O)
        z = self.pool(exp_out) * self.MFB_K         # (N, C, O)
        z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
        z = F.normalize(z.view(batch_size, -1))         # (N, C*O)
        z = z.view(batch_size, -1, self.MFB_O)      # (N, C, O)
        return z

# %%
texts = [
    'Three blind horses listening to Mozart.',
    'Älgen är skogens konung!',
    'Wie leben Eisbären in der Antarktis?',
    'Вы знали, что все белые медведи левши?'
]
model_name = 'M-CLIP/XLM-Roberta-Large-Vit-L-14'
# Load Model & Tokenizer
model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
embeddings = model.forward(texts, tokenizer)
print(embeddings.shape)
# clip_model, compose = clip.load('RN50x4', device = device)
# clip_model, compose = clip.load("ViT-B/32", device = device)
# text_inputs = (clip.tokenize(data.Text_Transcription.values[321],truncate=True)).to(device)
# print(text_inputs)
import pickle
def get_data(data):
  #data = pd.read_csv(dataset_path)
  text = list(data['Text'])
  img_path = list(data['file_name'])
  name = list(data['file_name'])
  label = list(data['Level1'])
  pol = list(data['political'])

  #optimize memory for features
  text_features,image_features,l,Name,v = [],[],[],[],[]
  bias_new=[]
  for txt,img,L,n in tqdm(zip(text,img_path,label,name)):
    try:
      #img = preprocess(Image.open('/content/drive/.shortcut-targets-by-id/1Z57L19m3ZpJ6bEPdyaIMYuI00Tc2RT1I/memes_our_dataset_hindi/my_meme_data/'+img)).unsqueeze(0).to(device)
      img = Image.open('/DATA/gitanjali_2021cs03/CLIP/work_on_bias_final/training_images/'+img)
    except Exception as e:
      print(e)
      continue

    img = torch.stack([compose(img).to(device)])
    l.append(L)
    Name.append(n)
    # bias_new.append(bias_list_check)
    # v.append(V)
    #txt = torch.as_tensor(txt)
    with torch.no_grad():
      temp_txt=model.forward(txt, tokenizer).detach().cpu().numpy()
      text_features.append(temp_txt)
      temp_img = clip_model.encode_image(img).detach().cpu().numpy()
      image_features.append(temp_img)

      del temp_txt
      del temp_img

      torch.cuda.empty_cache()

    del img
    #del txtla
    torch.cuda.empty_cache()
  return text_features,image_features,l,pol,Name



class HatefulDataset(Dataset):

  def __init__(self,data):
    self.t_f,self.i_f,self.label,self.pol,self.name = get_data(data)
    self.t_f = np.squeeze(np.asarray(self.t_f),axis=1)
    self.i_f = np.squeeze(np.asarray(self.i_f),axis=1)



  def __len__(self):
    return len(self.label)

  def __getitem__(self,idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    #print(idx)
    name=self.name[idx]
    label = self.label[idx]
    pol = self.pol[idx]

    T = self.t_f[idx,:]
    I = self.i_f[idx,:]

    # v = self.v[idx]
    # a = self.a[idx]
    sample = {'label':label,'processed_txt':T,'processed_img':I,'pol':pol,'name':name}
    return sample


# %%
sample_dataset = HatefulDataset(data)
hm_data = torch.load('/DATA/gitanjali_2021cs03/CLIP/work_on_bias_final/off_pol_meme.pt')

# %%
torch.manual_seed(123)
t_p,te_p = torch.utils.data.random_split(sample_dataset,[2000,334])

torch.manual_seed(123)
t_p,v_p = torch.utils.data.random_split(t_p,[1500,500])

# %%
t_p[1]["processed_txt"].shape
# %%
outliers = ['ravan38.png',
'ravan283.png',
'ravan26.png',
'ravan283.png',
'ravan282.png',
'rel563.png',
'ravan341.png',
'politics363.png',
'ravan26.png',
'ravan284.png',
'hin32.png',
'ravan299.png',
'ravan296.png',
'ravan25.png',
'file_new_426.png',
'm_50.png',
'test31.png',
'ravan24.png',
'rel563.png',
'kalam360.png',
'chaukidar43.png',
'chaukidar37.png',
'ravan23.png',
'ravan344.png',
'ravan283.png',
'gandhiji77.png',
'test25.png',
'ravan342.png',
'ravan38.png',
'mix_meme7.png',
'chaukidar37.png',
'match251.png',
'file_new_5.png',
'ravan284.png',
'chaukidar37.png',
'ravan284.png',
'ravan26.png',
'rel563.png',
'ravan2.png',
'ravan343.png',
'ravan20.png',
'ravan38.png']

# %%
# filtering out the rows which contain the outlier images
data = data[~data['Name'].isin(outliers)]

# %%
len(data)
from pytorch_lightning.callbacks import ModelCheckpoint

# %%

import torch
from torch import nn
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score,precision_score
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import os

class offensive(nn.Module):
   def __init__(self):
    super().__init__()

    # self.MFB = MFB(640,640,True,256,64,0.1)
    self.MFB = MFB(512,768,True,256,64,0.1)
    self.fin = torch.nn.Linear(64,2)

    def forward(self, x,y):
      x_,y_ = x,y
      z = self.MFB(torch.unsqueeze(y,axis=1),torch.unsqueeze(x,axis=1))

      c = self.fin(torch.squeeze(z,dim=1))
      # probability distribution over labels
      c = torch.log_softmax(c, dim=1)
      return c


class adversarial(nn.Module):
   def __init__(self):
    super().__init__()

    # self.MFB = MFB(640,640,True,256,64,0.1)
    self.MFB = MFB(512,768,True,256,64,0.1)
    self.fin = torch.nn.Linear(64,2)

    def forward(self, x,y):
      x_,y_ = x,y
      z = self.MFB(torch.unsqueeze(y,axis=1),torch.unsqueeze(x,axis=1))

      c = self.fin(torch.squeeze(z,dim=1))
      # probability distribution over labels
      c = torch.log_softmax(c, dim=1)
      return c


class Classifier(pl.LightningModule):
  def __init__(self,offensive,adversarial):
    super().__init__()

    # self.MFB = MFB(640,640,True,256,64,0.1)
    self.offensive= offensive
    self.adversarial= adversarial
    self.MFB = MFB(512,768,True,256,64,0.1)
    self.fin = torch.nn.Linear(64,2)
    self.pretraining = True

  def forward(self, x,y):
      x_,y_ = x,y
      z = self.MFB(torch.unsqueeze(y,axis=1),torch.unsqueeze(x,axis=1))

      c = self.fin(torch.squeeze(z,dim=1))
      # probability distribution over labels
      c = torch.log_softmax(c, dim=1)
      return c

  def cross_entropy_loss(self, logits, labels):
    return F.nll_loss(logits, labels)

  def training_step(self, train_batch, batch_idx):
      lab,txt,img,pol,name= train_batch

      lab = train_batch[lab]
      pol = train_batch[pol]
      #print(lab)
      txt = train_batch[txt]
      #print(txt)

      img = train_batch[img]
      if self.pretraining:
         out= self.offensive(txt, img)
         off_loss = F.cross_entropy(out, lab)
      else:
         out_pol= self.adversarial(txt, img)
         pol_loss = F.cross_entropy(out_pol, pol)
      # print(logit_offen)
      loss = self.alpha * off_loss + (1 - self.alpha) * pol_loss

      self.log('train_loss', loss)
      return loss

  def validation_step(self, val_batch, batch_idx):
      lab,txt,img,pol,name= val_batch
      lab = val_batch[lab]
      txt = val_batch[txt]
      img = val_batch[img]
      logits = self.forward(txt,img)
      tmp = np.argmax(logits.detach().cpu().numpy(),axis=-1)
      loss = self.cross_entropy_loss(logits, lab)
      lab = lab.detach().cpu().numpy()
      self.log('val_acc', accuracy_score(lab,tmp))
      self.log('val_roc_auc',roc_auc_score(lab,tmp))
      self.log('val_loss', loss)
      tqdm_dict = {'val_acc': accuracy_score(lab,tmp)}
      return {
                'progress_bar': tqdm_dict,
      'val_f1 offensive': f1_score(lab,tmp,average='macro')
      }

  def validation_epoch_end(self, validation_step_outputs):
    outs = []
    outs1,outs2,outs3,outs4,outs5,outs6,outs7,outs8,outs9,outs10,outs11,outs12,outs13,outs14,outs16,outs17 = \
    [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    for out in validation_step_outputs:
      outs.append(out['progress_bar']['val_acc'])
      outs14.append(out['val_f1 offensive'])
    self.log('val_acc_all_offn', sum(outs)/len(outs))
    self.log('val_f1 offensive', sum(outs14)/len(outs14))
    print(f'***val_acc_all_offn at epoch end {sum(outs)/len(outs)}****')
    print(f'***val_f1 offensive at epoch end {sum(outs14)/len(outs14)}****')

  def test_step(self, batch, batch_idx):
      lab,txt,img,pol,name= batch
      lab = batch[lab]
      txt = batch[txt]
      img = batch[img]
      logits = self.forward(txt,img)
      tmp = np.argmax(logits.detach().cpu().numpy(),axis=-1)
      loss = self.cross_entropy_loss(logits, lab)
      lab = lab.detach().cpu().numpy()
      self.log('test_acc', accuracy_score(lab,tmp))
      self.log('test_roc_auc',roc_auc_score(lab,tmp))
      self.log('test_loss', loss)
      tqdm_dict = {'test_acc': accuracy_score(lab,tmp)}
      #print('Val acc {}'.format(accuracy_score(lab,tmp)))
      return {
                'progress_bar': tqdm_dict,
                'test_acc': accuracy_score(lab,tmp),
                'test_f1_score': f1_score(lab,tmp,average='macro'),
      }
  def test_epoch_end(self, outputs):
      # OPTIONAL
      outs = []
      outs1,outs2,outs3,outs4,outs5,outs6,outs7,outs8,outs9,outs10,outs11,outs12,outs13,outs14 = \
      [],[],[],[],[],[],[],[],[],[],[],[],[],[]
      for out in outputs:
        # outs15.append(out['test_loss_target'])
        outs.append(out['test_acc'])
        outs2.append(out['test_f1_score'])
      self.log('test_acc', sum(outs)/len(outs))
      self.log('test_f1_score', sum(outs2)/len(outs2))

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=5e-3)
    return optimizer
  def on_epoch_end(self):
    if self.current_epoch == self.trainer.max_epochs // 2:
        self.pretraining = False  # Transition to the second phase of training

class HmDataModule(pl.LightningDataModule):

  def setup(self, stage):
    self.hm_train = t_p
    self.hm_val = v_p
    self.hm_test = te_p
  def train_dataloader(self):
    return DataLoader(self.hm_train, batch_size=64)

  def val_dataloader(self):
    return DataLoader(self.hm_val, batch_size=64)

  def test_dataloader(self):
    return DataLoader(self.hm_test, batch_size=128)

data_module = HmDataModule()


checkpoint_callback = ModelCheckpoint(
     monitor='val_acc_all_offn',
     dirpath='ckpt_JIIS/',
     filename='epoch{epoch:02d}-val_f1_all_offn{val_acc_all_offn:.2f}',
     auto_insert_metric_name=False,
     save_top_k=1,
    mode="max",
 )
all_callbacks = []
all_callbacks.append(checkpoint_callback)
# train
from pytorch_lightning import seed_everything
seed_everything(42, workers=True)
hm_model = Classifier()
gpus = 1 if torch.cuda.is_available() else 0
trainer = pl.Trainer(gpus=gpus,deterministic=True,max_epochs=10,precision=16,callbacks=all_callbacks)
trainer.fit(hm_model, data_module)
# %%
test_dataloader = DataLoader(dataset=te_p, batch_size=1478)
ckpt_path = '/DATA/gitanjali_2021cs03/CLIP/work_on_bias_final/ckpt_JIIS/epoch06-val_f1_all_offn0.83.ckpt' # put ckpt_path according to the path output in the previous cell
trainer.test(dataloaders=test_dataloader,ckpt_path=ckpt_path)