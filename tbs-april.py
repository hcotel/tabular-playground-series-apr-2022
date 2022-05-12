##########################Load Libraries  ####################################
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import random
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import RichProgressBar, LearningRateMonitor, Callback
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torchmetrics

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import os
from einops import rearrange
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertLayer as _BertLayer
import gc

gc.collect()
torch.cuda.empty_cache()
device = 'cuda'

df_train = pd.read_csv('data/train.csv')
t_lbls = pd.read_csv('data/train_labels.csv')
df_test = pd.read_csv('data/test.csv')
ss = pd.read_csv('data/sample_submission.csv')
features = df_train.columns.tolist()[3:]
def prep(df):
    for feature in features:
        df[feature + '_lag1'] = df.groupby('sequence')[feature].shift(1)
        df[feature + '_lead1'] = df.groupby('sequence')[feature].shift(-1)
        df.fillna(0, inplace=True)
        df[feature + '_diff1'] = df[feature] - df[feature + '_lag1']

prep(df_train)
prep(df_test)
features = df_train.columns.tolist()[3:]
df_train = pd.merge(df_train, t_lbls, on="sequence")
df_test['state'] = 0


sc = StandardScaler()
df_train[features] = sc.fit_transform(df_train[features])
df_test[features] = sc.transform(df_test[features])

F_train = df_train[features].values.reshape(df_train.shape[0] // 60, 60, len(features))
F_test = df_test[features].values.reshape(df_test.shape[0] // 60, 60, len(features))

train_subject_counts = (df_train.subject.value_counts() / 60).astype(int)
test_subject_counts = (df_test.subject.value_counts() / 60).astype(int)
index_df_train = df_train[["sequence", "subject", "state"]].drop_duplicates()
index_df_test = df_test[["sequence", "subject", "state"]].drop_duplicates()
# index_df_train['length'] = train_subject_counts[index_df_train.subject]
# index_df_test['length'] = test_subject_counts[index_df_test.subject]

F_train_subject = index_df_train.subject.values
F_test_subject = index_df_test.subject.values

SEED = 22
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


trial = 32
patiance = 7
seed_everything(SEED)

class GlobalMaxPooling1D(nn.Module):

    def __init__(self, data_format='channels_last'):
        super(GlobalMaxPooling1D, self).__init__()
        self.data_format = data_format
        self.step_axis = 1 if self.data_format == 'channels_last' else 2

    def forward(self, input):
        return torch.max(input, axis=self.step_axis).values

class GlobalAvgPooling1D(nn.Module):

    def __init__(self, data_format='channels_last'):
        super(GlobalAvgPooling1D, self).__init__()
        self.data_format = data_format
        self.step_axis = 1 if self.data_format == 'channels_last' else 2

    def forward(self, input):
        return torch.mean(input, dim=self.step_axis)


class TPSAprilNet(pl.LightningModule):
    def __init__(self):
        super(TPSAprilNet, self).__init__()

        self.bi_lstm1 = nn.LSTM(13*4, 512, bidirectional=True, batch_first=True)
        self.bi_lstm2 = nn.LSTM(1024, 256, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(1024, 256, bidirectional=True, batch_first=True)
        self.bi_lstm3 = nn.LSTM(1024, 128, bidirectional=True, batch_first=True)
        self.avg_pool = GlobalMaxPooling1D()
        self.dense = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.SELU(),
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x, x_l):
        x_1, _ = self.bi_lstm1(x[0])
        x_2, _ = self.bi_lstm2(x_1)
        x_3, _ = self.gru(x_1)
        x_cat = torch.cat([x_2, x_3], dim=2)
        x_4, _ = self.bi_lstm3(x_cat)
        x_4 = self.avg_pool(x_4)
        output = self.dense(x_4)
        return output

    def training_step(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        pass


class TPSAprilNet_2(pl.LightningModule):
    def __init__(self):
        super(TPSAprilNet_2, self).__init__()

        self.emb = nn.Embedding(991, 1024)

        self.conv1 = nn.Conv1d(in_channels=13*4, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=384, kernel_size=3)

        self.bi_lstm1 = nn.LSTM(384, 768, bidirectional=True, batch_first=True, dropout=0.2)
        self.bi_lstm21 = nn.LSTM(768 * 2, 512, bidirectional=True, batch_first=True)
        self.bi_lstm22 = nn.LSTM(384, 512, bidirectional=True, batch_first=True)
        self.bi_lstm31 = nn.LSTM(2048, 384, bidirectional=True, batch_first=True)
        self.bi_lstm32 = nn.LSTM(1024, 384, bidirectional=True, batch_first=True)

        self.pool = GlobalMaxPooling1D()
        self.dense = nn.Sequential(
            nn.BatchNorm1d(6144),
            nn.Linear(in_features=6144, out_features=128),
            nn.SELU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid()
        )
        self.bce = nn.BCELoss(reduce=True, reduction='mean')

        self.train_auc = torchmetrics.AUROC()
        self.val_auc = torchmetrics.AUROC()

    def forward(self, x, x_s):
        x = torch.transpose(x, 1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x_0 = self.conv3(x)
        x_0 = torch.transpose(x_0, 1, 2)
        x_1, _ = self.bi_lstm1(x_0)
        x_21, _ = self.bi_lstm21(x_1)
        x_22, _ = self.bi_lstm22(x_0)
        x_2 = torch.cat([x_21, x_22], dim=2)

        x_31, _ = self.bi_lstm31(x_2)
        x_32, _ = self.bi_lstm32(x_21)
        x_3 = torch.cat([x_31, x_32], dim=2)

        # x_41, _ = self.bi_lstm41(x_3)
        # x_42, _ = self.bi_lstm42(x_31)
        # x_4 = torch.cat([x_41, x_42], dim=2)

        x_5 = torch.cat([x_1, x_2, x_3], dim=2)
        x_5 = self.pool(x_5)
        x_emb = self.emb(x_s)
        x_emb = x_emb.view(-1, 1024)
        x_6 = torch.cat([x_5, x_emb], dim=1)
        output = self.dense(x_6)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)
        scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.3, patience=2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "valid_auc_epoch"}

    def training_step(self, batch, batch_idx):
        x, x_l, y = batch[0], batch[1], batch[2]
        y_hat = self(x, x_l)
        loss = self.bce(y_hat, y)
        self.train_auc(y_hat, y.to(torch.int))
        self.log('train_auc', self.train_auc, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def training_epoch_end(self, outputs):
        self.train_auc.reset()

    def validation_step(self, batch, batch_idx):
        x, x_s, y = batch[0], batch[1], batch[2]
        y_hat = self(x, x_s)
        loss = self.bce(y_hat, y)
        self.val_auc(y_hat, y.to(torch.int))
        self.log('val_auc', self.val_auc.compute(), on_step=True, on_epoch=False)
        return loss

    def validation_epoch_end(self, outputs):
        self.log('valid_auc_epoch', self.val_auc.compute(), prog_bar=True)
        self.val_auc.reset()

    def predict_step(self, batch, batch_idx):
        x, x_s, y = batch[0], batch[1], batch[2]
        return self(x, x_s)

class TPSAprilNet_Attention(nn.Module):
    def __init__(self):
        super(TPSAprilNet_Attention, self).__init__()

        self.bi_lstm1 = nn.LSTM(13*4, 256, bidirectional=True, batch_first=True, dropout=0.2)
        self.bi_lstm21 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.bi_lstm22 = nn.LSTM(13*4, 256, bidirectional=True, batch_first=True)
        self.bi_lstm31 = nn.LSTM(1024, 256, bidirectional=True, batch_first=True)
        self.bi_lstm32 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.bi_lstm41 = nn.LSTM(1024, 256, bidirectional=True, batch_first=True)
        self.bi_lstm42 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.pool = GlobalMaxPooling1D()
        self.dense = nn.Sequential(
            nn.Linear(in_features=3584, out_features=128),
            nn.SELU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_1, _ = self.bi_lstm1(x[0])
        x_21, _ = self.bi_lstm21(x_1)
        x_22, _ = self.bi_lstm22(x[0])
        x_2 = torch.cat([x_21, x_22], dim=2)

        x_31, _ = self.bi_lstm31(x_2)
        x_32, _ = self.bi_lstm32(x_21)
        x_3 = torch.cat([x_31, x_32], dim=2)

        x_41, _ = self.bi_lstm41(x_3)
        x_42, _ = self.bi_lstm42(x_31)
        x_4 = torch.cat([x_41, x_42], dim=2)

        x_6 = torch.cat([x_1, x_2, x_3, x_4], dim=2)
        x_6 = self.pool(x_6)
        output = self.dense(x_6)
        return output

class FlattenBatchNorm1d(nn.BatchNorm1d):
    "BatchNorm1d that treats (N, C, L) as (N*C, L)"

    def forward(self, input):
        sz = input.size()
        return super().forward(input.view(-1, sz[-1])).view(*sz)

class BertLayer(_BertLayer):
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)[0]


class TPSAprilNet_Bert(nn.Module):
    def __init__(self):
        super(TPSAprilNet_Bert, self).__init__()
        szs = [13*4, 256, 512, 1024, 256, 32]
        mhas = [1]
        self.layers = nn.Sequential(*self.get_layers(mhas, szs))

    def get_layers(self, mhas, szs):
        layers = []

        for layer_i, (in_sz, out_sz) in enumerate(zip(szs[:-1], szs[1:])):
            layers.append(nn.Linear(in_sz, out_sz))
            layers.append(FlattenBatchNorm1d(out_sz))
            layers.append(nn.SiLU(inplace=True))
            layers.append(nn.Dropout(p=0.2, inplace=True))

            if layer_i in mhas:
                layers.append(BertLayer(BertConfig(
                    num_attention_heads=64,
                    hidden_size=out_sz,
                    intermediate_size=out_sz)))

        layers.append(GlobalAvgPooling1D())
        layers.append(nn.Linear(szs[-1], 1))
        layers.append(nn.Sigmoid())
        return layers

    def forward(self, x):
        return self.layers(x[0])


class TPSAprilDataset(Dataset):
    def __init__(self, df, is_test=False):
        if is_test:
            self.indices = df['sequence'] - 25968
        else:
            self.indices = df['sequence']
        self.targets = df['state']
        self.is_test = is_test
        self.df = df

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if not self.is_test:
            X = F_train[self.indices[idx]]
            X_s = self.df.loc[idx].subject
        else:
            X = F_test[self.indices[idx]]
            X_s = self.df.loc[idx].subject
        y = self.targets[idx]
        return torch.FloatTensor(X), torch.IntTensor([X_s]), torch.FloatTensor([y])

class TPSAprilDataLoader(pl.LightningDataModule):
    def __init__(self, df, batch_size=256, fold=None):
        super().__init__()
        self.batch_size = batch_size
        self.df = df
        self.fold = fold

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        dataset = TPSAprilDataset(self.df)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=8, shuffle=True, drop_last=False)
        return train_loader

    def valid_dataloader(self):
        dataset = TPSAprilDataset(self.df)
        valid_loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=8, shuffle=False, drop_last=False)
        return valid_loader

    def test_dataloader(self):
        dataset = TPSAprilDataset(self.df, is_test=True)
        test_loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=8, shuffle=False, drop_last=False)
        return test_loader


def read_data(data):
    return tuple(d.cuda() for d in data[:-1]), data[-1].cuda()


N_FOLDS = 4

gkf = StratifiedGroupKFold(N_FOLDS, shuffle=True)
y_oof = np.zeros(index_df_train.shape[0])
y_test = np.zeros(ss.shape[0])

ix = 0
for train_ind, val_ind in gkf.split(index_df_train, index_df_train["state"], groups=index_df_train["subject"]):
    print(f"******* Fold {ix} ******* ")
    train_df, val_df = index_df_train.iloc[train_ind].reset_index(drop=True), index_df_train.iloc[val_ind].reset_index(drop=True)

    train_loader = TPSAprilDataLoader(train_df).train_dataloader()
    val_loader = TPSAprilDataLoader(val_df).valid_dataloader()
    test_loader = TPSAprilDataLoader(ss).test_dataloader()

    model = TPSAprilNet_2()

    early_stop_callback = EarlyStopping(monitor='valid_auc_epoch', min_delta=0.00, patience=4, verbose=True, mode='max')
    rich_progress_bar_callback = RichProgressBar()
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(limit_train_batches=0.5, callbacks=[early_stop_callback], max_epochs=50, gpus=1, accumulate_grad_batches=2)
    trainer.fit(model, train_loader, val_loader)
    torch.save(model.state_dict(), 'models/encoder.pkl')
    val_pred_list = trainer.predict(model, val_loader)
    val_pred = torch.cat(val_pred_list, dim=0).detach().cpu().numpy().ravel()
    test_pred_list = trainer.predict(model, test_loader)
    test_pred = torch.cat(test_pred_list, dim=0).detach().cpu().numpy().ravel()
    y_oof[val_ind] = val_pred
    y_test += test_pred / N_FOLDS
    ix = ix + 1

cv_auc = np.round(roc_auc_score(index_df_train['state'].values, y_oof), 4)
print("CV Val AUC:", cv_auc)
ss['state'] = y_test
ss.to_csv(f"submissions/submission_{trial}_{cv_auc}.csv", sep=",", index=False)


