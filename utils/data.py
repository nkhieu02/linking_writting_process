import os
import pandas as pd
import torch

base_dir = "linking-writing-processes-to-writing-quality"
train_dir = "train_logs.csv"
label_dir = "train_scores.csv"
test_dit = "test_logs.csv"

df = pd.read_csv(os.path.join(base_dir, train_dir), delimiter = ',')
label_df = pd.read_csv(os.path.join(base_dir, label_dir), delimiter = ',')

# Text change to number of text changed:
df.loc[df['text_change'] == 'NoChange', 'text_change'] = ''
df['text_change'] = df['text_change'].str.len()

# Group all 'Move From' as one class:
df.loc[df['activity'].str.contains('Move From'), 'activity'] = 'Move'

# Try different version of data

## Number of feature
features = ['id', 'event_id', 'action_time',\
                         'activity', 'text_change', 'word_count']
d_feat = ['activity']
c_feat = ['action_time', 'text_change', 'word_count']
df = df[features]
activities = list(df['activity'].unique())
activities_map = {k:v for v,k in enumerate(activities)}
df['activity'] = df['activity'].\
                                   replace(activities_map)

from torch.utils.data import Dataset
from torch.nn.utils import rnn

class EssayDataset(Dataset):
    def __init__(self, data, ids, labels, c_feat, d_feat):
        self.data = data
        self.ids = ids
        self.labels = labels
        self.d_feat = d_feat
        self.c_feat = c_feat
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, i):
        id = self.ids[i]
        d_data = torch.tensor(self.data[self.data['id'] == id].reset_index(drop = True)\
            [self.d_feat].to_numpy())
        c_data = torch.tensor(self.data[self.data['id'] == id].reset_index(drop = True)\
            [self.c_feat].to_numpy())
        label = self.labels.loc[label_df['id'] == id, \
                               'score'].item()
        seq_length = len(c_data)
        return c_data, d_data, label, seq_length

def generate_batch(data_batch):
    d_datas = rnn.pad_sequence([x for _,x, _ , _ in data_batch], batch_first= True)
    c_datas = rnn.pad_sequence([x for x,_, _ , _ in data_batch], batch_first= True)
    labels = torch.tensor([label for _,_, label, _ in data_batch])
    seq_lengths = torch.tensor([seq_length for _,_, _, seq_length in data_batch])
    return c_datas, d_datas, labels, seq_lengths