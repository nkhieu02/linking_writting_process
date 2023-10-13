import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.data import EssayDataset, generate_batch, activities,\
                       df, label_df, c_feat, d_feat
from utils.utils import train, evaluate, EarlyStopping, CosineAnnealingWarmupRestarts
from utils.models import Grader

import argparse

parser = argparse.ArgumentParser()

## Model:
parser.add_argument("--HIDDEN_DIM", type=int, default=200)
parser.add_argument("--LINEAR_HIDDEN_DIM", type=int, default=10)
parser.add_argument("--EMBEDDING_SIZE", type=int, default=10)

## Data:
parser.add_argument("--SPLIT", type=float, default=0.2)
parser.add_argument("--NOF_DATA", type=int, default=100)

## Training:
parser.add_argument("--EPOCHS", type=int, default=3)
parser.add_argument("--CLIP", type=float, default=1)
parser.add_argument("--BATCH", type=int, default=10)
parser.add_argument("--LR", type=float, default=0.001)
parser.add_argument("--DELTA", type=float, default=0.001)
parser.add_argument("--PATIENCE", type=int, default=3)
parser.add_argument("--MIN_LR", type=float, default=0.001)
parser.add_argument("--MAX_LR", type=float, default=0.1)

args = parser.parse_args()


# Now you can access the values using args.<variable_name>
HIDDEN_DIM = args.HIDDEN_DIM
LINEAR_HIDDEN_DIM = args.LINEAR_HIDDEN_DIM
EMBEDDING_SIZE = args.EMBEDDING_SIZE

SPLIT = args.SPLIT
NOF_DATA = args.NOF_DATA

EPOCHS = args.EPOCHS
CLIP = args.CLIP
BATCH = args.BATCH
LR = args.LR
DELTA = args.DELTA
PATIENCE = args.PATIENCE
MIN_LR = args.MIN_LR
MAX_LR = args.MAX_LR

# Your script logic goes here

# Create a file name based on the attributes
## Create the file name to save:
DIR = f"rnn_H{HIDDEN_DIM}_LH{LINEAR_HIDDEN_DIM}_E{EMBEDDING_SIZE}_" \
      f"S{SPLIT}_N{NOF_DATA}_E{EPOCHS}_C{CLIP}_B{BATCH}_" \
      f"LR{LR}_D{DELTA}_P{PATIENCE}_MinLR{MIN_LR}_MaxLR{MAX_LR}.pt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the data
ids = df['id'].unique()[0:NOF_DATA]
from sklearn.model_selection import train_test_split
train_ids, val_ids = train_test_split(ids, test_size = SPLIT)
train_dataset = EssayDataset(df, train_ids, label_df, c_feat, d_feat)
val_dataset = EssayDataset(df, val_ids, label_df, c_feat, d_feat)
train_dataloader = DataLoader(train_dataset, batch_size = BATCH,
                        collate_fn = generate_batch)
val_dataloader = DataLoader(val_dataset, batch_size = BATCH,
                        collate_fn = generate_batch)

# Define model for training
class_size_dict = {'activity': len(activities)}
model = Grader(HIDDEN_DIM, LINEAR_HIDDEN_DIM, class_size_dict,
               EMBEDDING_SIZE, c_feat, d_feat).to(device)
training_steps = EPOCHS * len(train_dataloader)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LR)
early_stop = EarlyStopping(patience= PATIENCE, delta= DELTA)
lr_scheduler = CosineAnnealingWarmupRestarts(optimizer, training_steps, min_lr = MIN_LR, 
                                             max_lr= MAX_LR, warmup_steps= int(training_steps * 0.1))

for epoch in range(EPOCHS):


    train_loss = train(model, train_dataloader, val_dataloader,optimizer, criterion, CLIP,
                       evaluate, early_stop, )
    valid_loss = evaluate(model, val_dataloader, criterion)

    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f}') # 0.723

