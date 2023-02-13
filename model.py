import numpy as np
import pandas as pd
import math
from tensorflow.keras.preprocessing.sequence import pad_sequences

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# hyperparameters
batch_size = 32 
window_size = 30
learning_rate = 0.0001
epochs = 70
embed_dim = 60
n_head = 1
n_layer = 1
dropout = 0.2
# ------------

torch.manual_seed(1337)

## Load data
df = pd.read_csv("data/diginetica_preprocessed.csv")
inp = df.groupby(['session_id'])['item_id'].apply(list)
inpc = df.groupby(['session_id'])['category_id'].apply(list)
inpf = df.groupby(['session_id'])['price'].apply(list)

vocab = df['item_id'].nunique()+5
vocab_cat = df['category_id'].nunique()+5


def seq_to_window(arr, window_size):
    """Converts variable length item sequence to fixed size and splits into train and test.

    Args:
        arr (list): variable length item sequence
        window_size (int): max length of sequence 'T'

    Returns:
        seq (list): Fixed length item sequence
        pos (list): Fixed length next item sequence
        test (list): Last item in the sequence
    """
    seq, pos, test = [], [], []
    for i, row in enumerate(arr):
        if (len(row) <= window_size) and (len(row) > 2):
            seq.append(row[:-2])
            pos.append(row[1:-1])
            test.append(row[-1])
    return seq, pos, test


seq, pos, test = seq_to_window(inp.values.tolist(), window_size=window_size)
seq = pad_sequences(seq, value=0, maxlen=window_size-1, dtype='int32')
pos = pad_sequences(pos, value=0, maxlen=window_size-1, dtype='int32')

def random_neq(seq, vocab):
    """Choose a random element that is not in the sequence."""
    t = np.random.randint(1, vocab)
    while t in seq:
        t = np.random.randint(1, vocab)
    return t

neg=[]
for row in seq:
    lst=[]
    for elt in row:
        if elt == 0: lst.append(0)
        else: lst.append(random_neq(row, vocab=vocab))
    neg.append(lst)
neg = np.array(neg)


def seq_to_window(arr, window_size):
    """Converts variable length category sequence to fixed size.  """
    seq = []
    for i, row in enumerate(arr):
        if (len(row) <= window_size) and (len(row) > 2):
            seq.append(row[:-2])
    return seq


seqc = seq_to_window(inpc.values.tolist(), window_size=window_size)
seqc = pad_sequences(seqc, value=0, maxlen=window_size-1, dtype='int32')
seqn = seq_to_window(inpf.values.tolist(), window_size=window_size)
seqn = pad_sequences(seqn, value=0, maxlen=window_size-1, dtype='float32')
seqn = np.expand_dims(seqn, axis=2)


## --------------------------------Modelling------------------------------------
class TransformerModel(nn.Module):
    def __init__(self, vocab, embed_dim, maxlen, n_blocks=10, dropout=0.2):
        super(TransformerModel, self).__init__()
        
        self.encoder = nn.Embedding(vocab, embed_dim, padding_idx=0)
        self.cat_encoder = nn.Embedding(vocab_cat, embed_dim, padding_idx=0)
        self.weights = nn.Parameter(torch.rand(embed_dim, (2*embed_dim)+1), requires_grad=True)
        self.embed_dim = embed_dim
        
        self.decoder = nn.Linear(2*embed_dim, embed_dim)
        self.pos_emb = torch.nn.Embedding(maxlen, embed_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_head)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layer)
                   
        self.last_layernorm = nn.LayerNorm(embed_dim, eps=1e-8)
    
    def gating_network(self, *args): 
        """Gating network to combine the different embeddings."""      
        weights_normalized = nn.functional.softmax(self.weights, dim=-1)
        out = torch.matmul(torch.cat((args), dim=-1), weights_normalized.t())
        return out
    
    def log2feats(self, x, xc, xn):        
        seq = self.encoder(x) * math.sqrt(self.embed_dim)
        seqc = self.cat_encoder(xc) * math.sqrt(self.embed_dim)
        seq = self.gating_network(seq, seqc, xn)
        
        # Add positional encoding
        positions = np.tile(np.array(range(seq.shape[1])), [seq.shape[0], 1])
        seq += self.pos_emb(torch.LongTensor(positions))
        
        # Apply causal attention mask
        timeline_mask = torch.BoolTensor(x == 0)
        seq *= ~timeline_mask.unsqueeze(-1) #NxTx1
        
        tl = seq.shape[1] # timeline length
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool))
        seq.transpose_(0, 1)
        seq = self.transformer_encoder(seq, mask=attention_mask)#, src_key_padding_mask=timeline_mask) 
        seq.transpose_(0, 1)
        
        # modelling short sequences
        last_col = xc[:,[-1]]
        last_col_mask = (xc == last_col)
        seq_short = seq * ~last_col_mask.unsqueeze(-1) 
        mask = ~torch.tril(torch.ones((seq.size(1), seq.size(1)), dtype=torch.bool))
        seq_short.transpose_(0, 1)
        seq_short = self.transformer_encoder(seq_short, mask=mask) #,
        seq_short.transpose_(0, 1)
        seq = self.decoder(torch.cat((seq, seq_short), dim=-1))
        return seq

    def forward(self, x, y_pos, y_neg, xc, xn):
        """Forward pass of the model. 

        Args:
            x (tensor): sequence of items
            y_pos (tensor): positive sequence
            y_neg (tensor): negative sequence
            xc (tensor): item category sequence
            xn (tensor): item price sequence

        Returns:
            pos_logits (tensor): logits for the positive sequence
            neg_logits (tensor): logits for the negative sequence 
        """
        log_feats = self.log2feats(x, xc, xn)        
        pos_embs = self.encoder(y_pos)
        neg_embs = self.encoder(y_neg)
        
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, x, item_indices, xc, xn):
        """Given a sequence of items, item category and price, predict a score for candidate items.

        Args:
            x (tensor): sequence of items
            item_indices (tensor): candidate items
            xc (tensor): sequence of item categories
            xn (tensor): sequence of item prices

        Returns:
            logits (tensor): score for candidate items
        """
        log_feats = self.log2feats(x.unsqueeze(0), xc, xn)
        final_feat = log_feats[:, -1, :]
        item_embs = self.encoder(item_indices)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits



model = TransformerModel(vocab=vocab, embed_dim=embed_dim, 
                             maxlen=window_size-1, n_blocks=2, dropout=dropout)

bce_criterion = torch.nn.BCEWithLogitsLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98))

# Pytorch train data loader
class SeqDataset(Dataset):
    def __init__(self, x_seq, y_pos, y_neg, x_seqc, x_seqn):
        self.len = x_seq.shape[0]
        self.x_seq = x_seq
        self.y_pos = y_pos
        self.y_neg = y_neg
        self.x_seqc = x_seqc
        self.x_seqn = x_seqn
        
    def __getitem__(self, i):
        return self.x_seq[i], self.y_pos[i], self.y_neg[i], self.x_seqc[i], self.x_seqn[i]
    
    def __len__(self):
        return self.len

mask2d = seq==0    
dataset = SeqDataset(seq, pos, neg, seqc, seqn)
train_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle=True)


model.train()
total_loss=0
for epoch in range(epochs):
    for seq_, pos_, neg_, seqc_, seqn_ in train_loader:        
             
        pos_logits, neg_logits = model(seq_, pos_, neg_, seqc_, seqn_)        
        pos_labels, neg_labels = torch.ones(pos_logits.shape), torch.zeros(neg_logits.shape)
        
        indices = torch.where(pos_ != 0)
        optimizer.zero_grad()
        
        loss = bce_criterion(pos_logits[indices], pos_labels[indices])
        loss += bce_criterion(neg_logits[indices], neg_labels[indices])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(epoch, loss)


dataset = SeqDataset(seq, pos, test, seqc, seqn)
test_loader = DataLoader(dataset = dataset, batch_size = 1, shuffle=True)

model.eval()
NDCGL, HTL ,MRRL = [],[],[]
for _ in range(10):
    MRR, HT = 0.0, 0.0
    valid_user = 0.0
    z=0
    
    for seq_, pos_, test_, seqc_, seqn_ in test_loader:

        item_idx = test_.tolist()
        for _ in range(100):        
            t = np.random.randint(1, vocab)
            while t in seq_: t = np.random.randint(1, vocab)
            item_idx.append(t)
            
            
        seq_ = seq_.tolist()[0]
        predictions = -model.predict(torch.tensor(seq_), torch.tensor(item_idx), seqc_, seqn_)
        predictions = predictions[0]  
        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 20:
            if rank !=0:
                MRR += 1 / rank
            HT += 1
        if z==100:
            break
        z+=1
    
    HTL.append(HT / valid_user)
    MRRL.append(MRR / valid_user)
    print(f'{MRR / valid_user}, {HT / valid_user}')
print('--------')
print(np.array(MRRL).mean(), np.array(HTL).mean() )
