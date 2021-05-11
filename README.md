# pointer-transformer-model-pytorch
A transformer with copy mechanism to handle out-of-vocabulary words

This is a transformer model built with PyTorch. Yes, you noticed that the entire format is extremely close to this [Tutorial](https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb). We modified only the necessary parts to add the pointer mechanism.

To use this code, we sincerely recommend you to take a look at this [Tutorial](https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb). You can just import the model and replace the one in the tutorial.

As everyone may have their own application with this model, we would not publish a separate file for the usage, but rather paste it here. This usage is also very similar to the format you saw in the tutorial just now. One **BIG** difference you need to notice is that the original ```Field``` class instances are not enough to handle out-of-vocabulary words, and thus all that you need to do is to put the ```oov.py``` file from [here](https://cestwc.medium.com/how-do-you-write-a-clean-pointer-generator-model-with-pytorch-80d25bde113b) into the directory where your codes are running. You can assume this ```oov.py``` as a patch to standard Torchtext 0.9.1, till the day newer versions emerge.

## Usage
### Import all necessary packages, fix your seeds, and create your ```Field``` objects
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np

import random
import math
import time
```
```python
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
```
```python
spacy_en = spacy.load('en_core_web_sm')
max_input_length = 100
def tokenize_en(text):
	"""
	Tokenizes English text from a string into a list of strings
	"""
	return [tok.text for tok in spacy_en.tokenizer(text)][:max_input_length-2]
```
(A big chunk of codes you can see [here](https://cestwc.medium.com/how-do-you-write-a-clean-pointer-generator-model-with-pytorch-80d25bde113b))
```python
import torchtext.vocab as vocab

from oov import BucketIteratorOOV
from customized import ENGLISHTEXT

SRC = ENGLISHTEXT(tokenize = tokenize_en, build_vocab = True,
                  batch_first = True)

TRG = ENGLISHTEXT(tokenize = tokenize_en, batch_first = True)

TRG.vocab = SRC.vocab

from torchtext.legacy import data, datasets

fields = {'your_source_field': ('src', SRC), 'your_target_field': ('trg', TRG)}

train_data, valid_data, test_data = data.TabularDataset.splits(
                            path = 'your_path',
                            train = your_train_file,
                            validation = your_valid_file,
                            test = your_test_file,
                            format = 'json',
                            fields = fields
)

BATCH_SIZE = 16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = BucketIteratorOOV.splits(
    (train_data, valid_data, test_data), 
     batch_size = BATCH_SIZE,
     sort_within_batch = True,
     sort_key = lambda x : len(x.src),
     shuffle = True,
     device = device)
```

### import the model module
```python
from model import Encoder, Decoder, Seq2Seq

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc = Encoder(INPUT_DIM, 
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT, 
              device)

dec = Decoder(OUTPUT_DIM, 
              HID_DIM, 
              DEC_LAYERS, 
              DEC_HEADS, 
              DEC_PF_DIM, 
              DEC_DROPOUT, 
              device)
			  
SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
```

```python
def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

def initialize_weights(m):
	if hasattr(m, 'weight') and m.weight.dim() > 1:
		nn.init.xavier_uniform_(m.weight.data)
		
model.apply(initialize_weights)

LEARNING_RATE = 0.0005

optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

criterion = torch.nn.NLLLoss(ignore_index = TRG_PAD_IDX) # use NLLLoss here as we have applied softmax and log
```

### model training and evaluating
```python
from tqdm.notebook import tqdm

def train(model, iterator, optimizer, criterion, clip):
    
	model.train()

	epoch_loss = 0

	for i, batch in enumerate(tqdm(iterator)):

		src = batch.src
		trg = batch.trg

		optimizer.zero_grad()

		output, _ = model(src, trg[:,:-1])

		#output = [batch size, trg len - 1, output dim]
		#trg = [batch size, trg len]

		output_dim = output.shape[-1]

		output = output.contiguous().view(-1, output_dim)
		trg = trg[:,1:].contiguous().view(-1)

		#output = [batch size * trg len - 1, output dim]
		#trg = [batch size * trg len - 1]

		loss = criterion(output, trg)

		loss.backward()

		torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

		optimizer.step()

		epoch_loss += loss.item()

	return epoch_loss / len(iterator)
	
def evaluate(model, iterator, criterion):
    
	model.eval()

	epoch_loss = 0

	with torch.no_grad():

		for i, batch in enumerate(tqdm(iterator)):

			src = batch.src
			trg = batch.trg

			output, _ = model(src, trg[:,:-1])

			#output = [batch size, trg len - 1, output dim]
			#trg = [batch size, trg len]

			output_dim = output.shape[-1]

			output = output.contiguous().view(-1, output_dim)
			trg = trg[:,1:].contiguous().view(-1)

			#output = [batch size * trg len - 1, output dim]
			#trg = [batch size * trg len - 1]

			loss = criterion(output, trg)

			epoch_loss += loss.item()

    return epoch_loss / len(iterator)
	
def epoch_time(start_time, end_time):
	elapsed_time = end_time - start_time
	elapsed_mins = int(elapsed_time / 60)
	elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
	return elapsed_mins, elapsed_secs
	
N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), drivePath + 'tut6-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
```              
