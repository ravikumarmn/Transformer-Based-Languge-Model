import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 32 # sequence to process parallel
sequence_size = 8 # maximum context length
max_iters = 3000 
eval_interval = 300
learning_rate = 1e-2
device  = "cuda" if torch.cuda.is_available() else "cpu"
eval_iter = 200
#-----------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/ravikumarmn/Transformer-Based-Languge-Model/main/data/shakespeare/input.txt
with open("data/shakespeare/input.txt","r",encoding = "utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# mapping from charcters to integers
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s : [stoi[ch] for ch in s] # takes a string, outputs the list of integers
decode = lambda s : "".join([itos[ch] for ch in s]) # takes list of integers, return the string.

data = torch.tensor(encode(text),dtype = torch.long)
# split the data into training set and validation set.
n = int(0.9*len(data)) # first 90% will be train and rest will be validation.
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split=="train" else val_data
    ix = torch.randint(len(data)-sequence_size,(batch_size,))
    x = torch.stack([train_data[i:i+sequence_size] for i in ix])
    y = torch.stack([train_data[i+1:i+sequence_size+1] for i in ix])
    x,y  = x.to(device),y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = dict()
    for split in ["train","val"]:
        losses = torch.zeros(eval_iter)
        for k in range(eval_iter):
            X,y = get_batch(split)
            logits,loss = model(X,y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out


# simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super(BigramLanguageModel,self).__init__()
        """each token will have a vocab_sized values. 
        when the input idx is given, it goes to its location and takes out its raw
        """
        self.token_embedding_table  = nn.Embedding(vocab_size,vocab_size)
    
    def forward(self,token,targets= None):
        # token (N,C),targets (N,C)
        logits = self.token_embedding_table(token) # (N,C,d1,d2,dk)
        if targets is None:
            loss= None
        else:
            N,C,E = logits.shape
            logit = logits.view(N*C,-1)
            targets = targets.view(-1)
            loss = F.cross_entropy(logit,targets)
        return logits,loss

    def generate(self,token,max_new_tokens = 200):
        for _ in range(max_new_tokens):
            logits,loss = self(token)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim = -1)
            next_token = torch.multinomial(probs,num_samples=1)
            token = torch.cat((token,next_token),dim = 1)
        return token


model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(),lr = learning_rate)

for iter in range(max_iters):
    if iter % eval_interval ==0:
        losses = estimate_loss(m)
        print(f"step {iter}: train_loss {losses['train']:.4f},val loss {losses['val']:.4f}")
    xb,yb = get_batch("train")
    logits,loss = m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# generate text
context = torch.zeros((1,1),dtype = torch.long,device = device)
file1 = open("bigram.txt","w")
pred_text = decode(m.generate(context,max_new_tokens=10000)[0].tolist())
file1.write(pred_text)






