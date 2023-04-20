import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 64 # sequence to process parallel
sequence_size = 25#256 # maximum context length
max_iters = 3000 
eval_interval = 300
learning_rate = 3e-4
device  = "cuda" if torch.cuda.is_available() else "cpu"
eval_iter = 200
n_embd = 384
head_size = n_embd
num_heads = 6
n_layers = 6
dropout= 0.2
experiment_name = "bigram_lang_model_lm_multi_head_attention_blocks_layernorm"
#-----------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/ravikumarmn/Transformer-Based-Languge-Model/main/data/shakespeare/input.txt
with open("data/shakespeare/input.txt","r",encoding = "utf-8") as f:
    text = f.read()

chars = sorted(list(set(x)))
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


class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embd,head_size,bias = False)
        self.query = nn.Linear(n_embd,head_size,bias = False)
        self.value = nn.Linear(n_embd,head_size,bias = False)
        self.register_buffer("tril",torch.tril(torch.ones(sequence_size,sequence_size)))

        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x):
        B,S,C = x.shape
        k = self.key(x) # B,S,C
        q = self.query(x) # B,S,C
        v = self.value(x) # B,S,C
        # compute attention scores
        wei = q @ k.transpose(-2,-1)*C**-0.5 # B,S,C @ B,C,S --> B,S,S # scale : scaled attention
        wei = wei.masked_fill(self.tril == 0,float('-inf')) # B,S,S
        wei = F.softmax(wei,dim = -1) # B,S,S
        wei = self.dropout(wei)
        out = wei  @ v # B,S,S @ B,S,C --> B,S,C
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads,head_size): # head_size = n_embd
        super(MultiHeadAttention,self).__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd,n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads],dim = -1)
        out = self.dropout(self.proj(out))
        return out



class FeedForward(nn.Module):
    def __init__(self,n_embd):
        super(FeedForward,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(
                n_embd,4 * n_embd
            ),
            nn.ReLU(),
            nn.Linear(4 * n_embd,n_embd),
            nn.Dropout(dropout)
            
        )

    def forward(self,x):
        return self.net(x)
    


class Block(nn.Module):
    def __init__(self,n_embd,n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(n_heads,head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    


# simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super(BigramLanguageModel,self).__init__()
        """each token will have a vocab_sized values. 
        when the input idx is given, it goes to its location and takes out its raw
        """
        self.token_embeddings  = nn.Embedding(vocab_size,n_embd) 
        self.position_embedding_table = nn.Embedding(sequence_size,n_embd) #8,32
        self.blocks = nn.Sequential(*[Block(n_embd,num_heads) for _ in range(n_layers)])
        self.lm_head = nn.Linear(n_embd,vocab_size) # language_model_head layer
        # self.sa_head = MultiHeadAttention(num_heads,n_embd//4) # 4 heads of 8-dimentional self-attention
        # self.ffwd = FeedForward(n_embd)


    def forward(self,token,targets= None):
        B,S = token.shape
        token_embeds = self.token_embeddings(token) # B,S,C
        pos_emb = self.position_embedding_table(torch.arange(S,device = device)) # (S,C)
        x  = token_embeds + pos_emb # (B,S,C)
        x = self.blocks(x)
        # x = self.sa_head(x) # B,S,C
        # x = self.ffwd(x)
        logits = self.lm_head(x)
    
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
            cropped_token = token[:,-sequence_size:]
            logits,_ = self(cropped_token)
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
    xb,yb = get_batch ("train")
    logits,loss = m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,sequence_size),dtype = torch.long,device = device)
file1 = open(f"output/{experiment_name}.txt","w")
pred_text = decode(m.generate(context,max_new_tokens=10000)[0].tolist())
print(pred_text[:1000])
file1.write(pred_text)






