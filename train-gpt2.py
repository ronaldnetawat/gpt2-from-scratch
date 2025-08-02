from dataclasses import dataclass
import torch
from torch.cpu import is_available
import torch.nn as nn
from torch.nn import functional as F
import math
import os
from torch.distributed import init_process_group, destroy_process_group

@dataclass
class GPTConfig:
    block_size: int = 1024 # context_size
    vocab_size: int = 50257 # vocab_size
    n_layer: int = 12 # number of layers/blocks
    n_head: int = 12 # number of attention heads per block
    n_embd: int = 768 # number of embedding dimensions
    # bias: bool = True # true: Linear layers and LayerNorms, False: better and faster
    # dropout: float = 0.0
    # check


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 # check n_embd divides evenly among the heads
        # project input embeddings into q, k, v in one shot
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # project back to n_embd 
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.RESIDUAL_INIT = 1
        # regularization
        # self.dropout = config.dropout
        self.n_head= config.n_head
        self.n_embd = config.n_embd
        # masking
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # (batch_size, block_size, n_embd)

        # produce q, k, and v
        qkv = self.c_attn(x) # one large matrix
        q, k, v  = qkv.split(self.n_embd, dim=2) # split them into 3 across the last dimension
        # reshape and transpose for heads, for parallel processing in heads
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, nh, hs) --> (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, nh, hs) --> (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, nh, hs) --> (B, nh, T, hs)

        # attention calculation, with scaled attention
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, hs) @ (B, nh, hs, T) --> (B, nh, T, T)
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # causal masking
        # att = F.softmax(att, dim=-1) # softmax the outputs
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) --> (B, nh, T, hs), calculate value vectors

        # FlashAttention using torch.nn.functional.scaled_dot_product_attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # this optimizes for kernel fusion

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble and concatenate heads
        # output projection to mix information across heads after attention application
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    # the MLP block
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd) # expand embedding dim
        self.gelu = nn.GELU(approximate="tanh") # GELU nonlinearity, as per GPT2
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd) # project back down to n_embd
        self.c_proj.RESIDUAL_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    # the attention block
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd), # pos. embeddings
            # dropout = nn.Dropout(config.dropout), # dropout
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # attention blocks
            ln_f = nn.LayerNorm(config.n_embd), # final layernorm
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # final classifier

        # sharing the same weight for wte and lm_head (according to papers: Vaswani, and other)
        self.transformer.wte.weight = self.lm_head.weight

        # initialize the parameters
        self.apply(self._init_weights)

    
    # initializing weights according to GPT2
    def _init_weights(self, module):
        # for linear modules
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'RESIDUAL_INIT'):
                std *= (2*self.config.n_layer) **-0.5 # scale down by 1/sqrt(n)
            torch.nn.init.normal_(module.weight, mean=0.0,std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) # set bias weights to 0
        # for embedding modules, use the same params as linear
        elif isinstance(module, nn.Embedding):
           torch.nn.init.normal_(module.weight, mean=0.0,std=0.02) 

    
    def forward(self, idx, targets=None):
        # device = idx.device
        B, T = idx.size()
        # assert max context_size or block_size
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        # forward the embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, C)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    @classmethod
    def from_pretrained(cls, model_type):
        # load pretrainedd weights from HF for GPT2
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        # print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints

        # our model and keys
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)


        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    

# ================================================================

import tiktoken

# simple DataLoader class for getting batches
class DataLoaderSimple:
    def __init__(self, B, T): 
        self.B = B # batch dim
        self.T = T # time dim

        enc = tiktoken.get_encoding('gpt2')
        with open('input.txt', 'r') as f:
            text = f.read() # read the entire file
        tokens = enc.encode(text) # encode using encoder
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")  # print total # of tokens in dataset
        print(f"1 epoch has {len(self.tokens) // (B*T)} batches")   # print how many batches each epoch will have before returning to the beginning of dataset

        # start at first token
        self.current_position = 0

    # method to get next batch
    def next_batch(self):
        B,T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1] # +1 for targets
        x = buf[:-1].view(B, T) # input tensor
        y = buf[1:].view(B, T) # target tensor
        # next position in tensor, goes up by B*T
        self.current_position += B*T
        # if at the end of data
        if self.current_position+(B*T)+1 > len(self.tokens):
            self.current_position = 0
        return x, y
    
# Model definition complete
# ==========================================================
# training and inference code

# set up DDP
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    assert torch.cuda.is_available(), "we prolly need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # if not ddp, we are running on a single gpu, and one process
    ddp_rank = 0
    ddp_local_rank = 0
    master_process = True
    ddp_world_size = 1
    # autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

import time

# device config
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")


torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


# get training data
train_loader = DataLoaderSimple(B=16, T=1024) # (16, 1024) batches

# enable tf32 for faster training
torch.set_float32_matmul_precision('high')


# get logits for our model
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)
# logits, loss = model(x, y)

# consts for lr_scheduling (acc. to GPT3)
max_lr = 6e-4
min_lr = max_lr*0.1
warmup_steps = 10
max_steps = 50

# function for lr_scheduler with cosine decay and linear warm-up (copied from karpathy)
def get_lr(it):
    # 1) linear warmup for warmup_steps steps
    if it < warmup_steps:
        return max_lr*(it+1)/(warmup_steps)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (max_lr - min_lr)

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9,0.95), eps=1e-8) # good LR for initial debugging stage
for i in range(max_steps):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x = x.to(device)
    y = y.to(device)
    optimizer.zero_grad()
    # using blackwell architecture for now. this is for ampere usually.
    # only apply this to model output and loss in forward pass, not to .backward() and .step()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
        # import code; code.interact(local=locals())
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0) # clip the norm at 1
    # lr_scheduling
    lr = get_lr(i)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    torch.cuda.synchronize() # wait for GPU to finish processes
    t1 = time.time()
    dt = (t1 - t0)*1000 # in ms
    tokens_per_sec = (train_loader.B * train_loader.T) // (t1 - t0)
    print(f"step: {i}, loss: {loss.item()}, lr: {lr:.4f},  norm: {norm:.4f}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec}")

import sys; sys.exit(0)

# prefix tokens for inference
# tokenize using tiktoken
model.eval()
num_return_sequences = 5
max_length = 20
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # repeat 5 times
x = tokens.to(device)


# function to generate text
torch.manual_seed(42)
# torch.cuda.manual_seed(42)

while x.size(1) < max_length:
    # won't compute gradients
    with torch.no_grad():
        logits = model(x)
        logits = logits[:,-1,:] # only keep the logits (last col of T)
        probs = F.softmax(logits, dim=-1) # get the probabilities
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # only keep the top 50 most likely tokens
        ix = torch.multinomial(topk_probs, 1) # (B, 1) sample one for each batch
        xcol = torch.gather(topk_indices, -1, ix) # get the indices
        x = torch.cat((x, xcol), dim=1) # append to the last column

# finally, print the generated code
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)