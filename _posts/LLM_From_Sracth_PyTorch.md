---
layout: post
title: Large Language Model From Scratch
date: 2024-12-04
---
Review of Transformer Networks and Applications.

### Part I: Data Preparation and Preprocessing

In this section we cover the data preparation and sampling to get our input data ready for the LLM. You can download our sample data from here: https://en.wikisource.org/wiki/The_Verdict


```python
with open("sample_data/the-verdict.txt", encoding="utf-8") as f:
    raw_text = f.read()

print(f"Total number of characters: {len(raw_text)}")
print(raw_text[:20]) # print the first 20 charaters
```

    Total number of characters: 20479
    I HAD always thought
    

Next we tokenize and embed the input text for our LLM.
- First we develop a simple tokenizer based on some sample text that we then apply to the main input text above.


```python
import re
# Tokenize our input by splitting on whitespace and other characters
# Then we strip whitespace from each item and then filer out any empty strings
tokenized_raw_text = [item.strip() for item in re.split(r'([,.?_!"()\']|--|\s)', raw_text) if item.strip()]
print(len(tokenized_raw_text))
print(tokenized_raw_text[:20])
```

    4649
    ['I', 'HAD', 'always', 'thought', 'Jack', 'Gisburn', 'rather', 'a', 'cheap', 'genius', '--', 'though', 'a', 'good', 'fellow', 'enough', '--', 'so', 'it', 'was']
    

Next we convert the text tokens into token Ids that can be processed via embedding layers later. We can then build a vocabulary that consists of all the unique tokens.


```python
words = sorted(list(set(tokenized_raw_text)))
vocab_size = len(words)
print(f"Vocab size: {vocab_size}")
```

    Vocab size: 1159
    


```python
vocabulary = {token:integer for integer, token in enumerate(words)}

#Lets check the first 50 entries
for i, item in enumerate(vocabulary.items()):
    print(item)
    if i == 50:
        break
```

    ('!', 0)
    ('"', 1)
    ("'", 2)
    ('(', 3)
    (')', 4)
    (',', 5)
    ('--', 6)
    ('.', 7)
    (':', 8)
    (';', 9)
    ('?', 10)
    ('A', 11)
    ('Ah', 12)
    ('Among', 13)
    ('And', 14)
    ('Are', 15)
    ('Arrt', 16)
    ('As', 17)
    ('At', 18)
    ('Be', 19)
    ('Begin', 20)
    ('Burlington', 21)
    ('But', 22)
    ('By', 23)
    ('Carlo', 24)
    ('Carlo;', 25)
    ('Chicago', 26)
    ('Claude', 27)
    ('Come', 28)
    ('Croft', 29)
    ('Destroyed', 30)
    ('Devonshire', 31)
    ('Don', 32)
    ('Dubarry', 33)
    ('Emperors', 34)
    ('Florence', 35)
    ('For', 36)
    ('Gallery', 37)
    ('Gideon', 38)
    ('Gisburn', 39)
    ('Gisburns', 40)
    ('Grafton', 41)
    ('Greek', 42)
    ('Grindle', 43)
    ('Grindle:', 44)
    ('Grindles', 45)
    ('HAD', 46)
    ('Had', 47)
    ('Hang', 48)
    ('Has', 49)
    ('He', 50)
    

We can put these all together into our tokenizer class


```python
class TokenizerLayer:
    def __init__(self, vocabulary):
        self.token_to_int = vocabulary
        self.int_to_token = {integer:token for token, integer in vocabulary.items()}

    # The encode function turns text into token ids
    def encode(self, text):
        encoded_text = re.split(r'([,.?_!"()\']|--|\s)', text)
        encoded_text = [item.strip() for item in encoded_text if item.strip()]
        return [self.token_to_int[token] for token in encoded_text]

    # The decode function turns token ids back into text
    def decode(self, ids):
        text = " ".join([self.int_to_token[i] for i in ids])
        # Replace spaces before the specified punctuations
        return re.sub(r'\s+([,.?!"()\'])', r'\1', text)

# Initialize and test tokenizer layer
tokenizer = TokenizerLayer(vocabulary)
print(tokenizer.encode(""""It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""))
print(tokenizer.decode(tokenizer.encode("""It's the last he painted, you know," Mrs. Gisburn said with pardonable pride.""")))
```

    [1, 58, 2, 872, 1013, 615, 541, 763, 5, 1155, 608, 5, 1, 69, 7, 39, 873, 1136, 773, 812, 7]
    It' s the last he painted, you know," Mrs. Gisburn said with pardonable pride.
    

Next we special tokens for unknown words and to mark end of text.

SPecial tokens include:

[BOS] - Beginning of Sequence

[EOS] - End of Sequence. This markds the end of a text, usually used to concatenate multiple unrelated texts e.g. two different documents, wikipedia articles, books etc.

[PAD] - Padding: If we train an LLM with a batch size greater than 1, we may include multiple texts with different lenghts; with the padding token we pad the shorter texts to the longest length so that all texts have an equal lenght.

[UNK] - denotes words not included in the vocabulary
GPT2 only uses <|endoftext|> token for end of sequence and padding to reduce complexity which is analogous to [EOS].
Instead of <UNK> token for out-of-vocabulary words, GPT-2 uses byte-pair encoding (BPE) tokenizer, which breaks down words into subword unis.
For our application, we use <|endoftext|> tokens between two independent sources of text.



```python
tokenized_raw_text = [item.strip() for item in re.split(r'([,.?_!"()\']|--|\s)', raw_text) if item.strip()]
all_tokens = sorted(list(set(tokenized_raw_text)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocabulary = {token:integer for integer, token in enumerate(all_tokens)}
tokenizer = TokenizerLayer(vocabulary)
print(len(tokenized_raw_text))
print(tokenized_raw_text[:20])
```

    4649
    ['I', 'HAD', 'always', 'thought', 'Jack', 'Gisburn', 'rather', 'a', 'cheap', 'genius', '--', 'though', 'a', 'good', 'fellow', 'enough', '--', 'so', 'it', 'was']
    


```python
for i, item in enumerate(list(vocabulary.items())[-5:]):
    print(item)

# Get the new length of our vocabulary
print(len(vocabulary.items()))
```

    ('younger', 1156)
    ('your', 1157)
    ('yourself', 1158)
    ('<|endoftext|>', 1159)
    ('<|unk|>', 1160)
    1161
    


```python
class TokenizerLayer:
    def __init__(self, vocabulary):
        self.token_to_int = vocabulary
        self.int_to_token = {integer:token for token, integer in vocabulary.items()}

    # The encode function turns text into token ids
    def encode(self, text):
        encoded_text = re.split(r'([,.?_!"()\']|--|\s)', text)
        encoded_text = [item.strip() for item in encoded_text if item.strip()]
        encoded_text = [item if item in self.token_to_int else "<|unk|>" for item in encoded_text]
        return [self.token_to_int[token] for token in encoded_text]

    # The decode function turns token ids back into text
    def decode(self, ids):
        text = " ".join([self.int_to_token[i] for i in ids])
        # Replace spaces before the specified punctuations
        return re.sub(r'\s+([,.?!"()\'])', r'\1', text)

# Initialize and test tokenizer layer
tokenizer = TokenizerLayer(vocabulary)
print(tokenizer.encode(""""It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""))
print(tokenizer.decode(tokenizer.encode("""It's the last he painted, you know," Mrs. Gisburn said with pardonable pride.""")))

print(tokenizer.encode(""""This is a test! <|endoftext|> What is your favourite movie"""))
print(tokenizer.decode(tokenizer.encode("""This is a test! <|endoftext|> What is your favourite movie""")))
```

    [1, 58, 2, 872, 1013, 615, 541, 763, 5, 1155, 608, 5, 1, 69, 7, 39, 873, 1136, 773, 812, 7]
    It' s the last he painted, you know," Mrs. Gisburn said with pardonable pride.
    [1, 101, 595, 119, 1160, 0, 1159, 113, 595, 1157, 1160, 1160]
    This is a <|unk|>! <|endoftext|> What is your <|unk|> <|unk|>
    

#### Byte Pair Encoding (BPE)
GPT-2 uses BPE as its tokenizer. This allows it to break down words that aren't in its predefined vocabulary into smaller subword units or even individual characters, enabling it to handle out-of-vocabulary words.

For example, if GPT-2's vocabulary doesn't have the word "unfamiliarword," it might tokenize it as ["unfam", "iliar", "word"] or some other subword breakdown, depending on its trained BPE merges

Original BPE Tokenizer can be found here: https://github.com/openai/gpt-2/blob/master/src/encoder.py


To use BPE tokenizer, we can use OpenAI's open-source tiktoken library which implements its core algorithms in Rust to improve computational performance.


```python
# pip install tiktoken
```

    Collecting tiktoken
      Downloading tiktoken-0.6.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.8 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.8/1.8 MB[0m [31m8.3 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: regex>=2022.1.18 in /usr/local/lib/python3.10/dist-packages (from tiktoken) (2023.12.25)
    Requirement already satisfied: requests>=2.26.0 in /usr/local/lib/python3.10/dist-packages (from tiktoken) (2.31.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests>=2.26.0->tiktoken) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.26.0->tiktoken) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.26.0->tiktoken) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.26.0->tiktoken) (2024.2.2)
    Installing collected packages: tiktoken
    Successfully installed tiktoken-0.6.0
    


```python
import tiktoken
import importlib

print("tiktoken version:", importlib.metadata.version("tiktoken"))
```

    tiktoken version: 0.6.0
    


```python
tokenizer = tiktoken.get_encoding("gpt2")
text = "Hello, this is a test sentence from theouterspace. <|endoftext|> It's the last he painted, you know,"
token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(token_ids)

# Re-construct the input text using the token_ids
print(tokenizer.decode(token_ids))
```

    [15496, 11, 428, 318, 257, 1332, 6827, 13, 220, 50256, 632, 338, 262, 938, 339, 13055, 11, 345, 760, 11]
    Hello, this is a test sentence. <|endoftext|> It's the last he painted, you know,
    

BPE tokenizer breaks down the unknown words into subwords and individual characters.

#### Data sampling with sliding window
We train LLM to generate one word at a time, so we want to prepare the training data accordingly where the next word in a sequence represents the target to predict:


```python
from IPython.display import Image
Image(url="https://drive.google.com/file/d/1-IpY_qgU0n704QJmoQYf8cAFIpeTuvTx/view?usp=sharing")
```




<img src="https://drive.google.com/file/d/1-IpY_qgU0n704QJmoQYf8cAFIpeTuvTx/view?usp=sharing"/>




```python
with open("sample_data/the-verdict.txt", "r") as f:
    raw_text = f.read()

encoded_text = tokenizer.encode(raw_text)
print(len(encoded_text))
```

    5145
    

- For each ext chunk, we want inputs and targets
- Since we want the model to predict the next word, the targets are the inputs shifted by one position to the right.


```python
sample = encoded_text[:100]
context_length = 5

for i in range(1, context_length + 1):
    context = sample[:i]
    desired_target = sample[i]
    print(context, "->", desired_target)
```

    [40] -> 367
    [40, 367] -> 2885
    [40, 367, 2885] -> 1464
    [40, 367, 2885, 1464] -> 1807
    [40, 367, 2885, 1464, 1807] -> 3619
    


```python
for i in range(1, context_length + 1):
    context = sample[:i]
    desired_target = sample[i]
    print(tokenizer.decode(context), "->", tokenizer.decode([desired_target]))
```

    I ->  H
    I H -> AD
    I HAD ->  always
    I HAD always ->  thought
    I HAD always thought ->  Jack
    

### Data Loading
Next we implement a simple data loader ha iterates over the input dataset and returns the inputs and target shifted by one.


```python
import torch
print("PyTorch version:", importlib.metadata.version("torch"))
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-5-86d30c49cd31> in <cell line: 2>()
          1 import torch
    ----> 2 print("PyTorch version:", importlib.metadata.version("torch"))
    

    NameError: name 'importlib' is not defined


- We use sliding window approach where we slide the window one word at a time (this is also called stride=1)
- We create a dataset and dataloader object that extract chunks from the input text dataset.


```python
from torch.utils.data import Dataset, DataLoader

class LLMDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Iterate over the tokenized text
        for i in range(0, len(token_ids) - max_length, stride):
            context = token_ids[i:i+max_length]
            desired_target = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(context))
            self.target_ids.append(torch.tensor(desired_target))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_data_loader(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create the dataset
    dataset = LLMDataset(txt, tokenizer, max_length, stride)

    # Create the data loader
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)

```


```python
with open("sample_data/the-verdict.txt", "r") as f:
    raw_text = f.read()

dataloader = create_data_loader(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iterator = iter(dataloader)
batch = next(data_iterator)
print(batch)
```

    [tensor([[319, 616, 835, 284]]), tensor([[  616,   835,   284, 22489]])]
    


```python
batch_2 = next(data_iterator)
print(batch_2)
```

    [tensor([[ 11, 290,  11, 355]]), tensor([[ 290,   11,  355, 9074]])]
    


```python
# Increse the stride to remove overlaps between the batches since more overlap could lead to increased overfitting
dataloader = create_data_loader(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)
```

    Inputs:
     tensor([[41186, 39614,  1386,    11],
            [  373,  3957,   588,   262],
            [ 1169,  2994,   284,   943],
            [ 7067, 29396, 18443, 12271],
            [ 2666,   572,  1701,   198],
            [ 3666, 13674,    11,  1201],
            [ 1109,   815,   307,   900],
            [  465,  5986,   438,  1169]])
    
    Targets:
     tensor([[39614,  1386,    11,   287],
            [ 3957,   588,   262, 26394],
            [ 2994,   284,   943, 17034],
            [29396, 18443, 12271,   290],
            [  572,  1701,   198,   198],
            [13674,    11,  1201,   314],
            [  815,   307,   900,   866],
            [ 5986,   438,  1169,  3081]])
    

#### Creating token embeddings
Next we embed the token in a continuous vector representation using an embedding layer. Usually the embedding layers are part of the LLM itself and are updated (trained) during model training.


```python
# Suppose we have the following four input examples with ids 5,1,3 and 2 after tokenization
input_ids = torch.tensor([[5, 1, 3, 2]])
```

For simplicity, suppose we have a small vocabulary of only 6 words and we want to create embeddings of size 3:


```python
vocab_size = 6
embedding_size = 3

torch.manual_seed(42)
embedding_layer = torch.nn.Embedding(vocab_size, embedding_size)

# This would result in a 6x3 weight matrix
print(embedding_layer.weight)
```

    Parameter containing:
    tensor([[ 1.9269,  1.4873, -0.4974],
            [ 0.4396, -0.7581,  1.0783],
            [ 0.8008,  1.6806,  0.3559],
            [-0.6866,  0.6105,  1.3347],
            [-0.2316,  0.0418, -0.2516],
            [ 0.8599, -0.3097, -0.3957]], requires_grad=True)
    

The embedding output for our example input tensor will look as follows


```python
embedding_layer(input_ids)
```




    tensor([[[ 0.8599, -0.3097, -0.3957],
             [ 0.4396, -0.7581,  1.0783],
             [-0.6866,  0.6105,  1.3347],
             [ 0.8008,  1.6806,  0.3559]]], grad_fn=<EmbeddingBackward0>)



#### Encoding Word Positions

- Embedding layer convert Ids into identical vector representations regardless of where they are located in the input sequence.
- Positional embeddings are combined with the token embedding vector to form the input embedding for a large language model
- The BytePair encoder has a vocabulary size of 50,257
- To encode the input token to a 256-dimensional representation



```python
vocab_size = 50257
embedding_dim = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
```

- if we sample data from the dataloader, we embed the tokens in each batch into a 256-dim vector
- if we have a batch size of 8 with 4 tokens each, this will result in a 8x4x256 tensor:


```python
max_length = 4
dataloader = create_data_loader(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token Ids:\n", inputs)
print("\nInputs shape:\n", inputs.shape)
print("\nEmbedding shape:\n", token_embedding_layer(inputs).shape)
```

    Token Ids:
     tensor([[  273,  1807,   673,   750],
            [21978, 44896,    11,   290],
            [  991,  2045,   546,   329],
            [ 7808,   607, 10927,  1108],
            [ 3226,  1781,    11,  2769],
            [   11,   644,   561,   339],
            [  326,  9074,    13,   402],
            [  373, 37895,   422,   428]])
    
    Inputs shape:
     torch.Size([8, 4])
    
    Embedding shape:
     torch.Size([8, 4, 256])
    

- GPT-2 uses absolute position enbeddings, so we simply create another embedding layer



```python
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, embedding_dim)

position_embeddings = pos_embedding_layer(torch.arange(context_length))
print(position_embeddings.shape)
```

    torch.Size([4, 256])
    

- To create the input embeddings used in an LLM, we add the token and positional embeddings


```python
input_embeddings = token_embedding_layer(inputs) + position_embeddings
print(input_embeddings.shape)
```

    torch.Size([8, 4, 256])
    

The illustration below shows the end-to-end preprocessing steps of input tokens to an LLM model.

### Part II - Attention Mechanism
This section covers constructing the engine our LLM model. The multi-head attention mechanism is an extension of of self-attention with trainable weights that forms the basis of the mechanism used in LLM and the causal attention that allows a model to consider only previous and current inputs in a sequence, ensuring temporal order during the text generation. The multi-head attention enables the model to simultaneously attend to information from different representation subspaces.


```python
from importlib.metadata import version

print("torch version:", version("torch"))
```

    torch version: 2.2.1+cu121
    

#### Self - Attention

To compute the self-attention for each token in the input sequence, first we calculate the attention scores which is a sum over the dot product of each token with the surrounding token.

For example, given an input sequence $x^1$ to $x^T$, we compute the attention scores $λ^{ii}$ to $λ^{tt}$ for each input token.

$$λ^{ij} = q^i . k^j$$

where $q^i$ = $k^j$ when $i$ = $j$ for each token in the input sequence.

We then normalize the attention scores to derive attention weights that sum to 1 using softmax.

$$ω_k = \lambda^k / ∑_k e^{λ}$$

We use the attention weights to compute the context vectors through a weighted summation of the inputs.

For example, take the input sequence of length 7 "I love to eat apples and bananas" each representated by a a 5-dim input vector.


```python
inputs = torch.randn(7, 5)
print(inputs)
```

    tensor([[-1.3097, -0.2521, -0.3730,  0.1809,  0.2388],
            [-0.4120,  0.2998, -1.0139,  0.3835,  0.9702],
            [-0.9049,  0.5816, -0.7129, -1.1269,  1.7399],
            [-0.0759, -0.7884, -0.6603, -0.4297, -1.2277],
            [ 1.2722,  1.0661,  0.7959,  0.7088, -0.1230],
            [ 0.1089, -0.6882,  0.6953,  0.5249, -0.3855],
            [-0.2718, -2.1963,  0.2301, -0.0101,  1.0587]])
    


```python
attention_weights = torch.softmax(inputs @ inputs.T, dim=-1)
print(attention_weights)
```

    tensor([[3.6912e-01, 1.5550e-01, 2.2576e-01, 5.8963e-02, 5.8738e-03, 3.9562e-02,
             1.4523e-01],
            [1.0880e-01, 3.7323e-01, 4.3361e-01, 1.4222e-02, 1.4691e-02, 1.1216e-02,
             4.4244e-02],
            [1.1161e-02, 3.0639e-02, 9.5240e-01, 5.0936e-04, 2.9670e-04, 2.5658e-04,
             4.7327e-03],
            [5.8113e-02, 2.0034e-02, 1.0154e-02, 7.6827e-01, 9.7067e-03, 6.7473e-02,
             6.6254e-02],
            [2.2752e-03, 8.1332e-03, 2.3246e-03, 3.8148e-03, 9.5407e-01, 2.8013e-02,
             1.3683e-03],
            [6.9343e-02, 2.8097e-02, 9.0965e-03, 1.1999e-01, 1.2676e-01, 3.4991e-01,
             2.9680e-01],
            [6.5949e-03, 2.8716e-03, 4.3471e-03, 3.0526e-03, 1.6042e-04, 7.6895e-03,
             9.7528e-01]])
    


```python
# Verify that each row sums to 1
torch.sum(attention_weights, dim=-1)
```




    tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000])



We use the attention weights above to compute the hidden context vector as follows.

$z^i = ω^i . x^i$


```python
z = attention_weights @ inputs
print(z)
```

    tensor([[-7.8395e-01, -3.0156e-01, -4.2960e-01, -1.2986e-01,  6.9722e-01],
            [-6.8183e-01,  2.3620e-01, -7.0782e-01, -3.1608e-01,  1.1658e+00],
            [-8.9003e-01,  5.4959e-01, -7.1300e-01, -1.0594e+00,  1.6938e+00],
            [-1.5019e-01, -7.9001e-01, -4.8666e-01, -2.8172e-01, -8.4929e-01],
            [ 1.2077e+00,  9.9504e-01,  7.6583e-01,  6.9018e-01, -1.1895e-01],
            [-1.0345e-03, -8.5588e-01,  2.7241e-01,  2.3205e-01,  7.6075e-02],
            [-2.7799e-01, -2.1478e+00,  2.1940e-01, -9.5699e-03,  1.0378e+00]])
    




```python

```
