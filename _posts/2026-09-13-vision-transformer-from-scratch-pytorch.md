---
title: "Vision Transformer From Scratch in PyTorch: A Deep Dive into the Architecture"
date: 2026-09-13
permalink: /posts/2026/09/vision-transformer-from-scratch-pytorch/
blog_category: vision-language-modeling-from-scratch
blog_section: Vision Transformers
blog_series: Vision Transformer Fundamentals
blog_order: 10
blog_summary: "Build a Vision Transformer encoder from scratch in PyTorch, following an implementation that turns image patches into tokens, adds spatial context, applies pre-norm self-attention blocks, and trains on CIFAR-10."
read_time: true
tags:
  - Vision Transformers
  - PyTorch
  - Computer Vision
  - Multimodal AI
  - Deep Learning
---

A Vision Transformer (ViT) treats an image as a sequence of tokens. Instead of applying a convolutional stack that gradually builds spatial features, a ViT cuts the image into fixed-size patches, projects each patch into an embedding, adds positional information, and lets Transformer self-attention model relationships between all patches.

This post follows the accompanying PyTorch notebook and expands the comments into a complete explanation of the architecture. We will build each component independently before composing them into a classifier:

1. A convolutional **patch embedding** layer.
2. Learned **positional embeddings** and a trainable `[CLS]` token.
3. Multi-head self-attention using `nn.MultiheadAttention`.
4. A pre-normalization Transformer encoder block with residual connections and an MLP.
5. A complete Vision Transformer classifier.
6. A CIFAR-10 training loop with `CrossEntropyLoss` and Adam.

A ViT is a **vision encoder**, not yet a complete vision-language model. It produces contextual visual representations. A multimodal model can later connect those representations to a language model through a projection layer, a cross-attention module, or another learned interface.

## 1. The ViT Data Flow

The complete path through the model is:

```text
RGB image
   |
   |  Conv2d(kernel_size=P, stride=P)
   v
Patch embeddings: (B, N, D)
   |
   |  prepend [CLS] and add learned positions
   v
Token sequence: (B, N + 1, D)
   |
   |  repeated Transformer encoder blocks
   v
Contextual token sequence: (B, N + 1, D)
   |
   |  select the [CLS] token
   v
Classification head: (B, num_classes)
```

Here:

- $B$ is the batch size.
- $H$ and $W$ are the image height and width.
- $P$ is the patch size.
- $N = (H/P)(W/P)$ is the number of image patches.
- $D$ is the embedding dimension.

For a $224 \times 224$ RGB image with $16 \times 16$ patches:

$$
N = \frac{224}{16} \times \frac{224}{16} = 14 \times 14 = 196.
$$

Each raw patch contains:

$$
16 \times 16 \times 3 = 768
$$

pixel values, but the model can project that patch into any embedding dimension $D$, such as $192$, $384$, or $768$.

## 2. Patch Embedding: Turning Pixels into Tokens

A Transformer expects a sequence of vectors. An image is a grid, so the first job is to convert the grid into a sequence without losing the local patch structure.

### What a patch represents

A patch is a small rectangular region of the original image. With a patch size of $P \times P$, the image is divided into non-overlapping regions. Each region is flattened conceptually and passed through a learned projection:

$$
\mathbf{e}_i = \mathbf{r}_i W_E + \mathbf{b}_E,
$$

where $\mathbf{r}_i$ is the flattened pixel vector and $\mathbf{e}_i \in \mathbb{R}^{D}$ is the resulting token.

The notebook implements this projection with a convolution:

```python
nn.Conv2d(
    in_channels=3,
    out_channels=embed_dim,
    kernel_size=patch_size,
    stride=patch_size,
)
```

This is an efficient equivalence. The kernel covers one patch at a time, and the stride ensures neighboring convolution windows do not overlap. The output spatial grid is therefore the patch grid.

### PyTorch implementation

```python
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # x: (batch, channels, height, width)
        x = self.proj(x)
        # x: (batch, embed_dim, grid_height, grid_width)
        x = x.flatten(2)
        # x: (batch, embed_dim, num_patches)
        x = x.transpose(1, 2)
        # x: (batch, num_patches, embed_dim)
        return x


patch_embedding = PatchEmbedding()
images = torch.randn(2, 3, 224, 224)
embeddings = patch_embedding(images)
print(embeddings.shape)
# torch.Size([2, 196, 768])
```

The two tensor operations after the convolution are important:

- `flatten(2)` keeps the batch and channel dimensions while flattening the two-dimensional patch grid.
- `transpose(1, 2)` moves the embedding dimension to the final position, producing the sequence convention expected by `batch_first=True` attention modules.

### Patch embedding is not a full CNN

Patch embedding performs tokenization and projection, but it does not build a deep hierarchy of features by itself. A conventional CNN repeatedly applies convolution, activation, and pooling to move from edges to textures, parts, and objects. A ViT performs one initial patch projection and delegates most feature interaction to attention and MLP blocks.

That difference is architectural, not a claim that one approach is always better. CNNs have strong local and translation-related inductive biases. ViTs have a direct global communication mechanism once the patches become tokens.

## 3. Positional Embeddings: Telling the Model Where a Patch Came From

Self-attention compares tokens, but the attention operation itself does not know whether a token came from the top-left or bottom-right of an image. Without positional information, the patch sequence behaves like an unordered set.

The notebook uses learned positional embeddings:

```python
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, max_len + 1, embed_dim)
        )

    def forward(self, x):
        return x + self.pos_embedding[:, :x.size(1), :]
```

For token $i$, the initial Transformer representation is:

$$
\mathbf{x}_i^{(0)} = \mathbf{e}_i + \mathbf{p}_i,
$$

where $\mathbf{e}_i$ describes what is in the patch and $\mathbf{p}_i$ describes where the patch is located.

This addition is simple but essential. A patch containing an eye should not be treated identically when it occurs above a mouth versus below it. Positional embeddings give the attention mechanism a learned coordinate system.

The `max_len + 1` capacity leaves room for the extra `[CLS]` token that will be prepended later. The slice `[:, :x.size(1), :]` makes the module compatible with the actual sequence length at runtime.

## 4. The `[CLS]` Token and Classification

The ViT prepends a trainable classification token to the patch sequence:

```text
[CLS], patch_1, patch_2, ..., patch_N
```

The `[CLS]` token does not correspond to a physical image region. It is a learned aggregation slot. Because it participates in every self-attention layer, it can gather information from all patch tokens. At the end of the encoder, the model uses the contextualized `[CLS]` representation as a summary of the image.

For a batch of images, the class token begins with shape `(1, 1, D)` and is expanded to `(B, 1, D)`:

```python
cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
x = torch.cat((cls_tokens, x), dim=1)
x = self.positional_encoding(x)
```

`expand` creates a batch-shaped view without copying the underlying parameter for each example. The concatenated sequence now has shape `(B, N + 1, D)`.

## 5. Multi-Head Self-Attention

After patchification, each image is represented as:

$$
X \in \mathbb{R}^{B \times (N+1) \times D}.
$$

Self-attention lets every token request information from every other token. For one head:

$$
Q = XW^Q, \qquad K = XW^K, \qquad V = XW^V,
$$

and:

$$
\operatorname{Attention}(Q,K,V) =
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

The three projections have separate jobs:

- **Query:** What information does this token need?
- **Key:** What searchable characteristics does this token offer?
- **Value:** What content should be sent if the token is selected?

For image patches, this allows a patch containing part of a wheel to interact with patches containing the rest of the vehicle, the road, or the surrounding context. The model can learn these relationships rather than receiving them from a hand-designed receptive field.

### Why multiple heads?

If $D$ is split across $H$ heads, each head operates on a smaller subspace with dimension:

$$
 d_{\text{head}} = \frac{D}{H}.
$$

Different heads can learn different relationships: one may emphasize local shape continuity, another may connect distant object parts, and another may focus on texture or background context. This is a useful interpretation, not a guarantee that each head learns a clean human-readable role.

The number of heads must divide the embedding dimension. For example, $D=64$ and $H=8$ gives $d_{\text{head}}=8$.

### PyTorch implementation

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x):
        # Passing x three times creates self-attention:
        # x supplies queries, keys, and values.
        return self.attn(x, x, x, need_weights=False)[0]
```

Passing the same tensor three times does not mean the model uses one shared projection. `nn.MultiheadAttention` applies distinct learned projections internally. The same sequence is simply the source for the query, key, and value roles.

The `dropout` argument regularizes attention probabilities during training. It discourages the model from depending too heavily on one exact token-to-token connection and is disabled by `model.eval()` during evaluation.

## 6. Transformer Encoder Block

A ViT stacks encoder blocks. Each block combines:

1. Multi-head self-attention for communication between tokens.
2. A token-wise MLP for nonlinear feature transformation.
3. Residual connections for stable information and gradient flow.
4. Layer normalization for controlled feature scales.

The notebook uses a pre-normalization design:

$$
Y = X + \operatorname{MHA}(\operatorname{LayerNorm}(X)),
$$

$$
Z = Y + \operatorname{MLP}(\operatorname{LayerNorm}(Y)).
$$

The shape remains unchanged:

$$
(B, N+1, D) \rightarrow (B, N+1, D).
$$

Preserving the shape is what makes residual addition possible and allows the same block pattern to be repeated many times.

```python
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        normalized_x = self.norm1(x)
        attention_output, _ = self.mha(
            normalized_x,
            normalized_x,
            normalized_x,
            need_weights=False,
        )
        x = x + attention_output
        x = x + self.mlp(self.norm2(x))
        return x
```

### Why the MLP is needed

Attention mixes information across token positions. The MLP then transforms each token independently at every position. A typical MLP expands the feature width from $D$ to a larger `mlp_dim`, applies GELU, and projects back to $D$:

$$
\operatorname{MLP}(x) = W_2\,\operatorname{GELU}(W_1x+b_1)+b_2.
$$

Attention answers “which other tokens should influence this token?” The MLP answers “how should this token transform the information it now contains?” They are complementary operations.

## 7. Assemble the Vision Transformer

The complete model validates two important configuration constraints:

- `img_size` must be divisible by `patch_size`, otherwise the patch grid is fractional.
- `embed_dim` must be divisible by `num_heads`, otherwise the head dimension is not integral.

```python
class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        num_heads=8,
        mlp_dim=1024,
        num_layers=6,
        num_classes=10,
        dropout=0.1,
    ):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.img_size = img_size
        self.patch_embedding = PatchEmbedding(
            patch_size, in_channels, embed_dim
        )
        num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.positional_encoding = PositionalEncoding(embed_dim, num_patches)
        self.transformer_encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim, num_heads, mlp_dim, dropout=dropout
            )
            for _ in range(num_layers)
        ])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        if x.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(
                f"Expected images of size {self.img_size}x{self.img_size}, "
                f"but received {x.shape[-2]}x{x.shape[-1]}"
            )

        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.positional_encoding(x)

        for block in self.transformer_encoder_blocks:
            x = block(x)

        cls_output = x[:, 0]
        return self.mlp_head(cls_output)
```

### Shape trace with a lightweight configuration

The notebook tests a smaller model so a forward pass is inexpensive:

```python
model = VisionTransformer(
    img_size=32,
    patch_size=8,
    embed_dim=64,
    num_heads=4,
    mlp_dim=128,
    num_layers=2,
    num_classes=10,
    dropout=0.1,
)

test_images = torch.randn(2, 3, 32, 32)
logits = model(test_images)
print(logits.shape)
# torch.Size([2, 10])
```

For these settings:

- A $32 \times 32$ image produces $(32/8)^2 = 16$ patches.
- The `[CLS]` token makes the sequence length $17$.
- Each token has width $64$.
- The encoder output has shape `(2, 17, 64)`.
- Selecting `x[:, 0]` produces `(2, 64)`.
- The classification head produces `(2, 10)` logits.

## 8. Why Depth Matters

One self-attention layer already gives every patch a global receptive field. Adding encoder layers is therefore not primarily about making the receptive field larger. It is about iterative refinement:

```text
Early blocks    -> edges, textures, local patch relationships
Middle blocks   -> shapes, parts, and spatial configurations
Later blocks    -> objects, global context, task-specific semantics
```

Every encoder block has its own parameters unless weights are explicitly shared. Increasing `num_layers` increases model capacity, memory usage, training time, and inference latency.

The right depth depends on dataset size, task complexity, embedding width, regularization, and compute budget. A small educational model with two blocks is useful for understanding the data flow; a production-scale ViT uses a much larger training recipe and usually additional normalization, augmentation, scheduling, and evaluation choices.

## 9. Train on CIFAR-10

The notebook uses CIFAR-10 as a compact image-classification example. CIFAR-10 images are $32 \times 32$, while the default model expects $224 \times 224$, so the transform resizes each image before converting it to a tensor.

```python
import torch.optim as optim
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)
test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
)

model = VisionTransformer(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(10):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch [{epoch + 1}/10], "
        f"Loss: {running_loss / len(train_loader):.4f}"
    )
```

`CrossEntropyLoss` expects raw class logits and integer class labels. The model should not apply `softmax` before the loss because `CrossEntropyLoss` combines the log-softmax and negative log-likelihood operations internally in a numerically stable way.

### Practical limitations of this training example

The notebook provides a simplified training loop rather than a full benchmark. For a serious CIFAR-10 experiment, add a test evaluation loop, normalization with the dataset's channel statistics, data augmentation, learning-rate scheduling, checkpointing, and accuracy metrics. Resizing $32 \times 32$ images to $224 \times 224$ also increases compute without adding new visual information, so smaller model settings can be useful for experimentation.

## 10. From ViT to Vision-Language Modeling

This implementation ends with an image classification head. A vision-language model needs one more bridge between visual representations and language representations.

### Self-attention versus cross-attention

The ViT encoder uses self-attention because queries, keys, and values all originate from the same visual sequence:

$$
Q = XW^Q, \qquad K = XW^K, \qquad V = XW^V.
$$

In cross-attention, the query sequence and the context sequence are different. For example, language hidden states can provide queries while visual tokens provide keys and values:

$$
Q = LW^Q, \qquad K = VW^K, \qquad V = VW^V,
$$

where $L$ represents language tokens and $V$ represents visual tokens. A language token can therefore ask which visual regions are relevant, while the visual sequence supplies the information to retrieve. This is one common bridge from a standalone ViT encoder to a multimodal language model.

A common architecture is:

```text
Image -> ViT encoder -> visual token representations
                         |
                         +-> projection / resampler
                         |
Text tokens -> language model <- visual context
```

The ViT supplies visual tokens, but it does not produce text by itself. A multimodal language model may:

- Project visual embeddings into the language model's hidden dimension.
- Insert visual tokens into a language sequence.
- Use cross-attention from language queries to visual keys and values.
- Train the vision encoder, connector, language model, or selected subsets of them.

This distinction matters when reading model documentation. “Vision Transformer” describes the visual encoder architecture. “Vision-language model” describes a larger system that connects visual representations to language generation or understanding.

## Summary

The notebook's ViT implementation is a compact but complete study of the architecture:

1. **Patch embedding** converts a spatial image grid into a token sequence.
2. **Positional embeddings** restore spatial information that attention alone does not provide.
3. A trainable **`[CLS]` token** gathers image-level information for classification.
4. **Multi-head self-attention** lets every patch exchange information with every other patch.
5. The **MLP** transforms each token after attention has mixed contextual information.
6. **Pre-normalization and residual connections** stabilize the repeated encoder blocks.
7. The final **classification head** maps the contextual `[CLS]` token to class logits.
8. The same visual encoder can become part of a **vision-language model** when connected to a language system through a learned interface.

The main mental model is simple: a ViT turns an image into a sentence-like sequence of patch tokens, adds coordinates, repeatedly lets those tokens communicate, and then uses the resulting contextual representation for a downstream task.
