---
title: "Building Multimodal ChatBot System"
excerpt: "Building Chatbot"
collection: portfolio
---

This project covers how to building a multimodal chat bot system using Multimodal Language Model.

``` python
# !pip install --upgrade transformers tokenizers
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

``` python
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

# Load the BLIP-2 model and processor
model_name = "Salesforce/blip2-opt-2.7b" # Specify the BLIP-2 model you want to use (e.g., "Salesforce/blip2-opt-2.7b")

processor = AutoProcessor.from_pretrained(model_name, revision="51572668da0eb669e01a189dc22abe6088589a24") # Load the processor for the specified BLIP-2 model
model = Blip2ForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype = torch.float16) # Load the BLIP-2 model for conditional generation (e.g., image captioning)

# send model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu" # Check if a
model.to(device) # Move the model to the appropriate device (GPU or CPU)

model
```

    Blip2ForConditionalGeneration(
      (vision_model): Blip2VisionModel(
        (embeddings): Blip2VisionEmbeddings(
          (patch_embedding): Conv2d(3, 1408, kernel_size=(14, 14), stride=(14, 14))
        )
        (encoder): Blip2Encoder(
          (layers): ModuleList(
            (0-38): 39 x Blip2EncoderLayer(
              (self_attn): Blip2Attention(
                (qkv): Linear(in_features=1408, out_features=4224, bias=True)
                (projection): Linear(in_features=1408, out_features=1408, bias=True)
              )
              (layer_norm1): LayerNorm((1408,), eps=1e-06, elementwise_affine=True)
              (mlp): Blip2MLP(
                (activation_fn): GELUActivation()
                (fc1): Linear(in_features=1408, out_features=6144, bias=True)
                (fc2): Linear(in_features=6144, out_features=1408, bias=True)
              )
              (layer_norm2): LayerNorm((1408,), eps=1e-06, elementwise_affine=True)
            )
          )
        )
        (post_layernorm): LayerNorm((1408,), eps=1e-06, elementwise_affine=True)
      )
      (qformer): Blip2QFormerModel(
        (layernorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (encoder): Blip2QFormerEncoder(
          (layer): ModuleList(
            (0): Blip2QFormerLayer(
              (attention): Blip2QFormerAttention(
                (attention): Blip2QFormerMultiHeadAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): Blip2QFormerSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (crossattention): Blip2QFormerAttention(
                (attention): Blip2QFormerMultiHeadAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=1408, out_features=768, bias=True)
                  (value): Linear(in_features=1408, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): Blip2QFormerSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate_query): Blip2QFormerIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output_query): Blip2QFormerOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (1): Blip2QFormerLayer(
              (attention): Blip2QFormerAttention(
                (attention): Blip2QFormerMultiHeadAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): Blip2QFormerSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate_query): Blip2QFormerIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output_query): Blip2QFormerOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (2): Blip2QFormerLayer(
              (attention): Blip2QFormerAttention(
                (attention): Blip2QFormerMultiHeadAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): Blip2QFormerSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (crossattention): Blip2QFormerAttention(
                (attention): Blip2QFormerMultiHeadAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=1408, out_features=768, bias=True)
                  (value): Linear(in_features=1408, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): Blip2QFormerSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate_query): Blip2QFormerIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output_query): Blip2QFormerOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (3): Blip2QFormerLayer(
              (attention): Blip2QFormerAttention(
                (attention): Blip2QFormerMultiHeadAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): Blip2QFormerSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate_query): Blip2QFormerIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output_query): Blip2QFormerOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (4): Blip2QFormerLayer(
              (attention): Blip2QFormerAttention(
                (attention): Blip2QFormerMultiHeadAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): Blip2QFormerSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (crossattention): Blip2QFormerAttention(
                (attention): Blip2QFormerMultiHeadAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=1408, out_features=768, bias=True)
                  (value): Linear(in_features=1408, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): Blip2QFormerSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate_query): Blip2QFormerIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output_query): Blip2QFormerOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (5): Blip2QFormerLayer(
              (attention): Blip2QFormerAttention(
                (attention): Blip2QFormerMultiHeadAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): Blip2QFormerSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate_query): Blip2QFormerIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output_query): Blip2QFormerOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (6): Blip2QFormerLayer(
              (attention): Blip2QFormerAttention(
                (attention): Blip2QFormerMultiHeadAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): Blip2QFormerSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (crossattention): Blip2QFormerAttention(
                (attention): Blip2QFormerMultiHeadAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=1408, out_features=768, bias=True)
                  (value): Linear(in_features=1408, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): Blip2QFormerSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate_query): Blip2QFormerIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output_query): Blip2QFormerOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (7): Blip2QFormerLayer(
              (attention): Blip2QFormerAttention(
                (attention): Blip2QFormerMultiHeadAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): Blip2QFormerSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate_query): Blip2QFormerIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output_query): Blip2QFormerOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (8): Blip2QFormerLayer(
              (attention): Blip2QFormerAttention(
                (attention): Blip2QFormerMultiHeadAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): Blip2QFormerSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (crossattention): Blip2QFormerAttention(
                (attention): Blip2QFormerMultiHeadAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=1408, out_features=768, bias=True)
                  (value): Linear(in_features=1408, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): Blip2QFormerSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate_query): Blip2QFormerIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output_query): Blip2QFormerOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (9): Blip2QFormerLayer(
              (attention): Blip2QFormerAttention(
                (attention): Blip2QFormerMultiHeadAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): Blip2QFormerSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate_query): Blip2QFormerIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output_query): Blip2QFormerOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (10): Blip2QFormerLayer(
              (attention): Blip2QFormerAttention(
                (attention): Blip2QFormerMultiHeadAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): Blip2QFormerSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (crossattention): Blip2QFormerAttention(
                (attention): Blip2QFormerMultiHeadAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=1408, out_features=768, bias=True)
                  (value): Linear(in_features=1408, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): Blip2QFormerSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate_query): Blip2QFormerIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output_query): Blip2QFormerOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (11): Blip2QFormerLayer(
              (attention): Blip2QFormerAttention(
                (attention): Blip2QFormerMultiHeadAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): Blip2QFormerSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate_query): Blip2QFormerIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output_query): Blip2QFormerOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
          )
        )
      )
      (language_projection): Linear(in_features=768, out_features=2560, bias=True)
      (language_model): OPTForCausalLM(
        (model): OPTModel(
          (decoder): OPTDecoder(
            (embed_tokens): Embedding(50304, 2560, padding_idx=1)
            (embed_positions): OPTLearnedPositionalEmbedding(2050, 2560)
            (final_layer_norm): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
            (layers): ModuleList(
              (0-31): 32 x OPTDecoderLayer(
                (self_attn): OPTAttention(
                  (k_proj): Linear(in_features=2560, out_features=2560, bias=True)
                  (v_proj): Linear(in_features=2560, out_features=2560, bias=True)
                  (q_proj): Linear(in_features=2560, out_features=2560, bias=True)
                  (out_proj): Linear(in_features=2560, out_features=2560, bias=True)
                )
                (activation_fn): ReLU()
                (self_attn_layer_norm): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
                (fc1): Linear(in_features=2560, out_features=10240, bias=True)
                (fc2): Linear(in_features=10240, out_features=2560, bias=True)
                (final_layer_norm): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
              )
            )
          )
        )
        (lm_head): Linear(in_features=2560, out_features=50304, bias=False)
      )
    )

``` python
model.save_pretrained("blip2-opt-2.7b") # Save the BLIP-2 model to a local directory (e.g., "blip2-opt-2.7b")
processor.save_pretrained("blip2-opt-2.7b") # Save the processor for the BLIP-2 model to the same local directory (e.g., "blip2-opt-2.7b")
```

```
    ['blip2-opt-2.7b/processor_config.json']

``` python
from urllib.request import urlopen
from PIL import Image

car_path = "https://www.chinadaily.com.cn/trending/img/attachement/jpg/site1/20150716/b083fe955a74171160c504.jpg" #"https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/car.png" # URL of the image to be processed (e.g., an image of a car)
image = Image.open(urlopen(car_path)).convert('RGB') # Load the image from the specified URL and convert it to RGB format

image
```

![](dbfc2ff16cbc24cedbd67befac573139d785198a.jpg)


``` python
# Preprocess the image
inputs = processor(images=image, return_tensors="pt").to(device, torch.float16) # Preprocess the image using the BLIP-2 processor and move the inputs to the appropriate device (GPU or CPU)
inputs['pixel_values'].shape # Check the shape of the preprocessed image tensor (e.g., [1, 3, 384, 384])
```

``` python
# Process the text
processor.tokenizer
```

    GPT2Tokenizer(name_or_path='Salesforce/blip2-opt-2.7b', vocab_size=50265, model_max_length=1000000000000000019884624838656, padding_side='right', truncation_side='right', special_tokens={'bos_token': '</s>', 'eos_token': '</s>', 'unk_token': '</s>', 'pad_token': '<pad>'}, added_tokens_decoder={
    	1: AddedToken("<pad>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True),
    	2: AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True),
    	50265: AddedToken("<image>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    })

``` python
# Preprocess the text
text = "Her vocalization was remarkably melodic"
token_ids = processor(image, text, return_tensors="pt").to(device, torch.float16) # Preprocess the text using the BLIP-2 processor and move the inputs to the appropriate device (GPU or CPU)
token_ids = token_ids.to(device, torch.float16)['input_ids'][0] # Ensure the token IDs are in the correct data type and on the appropriate device

# Convert input ids back to tokens
tokens = processor.tokenizer.convert_ids_to_tokens(token_ids) # Convert the input token IDs back to human-readable tokens using the tokenizer
tokens
```

    ['</s>', 'Her', 'Ġvocal', 'ization', 'Ġwas', 'Ġremarkably', 'Ġmel', 'odic']

``` python
# Replace the space token with underscore
tokens = [token.replace('Ġ', '_') for token in tokens] # Replace the space token (e.g., 'Ġ') with an underscore for better readability
tokens
```

    ['</s>', 'Her', '_vocal', 'ization', '_was', '_remarkably', '_mel', 'odic']

#### Applications: Image Captioning

``` python
# Load an AI-generated image of a supercar
image = Image.open(urlopen("https://www.chinadaily.com.cn/trending/img/attachement/jpg/site1/20150716/b083fe955a74171160c504.jpg")).convert('RGB') # Load an AI-generated image of a supercar from the specified URL and convert it to RGB format

# Convert an image into inputs and preprocess it
image_inputs = processor(images=image, return_tensors="pt").to(device, torch.float16) # Preprocess the image using the BLIP-2 processor and move the inputs to the appropriate device (GPU or CPU)
image_inputs['pixel_values'].shape # Check the shape of the preprocessed image tensor (e.g., [1, 3, 384, 384])
```

``` python
# Generate image ids to be passed to the decoder (LLM)
generated_ids = model.generate(**image_inputs, max_new_tokens=20) # Generate image IDs using the BLIP-2 model for the given image inputs, specifying the maximum number of new tokens to generate (e.g., 20)


# Generate text from the image features
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0] # Decode the generated image IDs back into human-readable text using the processor's batch_decode method, skipping any special

generated_text = generated_text.strip() # Remove any leading or trailing whitespace from the generated text
generated_text
```

    'a man in a red dress holding a plate of food next to a futuristic car'

``` python
# Load another image
url = "https://as2.ftcdn.net/v2/jpg/15/03/72/47/1000_F_1503724764_pB2di1c2uwzmwT7fgsroGRz2LdGdh3CG.jpg" # URL of another image to be processed (e.g., a Rorschach blot)
image = Image.open(urlopen(url)).convert('RGB') # Load the image from the specified


# Generate caption
inputs = processor(images=image, return_tensors="pt").to(device, torch.float16) # Preprocess the image using the BLIP-2 processor and move the inputs to the appropriate device (GPU or CPU)
generated_ids = model.generate(**inputs, max_new_tokens=20) # Generate image IDs using the BLIP-2 model for the given image inputs, specifying the maximum number of new tokens to
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0] # Decode the generated image IDs back into human-readable text using the processor's batch_decode method, skipping any special tokens
generated_text = generated_text.strip() # Remove any leading or trailing whitespace from the generated text
generated_text
```

    'a tiger and a wild boar fighting in the water'

# Application 2: Multimodal Chat-Based Prompting

``` python
image = Image.open(urlopen("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcShvbifbz-oZrRzMIzLfK8HbDOOav893Y85Rg&s.jpeg")).convert('RGB') # Load an AI-generated image of a supercar from the specified URL and convert it to RGB format

# Visual question answering
prompt = "Question: Write down what you see in this picture. Answer:" # Define a prompt for visual question answering (e.g., asking the model to describe what it sees in the image)

# Preprocess the image and prompt together
inputs = processor(image, prompt, return_tensors="pt").to(device, torch.float16) # Preprocess the image and prompt together using the BLIP-2 processor and move the inputs to the appropriate device (GPU or CPU)

# Generate text
generated_ids = model.generate(**inputs, max_new_tokens=20) # Generate image IDs using the BLIP-2 model for the given image and prompt inputs, specifying the maximum number of new tokens to generate (e.g., 20)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0] # Decode the generated image IDs back into human-readable text using the processor's batch_decode method, skipping any special tokens
generated_text = generated_text.strip() # Remove any leading or trailing whitespace from the generated text
generated_text
```

    'Question: Write down what you see in this picture. Answer: The sun.'

``` python
# Visual question answering
prompt = "Question: Write down what you see in this picture. Answer:A sports car driving on the road at sunset. Question: What would it cost me to drive that car? Answer:" # Define a prompt for visual question answering (e.g., asking the model to describe what it sees in the image)

# Preprocess the image and prompt together
inputs = processor(image, prompt, return_tensors="pt").to(device, torch.float16) # Preprocess the image and prompt together using the BLIP-2 processor and move the inputs to the appropriate device (GPU or CPU)

# Generate text
generated_ids = model.generate(**inputs, max_new_tokens=20) # Generate image IDs using the BLIP-2 model for the given image and prompt inputs, specifying the maximum number of new tokens to generate (e.g., 20)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0] # Decode the generated image IDs back into human-readable text using the processor's batch_decode method, skipping any special tokens
generated_text = generated_text.strip() # Remove any leading or trailing whitespace from the generated text
generated_text
```

    'Question: Write down what you see in this picture. Answer:A sports car driving on the road at sunset. Question: What would it cost me to drive that car? Answer: $0.'

### Create an interactive chat bot with ipywidgets

``` python
from IPython.display import HTML, display
import ipywidgets as widgets


def text_eventhandler(*args):
    question = args[0]['new']
    if question:
        args[0]['owner'].value = ""

        # Create prompt
        if not memory:
            prompt = " Question: " + question + " Answer:"
        else:
            template = "Question: {} Answer: {}"
            prompt = " ".join([
                template.format(memory[i][0], memory[i][1]) 
                for i in range(len(memory))
                ]) + " Question: " + question + " Answer:"
            
        # Generate text
        inputs = processor(image, prompt, return_tensors="pt").to(device, torch.float16)
        generated_ids = model.generate(**inputs, max_new_tokens=100)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip().split("Question:")[0]

        # Update memory
        memory.append((question, generated_text))

        # Assign to output
        output.append_display_data(HTML(f"<b>USER:</b> " + question))
        output.append_display_data(HTML(f"<br>BLIP-2:<b> " + generated_text))
        output.append_display_data(HTML("<br>"))

# Prepare widgets
in_text = widgets.Text()
in_text.continuous_update = False
in_text.observe(text_eventhandler, names='value')
output = widgets.Output()

memory = []

# Display chatbot
display(
    widgets.VBox(
        children=[output, in_text],
        layout=widgets.Layout(
            display='inline_flex',
            flex_flow='column-reverse'),
            )
        )
```