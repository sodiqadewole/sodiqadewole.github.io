---
title: "Finetuning Llava-Next for Document Extraction"
date: 2026-09-02
permalink: /posts/2026/09/finetuning-llava-next-document-extraction/
blog_category: multimodal-finetuning
blog_section: Vision-Language Models
blog_summary: "Fine-tune LLaVA-NeXT with QLoRA on CORD-V2 receipt images for JSON-style document extraction, from dataset formatting through collators, Lightning training, and inference parsing."
read_time: true
tags:
  - Multimodal Finetuning
  - LLaVA-NeXT
  - Document Parsing
  - QLoRA
  - PEFT
  - CORD-V2
---

Document parsing is a useful test case for multimodal language models because the model has to read an image, understand the visual layout, and emit structured text. In this walkthrough, I fine-tune LLaVA-NeXT on receipt images from CORD-V2 so the model can convert a document image into a JSON-like extraction format.

The notebook uses `llava-hf/llava-v1.6-mistral-7b-hf`, Hugging Face datasets, a LLaVA-NeXT processor, PEFT/LoRA, 4-bit quantization, and PyTorch Lightning. The goal is not only to show the code, but to explain why each piece exists: how receipt labels are converted into target token sequences, how image-text batches are collated, how validation uses edit distance, and why memory constraints matter when training a 7B vision-language model locally.

This walkthrough references the Hugging Face Transformers documentation for [LLaVA-NeXT](https://huggingface.co/docs/transformers/main/en/model_doc/llava_next) and is based on the multimodal tutorials from [Niels Rogge's Transformers-Tutorials repository](https://github.com/NielsRogge/Transformers-Tutorials/tree/master).

What this post covers:

- Loading the CORD-V2 receipt parsing dataset.
- Inspecting the receipt image and ground-truth JSON.
- Loading a LLaVA-NeXT processor and quantized model.
- Applying LoRA adapters with PEFT.
- Building a PyTorch dataset that converts JSON labels into token sequences.
- Defining train and evaluation collators for multimodal batches.
- Wrapping the model in a Lightning module for training and validation.
- Running inference and converting generated extraction tokens back into JSON.
- Interpreting the CUDA out-of-memory failure from the saved training run.

## LLaVA-NeXT Image Processing at a Glance

The official Hugging Face documentation describes LLaVA-NeXT as a model that combines a vision backbone and a language model. The original LLaVA-NeXT blog highlights one of the most important changes from LLaVA-1.5: dynamic high-resolution image processing, which preserves more visual detail by splitting images into grids instead of forcing everything through a single low-resolution square.

![Original LLaVA-NeXT dynamic high-resolution image processing diagram](https://llava-vl.github.io/blog/assets/images/llava-1-6/high_res_arch_v2.png)

Source: [LLaVA-NeXT: Improved reasoning, OCR, and world knowledge](https://llava-vl.github.io/blog/2024-01-30-llava-next/).

For the document extraction workflow in this post, the practical implication is that receipt images can produce multiple visual crops or patches before reaching the language model. The processor prepares `pixel_values`, `image_sizes`, `input_ids`, and `attention_mask`; the vision tower produces image features; the multimodal projector maps those features into the language model's hidden space; and the decoder generates the extraction sequence. During QLoRA finetuning, the base model stays quantized and mostly frozen while LoRA adapters learn the document extraction behavior.

## Install Dependencies

The notebook starts with the packages needed for multimodal finetuning: PyTorch, Transformers, Datasets, PEFT, BitsAndBytes, Lightning, and NLTK. The install line is commented out because it only needs to be run when the environment is not already prepared.

```python
# !pip install torch torchvision transformers datasets accelerate bitsandbytes peft lightning nltk
```

The next cell suppresses warnings so the notebook output stays readable, and imports `pprint` for formatting long Python objects.

```python
import warnings
warnings.filterwarnings("ignore")

import pprint
```

## Define Core Settings

`MAX_LENGTH` controls the maximum number of new tokens generated during validation or inference. `MODEL_ID` points to the LLaVA-NeXT checkpoint that combines a vision encoder with a Mistral language model.

```python
MAX_LENGTH = 256
MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
```

## Load CORD-V2

CORD-V2 contains receipt images and structured labels. Each row has an image and a `ground_truth` field containing the target parse.

```python
from datasets import load_dataset

dataset = load_dataset("naver-clova-ix/cord-v2")

dataset
```

The saved notebook output shows the dataset splits:

```text
DatasetDict({
    train: Dataset({
        features: ['image', 'ground_truth'],
        num_rows: 800
    })
    validation: Dataset({
        features: ['image', 'ground_truth'],
        num_rows: 100
    })
    test: Dataset({
        features: ['image', 'ground_truth'],
        num_rows: 100
    })
})
```

## Inspect a Receipt Image

Before training, it helps to look at one sample. The image is resized only for display; this does not change the original dataset used later.

```python
sample = dataset["train"][0]

width, height = sample["image"].size
new_width = int(width * 0.5)
new_height = int(height * 0.5)

sample["image"] = sample["image"].resize((new_width, new_height))
sample["image"]
```

## Inspect the Ground Truth JSON

The `ground_truth` field is stored as a JSON string. For CORD-V2, the useful structured target is under `gt_parse`.

```python
import json

json.loads(sample["ground_truth"])["gt_parse"]
```

The sample receipt contains menu items, subtotal fields, and a final total:

```text
{
  'menu': [
    {'nm': 'Nasi Campur Bali', 'cnt': '1 x', 'price': '75,000'},
    {'nm': 'Bbk Bengil Nasi', 'cnt': '1 x', 'price': '125,000'},
    {'nm': 'MilkShake Starwb', 'cnt': '1 x', 'price': '37,000'},
    ...
  ],
  'sub_total': {
    'subtotal_price': '1,346,000',
    'service_price': '100,950',
    'tax_price': '144,695',
    'etc': '-45'
  },
  'total': {'total_price': '1,591,600'}
}
```

The training target is not plain JSON text. The notebook later converts this object into a sequence of structural tags like `<s_menu>`, `<s_nm>`, and `<sep/>` so the model can learn a deterministic extraction grammar.

## Load the Processor

The processor is responsible for preparing both text and images in the format expected by LLaVA-NeXT. It applies the chat template, tokenizes the prompt, preprocesses images, and returns tensors such as `input_ids`, `pixel_values`, and `image_sizes`.

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right"
```

Right-side padding is used because labels are aligned with the text sequence during training.

## Load the Quantized Model

The model is large, so the notebook uses QLoRA-style 4-bit loading. The base weights are quantized with BitsAndBytes, while LoRA adapters provide the trainable parameters.

```python
from transformers import BitsAndBytesConfig, LlavaNextForConditionalGeneration
import torch

USE_LORA = False
USE_QLORA = True

if USE_LORA or USE_QLORA:
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
        )
        model = LlavaNextForConditionalGeneration.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            torch_dtype="float16",
        )
    else:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            MODEL_ID,
            load_in_8bit=True,
        )
```

The intent is to keep memory low enough to run on a local GPU. Even with 4-bit weights, the model can still be memory-hungry because LLaVA-NeXT must process both visual tokens and language tokens.

## Apply PEFT with LoRA

PEFT freezes the base model and trains lightweight adapter layers. The helper below finds linear layers that can receive LoRA adapters, while skipping the multimodal projector and vision tower.

```python
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["multi_modal_projector", "vision_model"]

    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")

    return list(lora_module_names)


lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=find_all_linear_names(model),
    init_lora_weights="gaussian",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

model
```

The saved output shows a PEFT-wrapped LLaVA-NeXT model. In the training summary, the full model is about 3.9B parameters, but only 24.5M parameters are trainable. That is the practical advantage of LoRA: the adapter is small compared with the frozen backbone.

## Create a PyTorch Dataset

The custom dataset turns each CORD-V2 row into two objects: the receipt image and a target token sequence. The important method is `json2token`, which recursively converts nested JSON into an ordered tag sequence.

```python
from torch.utils.data import Dataset
from typing import Any, Dict
import random


class CustomLlavaNextDataset(Dataset):
    """
    PyTorch Dataset for LLaVA-NeXT. This class takes a Hugging Face dataset as input.

    Each row contains a receipt image and ground-truth extraction data.
    """
    def __init__(
        self,
        dataset_name_or_path: str,
        split: str = "train",
        sort_json_key: bool = True,
    ):
        super().__init__()

        self.split = split
        self.sort_json_key = sort_json_key

        self.dataset = load_dataset(dataset_name_or_path, split=self.split)
        self.dataset_length = len(self.dataset)

        self.gt_token_sequences = []
        for sample in self.dataset:
            ground_truth = json.loads(sample["ground_truth"])
            if "gt_parses" in ground_truth:
                assert isinstance(ground_truth["gt_parses"], list)
                gt_jsons = ground_truth["gt_parses"]
            else:
                assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
                gt_jsons = [ground_truth["gt_parse"]]

            self.gt_token_sequences.append(
                [
                    self.json2token(
                        gt_json,
                        sort_json_key=self.sort_json_key,
                    )
                    for gt_json in gt_jsons
                ]
            )

    def json2token(self, obj: Any, sort_json_key: bool = True):
        """
        Convert an ordered JSON object into a token sequence.
        """
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]

            output = ""
            keys = sorted(obj.keys(), reverse=True) if sort_json_key else obj.keys()
            for key in keys:
                output += (
                    fr"<s_{key}>"
                    + self.json2token(obj[key], sort_json_key)
                    + fr"</s_{key}>"
                )
            return output

        if type(obj) == list:
            return r"<sep/>".join(
                [self.json2token(item, sort_json_key) for item in obj]
            )

        return str(obj)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Dict:
        sample = self.dataset[idx]

        image = sample["image"]
        target_sequence = random.choice(self.gt_token_sequences[idx])

        return image, target_sequence
```

The dataset class handles both single-parse labels and multi-parse labels. That makes it reusable for datasets where a document can have multiple valid annotations.

```python
train_dataset = CustomLlavaNextDataset(
    "naver-clova-ix/cord-v2",
    split="train",
    sort_json_key=False,
)

val_dataset = CustomLlavaNextDataset(
    "naver-clova-ix/cord-v2",
    split="validation",
    sort_json_key=False,
)
```

Sampling one item shows the tagged target sequence:

```python
import pprint

train_example = train_dataset[0]
image, target_sequence = train_example
pprint.pprint(target_sequence)
```

The saved output begins like this:

```text
<s_menu><s_nm>Nasi Campur Bali</s_nm><s_cnt>1 x</s_cnt><s_price>75,000</s_price><sep/>
<s_nm>Bbk Bengil Nasi</s_nm><s_cnt>1 x</s_cnt><s_price>125,000</s_price><sep/>
...
</s_menu><s_sub_total><s_subtotal_price>1,346,000</s_subtotal_price>...
<s_total><s_total_price>1,591,600</s_total_price></s_total>
```

This is the sequence the model learns to generate after seeing the receipt image and the instruction `Extract JSON`.

## Define the Collate Functions

The training collator builds a full conversation containing a user image prompt and an assistant answer. The processor then turns the conversation and image list into a padded multimodal batch.

```python
def train_collate_fn(examples):
    images = []
    texts = []

    for example in examples:
        image, ground_truth = example
        images.append(image)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Extract JSON"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ground_truth},
                ],
            },
        ]
        text_prompt = processor.apply_chat_template(conversation)
        texts.append(text_prompt)

    batch = processor(
        text=texts,
        images=images,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]
    image_sizes = batch["image_sizes"]
    labels = batch["labels"]

    return input_ids, attention_mask, pixel_values, image_sizes, labels
```

Two details matter here. First, `truncation=False` avoids truncating before LLaVA expands the image placeholder into visual tokens. Second, padding tokens are masked with `-100` so they do not contribute to the language modeling loss.

The evaluation collator uses only the user turn and adds a generation prompt. The ground truth is returned separately for scoring.

```python
def eval_collate_fn(examples):
    images = []
    texts = []
    answers = []

    for example in examples:
        image, ground_truth = example
        images.append(image)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Extract JSON"},
                ],
            },
        ]
        text_prompt = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )
        texts.append(text_prompt)
        answers.append(ground_truth)

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]
    image_sizes = batch["image_sizes"]

    return input_ids, attention_mask, pixel_values, image_sizes, answers
```

## Wrap Training in Lightning

The Lightning module keeps the training loop organized. `training_step` computes model loss, while `validation_step` generates predictions and compares them with the target extraction string using normalized edit distance.

```python
import lightning as L
from torch.utils.data import DataLoader
import re
from nltk import edit_distance
import numpy as np


class LlavaModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model

        self.batch_size = config.get("batch_size")

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, pixel_values, image_sizes, labels = batch

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            labels=labels,
        )
        loss = outputs.loss

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        input_ids, attention_mask, pixel_values, image_sizes, answers = batch

        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            max_new_tokens=MAX_LENGTH,
        )

        predictions = self.processor.batch_decode(
            generated_ids[:, input_ids.size(1):],
            skip_special_tokens=True,
        )

        scores = []
        for pred, answer in zip(predictions, answers):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

            if self.config.get("verbose", False) and len(scores) == 1:
                print(f"Prediction: {pred}")
                print(f"    Answer: {answer}")
                print(f" Normed ED: {scores[0]}")

        self.log("val_edit_distance", np.mean(scores))

        return scores

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))
        return optimizer

    def train_dataloader(self):
        return DataLoader(
            train_dataset,
            collate_fn=train_collate_fn,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            val_dataset,
            collate_fn=eval_collate_fn,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )
```

The validation metric is approximate, but useful: lower normalized edit distance means the generated extraction sequence is closer to the reference extraction sequence.

## Configure the Training Run

The config uses a batch size of 1 with gradient accumulation. This is a common local finetuning pattern: keep each GPU step small, then accumulate gradients to simulate a larger effective batch.

```python
config = {
    "max_epochs": 10,
    "check_val_every_n_epoch": 1,
    "gradient_clip_val": 1.0,
    "accumulate_grad_batches": 8,
    "lr": 1e-4,
    "batch_size": 1,
    "num_nodes": 1,
    "warmup_steps": 50,
    "result_path": "./result",
    "verbose": True,
}

model_module = LlavaModelPLModule(config, processor, model)
```

## Train the Model

The trainer is configured for a single GPU, mixed precision, gradient accumulation, and limited validation batches.

```python
from lightning.pytorch.loggers import WandbLogger

# wandb_logger = WandbLogger(project=WANDB_PROJECT, name=WANDB_NAME)

trainer = L.Trainer(
    accelerator="gpu",
    devices=[0],
    max_epochs=config.get("max_epochs"),
    accumulate_grad_batches=config.get("accumulate_grad_batches"),
    check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
    gradient_clip_val=config.get("gradient_clip_val"),
    precision="16-mixed",
    limit_val_batches=5,
    num_sanity_val_steps=0,
)

trainer.fit(model_module)
```

The saved training output shows that Lightning recognized CUDA and wrapped the model as a PEFT model:

```text
Using 16bit Automatic Mixed Precision (AMP)
GPU available: True (cuda), used: True
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

Name   Type      Params  Mode
model  PeftModel 3.9 B   train

Trainable params: 24.5 M
Non-trainable params: 3.9 B
Total params: 3.9 B
```

The run then hit a CUDA out-of-memory error:

```text
OutOfMemoryError: CUDA out of memory. Tried to allocate 66.00 MiB.
GPU 0 has a total capacity of 15.61 GiB of which 3.50 MiB is free.
```

This is an important result, not just a failure. It shows that 4-bit loading and LoRA do not automatically make every multimodal finetuning run fit on a 16GB GPU. LLaVA-NeXT expands images into visual tokens, and the combined image-text sequence can still exhaust memory.

Practical adjustments to try next:

- Lower `MAX_LENGTH` during validation and inference.
- Reduce image resolution if the task allows it.
- Use fewer validation batches or validate less frequently.
- Enable gradient checkpointing where supported.
- Reduce LoRA target modules or rank.
- Use a smaller LLaVA-style model for local experiments.
- Clear the CUDA cache before training after heavy notebook inspection.

## Reload for Inference

The notebook then reloads the base LLaVA-NeXT model with a 4-bit quantization config for inference.

```python
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaNextForConditionalGeneration
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
)
```

## Prepare a Test Receipt

The test example comes from the CORD-V2 test split. The prompt asks the model to extract JSON from the receipt image.

```python
test_example = dataset["test"][10]
test_image = test_example["image"]
test_image
```

```python
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Extract JSON"},
        ],
    },
]

text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(text=text_prompt, images=[test_image], return_tensors="pt").to("cuda")

for key, value in inputs.items():
    if key == "image_sizes":
        print("image_sizes:", value)
    else:
        print(key, value.shape)
```

The saved tensor shapes show how much context the model receives from a single receipt:

```text
input_ids torch.Size([1, 2173])
attention_mask torch.Size([1, 2173])
pixel_values torch.Size([1, 5, 3, 336, 336])
image_sizes: tensor([[864, 576]], device='cuda:0')
```

The image is split into multiple visual crops or patches, represented by `pixel_values` with shape `[1, 5, 3, 336, 336]`. That is one reason multimodal inference and training can consume memory quickly.

## Generate a Parse

The model generates token IDs autoregressively, then the processor decodes those IDs back into text.

```python
generated_ids = model.generate(**inputs, max_new_tokens=MAX_LENGTH)

generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
pprint.pprint(generated_texts)
```

The saved output is a plain-language receipt interpretation plus a JSON code block:

```text
[INST]
Extract JSON [/INST] The image you've provided appears to be a receipt from Auntie Anne's...

- Cinnamon Sugar: 1 x 17,000
- Grand Total: 17,000
- Cash IDR: 20,000
- Change Due: 3,000

Here is the JSON representation of the items listed on the receipt:

{
  "items": [
    {
      "item": "Cinnamon Sugar",
      "quantity": 1,
      "price": 17000
    }
  ],
  "total": 17000,
  "cash_idr": 20000,
  "change_due": 3000
}
```

This output is understandable, but it is not yet in the exact tagged CORD format used during training. That is why the next helper attempts to convert tagged token sequences back into JSON.

## Convert Tagged Tokens Back to JSON

The `token2json` helper reverses the earlier `json2token` process. It looks for `<s_key>...</s_key>` spans and reconstructs nested dictionaries and lists.

```python
import re


def token2json(tokens, is_inner_value=False, added_vocab=None):
    """
    Convert a generated token sequence into an ordered JSON format.
    """
    if added_vocab is None:
        added_vocab = processor.tokenizer.get_added_vocab()

    output = {}

    while tokens:
        start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
        if start_token is None:
            break

        key = start_token.group(1)
        key_escaped = re.escape(key)

        end_token = re.search(rf"</s_{key_escaped}>", tokens, re.IGNORECASE)
        start_token = start_token.group()
        if end_token is None:
            tokens = tokens.replace(start_token, "")
        else:
            end_token = end_token.group()
            start_token_escaped = re.escape(start_token)
            end_token_escaped = re.escape(end_token)
            content = re.search(
                f"{start_token_escaped}(.*?){end_token_escaped}",
                tokens,
                re.IGNORECASE | re.DOTALL,
            )

            if content is not None:
                content = content.group(1).strip()
                if r"<s_" in content and r"</s_" in content:
                    value = token2json(content, is_inner_value=True, added_vocab=added_vocab)
                    if value:
                        if len(value) == 1:
                            value = value[0]
                        output[key] = value
                else:
                    output[key] = []
                    for leaf in content.split(r"<sep/>"):
                        leaf = leaf.strip()
                        if leaf in added_vocab and leaf[0] == "<" and leaf[-2:] == "/>":
                            leaf = leaf[1:-2]
                        output[key].append(leaf)
                    if len(output[key]) == 1:
                        output[key] = output[key][0]

            tokens = tokens[tokens.find(end_token) + len(end_token):].strip()
            if tokens[:6] == r"<sep/>":
                return [output] + token2json(tokens[6:], is_inner_value=True, added_vocab=added_vocab)

    if len(output):
        return [output] if is_inner_value else output

    return [] if is_inner_value else {"text_sequence": tokens}
```

Because the saved inference output did not use the learned `<s_key>` tags, the parser falls back to a `text_sequence` wrapper:

```python
generated_json = token2json(generated_texts[0])
print(generated_json)
```

```text
{
  'text_sequence': '[INST]  \nExtract JSON [/INST] The image you\'ve provided appears to be a receipt from Auntie Anne\'s...'
}
```

Printing the parsed object makes the fallback behavior explicit:

```python
for key, value in generated_json.items():
    print(key, value)
```

```text
text_sequence [INST]
Extract JSON [/INST] The image you've provided appears to be a receipt from Auntie Anne's...
```

## What This Run Shows

This notebook demonstrates the full structure of a LLaVA-NeXT document parsing finetuning workflow, even though the saved training run ran out of GPU memory before completing. The useful lessons are practical:

- CORD-style labels need careful conversion between JSON and token sequences.
- Multimodal collators must keep image tensors, image sizes, prompt tokens, masks, and labels aligned.
- QLoRA reduces trainable parameter count dramatically, but visual tokens can still make training memory-intensive.
- Validation with normalized edit distance is a simple way to monitor structured extraction quality.
- Inference needs the model to produce the same structural tag format used during training; otherwise, downstream parsing falls back to unstructured text.

## Takeaways

Fine-tuning a vision-language model for document parsing is less about a single training call and more about designing the full extraction contract. The prompt, target token grammar, collator, validation metric, and JSON parser all need to agree. Once those pieces are aligned, LoRA or QLoRA can make local experimentation possible, but GPU memory still has to be treated as a first-class constraint.

## References

- [Hugging Face Transformers: LLaVA-NeXT documentation](https://huggingface.co/docs/transformers/main/en/model_doc/llava_next)
- [Niels Rogge: Transformers-Tutorials](https://github.com/NielsRogge/Transformers-Tutorials/tree/master)