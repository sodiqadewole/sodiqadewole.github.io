---
title: 'Retrieval Augmented Generation with Phi-3-mini-4k-instruct and LlamaCpp'
date: 2026-03-26
permalink: /posts/2026/03/retrieval-augmented-generation/
blog_category: ai-agents
blog_section: Retrieval-Augmented Generation
blog_series: Local RAG Systems
blog_order: 10
blog_summary: "Build a local retrieval-augmented generation workflow with Phi-3-mini, LlamaCpp, and LangChain."
tags:
  - Retrieval Augmented Generation
  - Phi-3-mini-4k-instruct
  - LlamaCpp
  - Large Language Model
---

Retrieval-Augmented Generation (RAG) improves a language model's answers by giving it retrieved context at inference time. Instead of asking the model to answer only from its parameters, we first retrieve relevant text from a document store and pass that text into the prompt.

In this post, I build a local RAG workflow with Phi-3-mini-4k-instruct, LlamaCpp, LangChain, Hugging Face embeddings, and a FAISS vector store. The example uses content about large language models, retrieves relevant chunks, and asks Phi-3 to answer from the retrieved context.

What this post covers:

- Downloading a local GGUF Phi-3-mini model.
- Loading the model with LangChain's LlamaCpp wrapper.
- Creating sentence embeddings with `thenlper/gte-small`.
- Building a local FAISS vector database.
- Checking retrieval quality with similarity search.
- Connecting the retriever to a `RetrievalQA` chain.

### Downloading the Local Model

The first step is to download the GGUF model file. This makes the generation model available locally instead of calling a hosted API.

``` python
!wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-fp16.gguf
```

    --2026-03-28 22:59:12--  https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-fp16.gguf
    Resolving huggingface.co (huggingface.co)... 2600:9000:2732:a000:17:b174:6d00:93a1, 2600:9000:2732:fc00:17:b174:6d00:93a1, 2600:9000:2732:6200:17:b174:6d00:93a1, ...
    Connecting to huggingface.co (huggingface.co)|2600:9000:2732:a000:17:b174:6d00:93a1|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Saving to: ‘Phi-3-mini-4k-instruct-fp16.gguf’
    ....

    Phi-3-mini-4k-instr 100%[===================>]   7.12G  34.5MB/s    in 3m 35s  

    2026-03-28 23:02:48 (33.8 MB/s) - ‘Phi-3-mini-4k-instruct-fp16.gguf’ saved [7643295904/7643295904]

#### Load the generation model

Here, LlamaCpp loads the downloaded model and places the available layers on the GPU. The context window and maximum generation length are kept modest for a local experiment.

``` python
from langchain import LlamaCpp

# Load the model from the downloaded file
model_path = "Phi-3-mini-4k-instruct-fp16.gguf"
llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=-1,  # Use all available GPU layers
    n_ctx=2048,       # Context window size
    max_tokens=1024,   # Number of tokens to predict
    seed=42,          # Set a random seed for reproducibility
    verbose=False      # Enable verbose output
    )
```
#### Load the Embedding Model

The embedding model converts each text chunk into a vector. Those vectors make semantic search possible because similar meanings should land near each other in embedding space.

``` python
from langchain.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
```

Let's prepare the source text and set up the vector database using the embedding model.

``` python
import wikipedia
# Get a summary of what python is
page = wikipedia.page("large language models").content
print(page[:500])

# split text into lines and remove empty lines
lines = [text.strip('\n') for text in page.split(' \n\n ') if text.strip('\n') != '']

sentences = []
for line in lines:
    line_sentences = [sent.strip() for sent in line.split('.') if sent.strip() != '' and len(sent.strip()) > 20]
    sentences.extend(line_sentences)

sentences[:5]
```

    A large language model (LLM) is a computational model trained on a vast amount of data, designed for natural language processing tasks, especially language generation. The largest and most capable LLMs are generative pre-trained transformers (GPTs) that provide the core capabilities of modern chatbots. LLMs use large numbers of artificial intelligence parameters to generate, summarize, translate and reason over text in a large variety of contexts. Models may be fine-tuned to perform specific tas

    ['A large language model (LLM) is a computational model trained on a vast amount of data, designed for natural language processing tasks, especially language generation',
     'The largest and most capable LLMs are generative pre-trained transformers (GPTs) that provide the core capabilities of modern chatbots',
     'LLMs use large numbers of artificial intelligence parameters to generate, summarize, translate and reason over text in a large variety of contexts',
     'Models may be fine-tuned to perform specific tasks, or users can give them more specific prompts to refine their output',
     'As LLMs are trained on collections of human-written text, they are able to reflect patterns in natural language']

``` python
from langchain.vectorstores import FAISS

# Create a local FAISS vector store
vector_store = FAISS.from_texts(sentences, embedding_model)
```

Before building the full RAG chain, it helps to inspect the raw retrieval output for a simple query.

``` python
vector_store.similarity_search("What are large language models?", k=3)
```

    [Document(page_content='A large language model (LLM) is a computational model trained on a vast amount of data, designed for natural language processing tasks, especially language generation'),
     Document(page_content='The tendency towards larger models is visible in the list of large language models'),
     Document(page_content='"Baby steps in evaluating the capacities of large language models"')]

### Building the RAG Prompt and Chain

The prompt template inserts retrieved context above the user question. The `RetrievalQA` chain then handles retrieval, prompt construction, and generation.

``` python
from langchain import PromptTemplate

# Create a prompt template for RAG
prompt_template = PromptTemplate(
    template = """
    <|user|>
    Relevant information:
    {context}
    
    Provide a concise answer to the following question using the relevant information provided above:
    {question}<|end|>
    <|assistant|>
    """,
    input_variables=["context", "question"]
)

from langchain.chains import RetrievalQA

# Create a RetrievalQA chain
rag = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template},
    verbose=True
)

rag.invoke("language model architecture")
```

    > Entering new RetrievalQA chain...

    > Finished chain.

    {'query': 'language model architecture',
     'result': ' The architecture used by large language models (LLMs) is based on the transformer design. This marks an evolution from previous statistical and recurrent neural network approaches, offering enhanced capabilities for handling complex natural language processing tasks due to its effective scaling and ability to process sequential data.'}

The answer correctly uses the retrieved context to connect large language models with transformer architecture. For a stronger production workflow, the next step would be to improve chunking, add source citations, tune `k`, and evaluate answers against a small set of known questions.

### Takeaways

This local RAG pipeline shows the core pieces: a generator, an embedding model, a vector store, and a prompt that binds retrieved evidence to the question. Even with a compact setup, the model can answer more grounded questions when the right context is retrieved first.
