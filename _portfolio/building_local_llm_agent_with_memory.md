---
title: "LLM Agent with Memory"
excerpt: "Building Local LLM Agent with Memory"
collection: portfolio
---

This project covers how to build llm agent with conversation memory on your local machine using any pretrained model.
 
``` python
!wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-fp16.gguf
```


    --2026-03-22 16:16:47--  https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-fp16.gguf
    Resolving huggingface.co (huggingface.co)... 2600:9000:2732:3e00:17:b174:6d00:93a1, 2600:9000:2732:f200:17:b174:6d00:93a1, 2600:9000:2732:7400:17:b174:6d00:93a1, ...
    Connecting to huggingface.co (huggingface.co)|2600:9000:2732:3e00:17:b174:6d00:93a1|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    ...
    ...
    Resolving cas-bridge.xethub.hf.co (cas-bridge.xethub.hf.co)... 18.238.217.126, 18.238.217.88, 18.238.217.64, ...
    Connecting to cas-bridge.xethub.hf.co (cas-bridge.xethub.hf.co)|18.238.217.126|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 7643295904 (7.1G)
    Saving to: ‘Phi-3-mini-4k-instruct-fp16.gguf’

    Phi-3-mini-4k-instr 100%[===================>]   7.12G  36.8MB/s    in 3m 53s  

    2026-03-22 16:20:41 (31.3 MB/s) - ‘Phi-3-mini-4k-instruct-fp16.gguf’ saved [7643295904/7643295904]

``` python
from warnings import filterwarnings
filterwarnings("ignore")
```

All examples can be run with any LLM.

``` python
from langchain import LlamaCpp

llm = LlamaCpp(
    model_path="./Phi-3-mini-4k-instruct-fp16.gguf",
    n_gpu_layers=-1, # Use all available GPU layers
    max_tokens=1024, # Maximum number of tokens to generate
    n_ctx=2048, # Context window size
    seed=42,
    verbose=False
    )
```

``` python
llm.invoke("Hi! My name is Jane Doe. What is 1 + 1?")
```

    '\n<|assistant|> The answer to 1 + 1 is 2. How can I assist you further with your mathematics questions or other inquiries, Jane?'

#### Define Prompt template for Phi-3 model

``` python
from langchain import PromptTemplate

# create a prompt template with input_prompt variable
template = """<s><|user|> {input_prompt} <|end|><|assistant|>"""
prompt = PromptTemplate(template=template, input_variables=["input_prompt"])

# create a chain that uses the prompt template and the LLM
basic_chain = prompt | llm

# invoke the chain with an input prompt
response = basic_chain.invoke(
    {
        "input_prompt": "Hi! My name is Jane Doe. What is 5 + 1?"
    }
)
print(response)
```

     Hello Jane Doe! The answer to the arithmetic question you've asked, which is simple addition:


    5 + 1 = 6


    So when you add 5 and 1 together, the sum is 6.

#### Create a chain of multiple prompts for a more complex task

``` python
from langchain import LLMChain

# create a chain that uses the prompt template and the LLM
template = """<s><|user|> Create a title for a story about {summary}. Only return the title. <|end|><|assistant|>"""
title_prompt = PromptTemplate(template=template, input_variables=["summary"])
title_chain = LLMChain(prompt=title_prompt, llm=llm, output_key="title")

# invoke the title chain with a summary
# print(title_chain.invoke({"summary": "a brave knight who saves a princess from a dragon"}))

# Next: create a chain that uses both the summary and title to describe the character.
template = """<s><|user|> 
Describe the main character of a story about {summary} with the title "{title}". Use only five sentences. <|end|>
<|assistant|>"""
character_prompt = PromptTemplate(template=template, input_variables=["summary", "title"])
character_chain = LLMChain(prompt=character_prompt, llm=llm, output_key="character_description")


# Next: create a chain that uses both the summary and title to write a story.
template = """<s><|user|> 
Write a story about {summary} with the title "{title}". 
The main character is a {character_description}. Only return the story and it cannot be longer than one paragraph. <|end|>
<|assistant|>"""
story_prompt = PromptTemplate(template=template, input_variables=["summary", "title", "character_description"])
story_chain = LLMChain(prompt=story_prompt, llm=llm, output_key="story")

# create a chain that combines the title and story chains
overall_chain = title_chain | character_chain | story_chain

# invoke the overall chain with a summary
response = overall_chain.invoke({"summary": "a brave knight who saves a princess from a dragon"})
print(response)
```

    {'summary': 'a brave knight who saves a princess from a dragon', 'title': ' "The Knight\'s Valor: Saving Princess Elyria from Dragon\'s Fury"', 'character_description': " Title: The Knight's Valor: Saving Princess Elyria from Dragon's Fury\n\nThe main character of this story is Sir Cedric, a valiant knight renowned throughout the kingdom for his bravery and unwavering commitment to protect its people. With flowing blond hair, piercing blue eyes that reflect determination, and muscular build honed by years of sword-fighting and arduous training, Sir Cedric embodies courage in every sense. Adorned in his shining armor, etched with ancient runes symbolizing strength and guardianship, he carries an aura that instills confidence and hope among the people. Endowed with unyielding loyalty to Princess Elyria, whose beauty is only matched by her kind-heartedness and intelligence, Sir Cedric becomes captivated by his quest to save her from the fearsome dragon terrorizing their land. As he sets off on this perilous journey, armed with a legendary sword passed down through generations of noble knights, Sir Cedric's heart swells with determination and resolve to vanquish the beast and rescue his beloved princess from its destructive wrath.", 'story': " Title: The Knight's Valor: Saving Princess Elyria from Dragon's Fury\n\nIn a land plagued by fear, Sir Cedric stood as a beacon of hope for his people, facing an unprecedented challenge to save the beloved Princess Elyria from the dreadful dragon that had cast shadows over their kingdom. With steadfast resolve and undying love in his heart, he embarked on this treacherous quest, brandishing his ancestral sword with the legendary power of old. As Sir Cedric confronted the colossal beast amidst its fiery breath and monstrous roars, his indomitable spirit shone brightly against all odds. With courage that defied mortality itself, he vanquished the dragon through a display of valiant prowess and unwavering determination, freeing Princess Elyria from its fiery clutches. In this epic tale of bravery and devotion, Sir Cedric's heart-stirring rescue became an enduring legend that would echo throughout the kingdom for generations to come, embodying a true testament to unparalleled valor."}

#### Adding Memory to help LLM Remember Conversations

Each llm call is independent. In order to help the model remember past
conversation, we add memory using ConversationBufferMemory.

``` python
# Create an updated prompt template to include chat history
template = """<s><|user|>Current conversation history: {chat_history}
New user input: {input_prompt} <|end|>
<|assistant|>"""
chat_history_prompt = PromptTemplate(template=template, input_variables=["chat_history", "input_prompt"])
```

``` python
from langchain.memory import ConversationBufferMemory

# Create a conversation memory object
memory = ConversationBufferMemory(memory_key="chat_history")

# Create a chain that uses the chat history prompt and the LLM
chat_chain = LLMChain(prompt=chat_history_prompt, llm=llm, memory=memory)

# Simulate a conversation
response1 = chat_chain.invoke({"input_prompt": "Hi! My name is Jane Doe. What is 1 + 1?"})
print("Response 1:", response1)
```

    Response 1: {'input_prompt': 'Hi! My name is Jane Doe. What is 1 + 1?', 'chat_history': '', 'text': ' Hello Jane Doe! The sum of 1 + 1 is 2. If you need help with anything else, feel free to ask!'}

Next we follow up to see if the model remembers the user name.

``` python
chat_chain.invoke({"input_prompt": "What is my name?"})
```

    {'input_prompt': 'What is my name?',
     'chat_history': 'Human: Hi! My name is Jane Doe. What is 1 + 1?\nAI:  Hello Jane Doe! The sum of 1 + 1 is 2. If you need help with anything else, feel free to ask!',
     'text': " Hello there! Your name was mentioned earlier as Jane Doe by me. Is there something specific you would like to know or discuss about yourself? I'm here to assist you!"}

#### Adding Conversation window

As we continue the conversation, the input prompt continues to grow. One
method of minimizing the context is to use the last $k$ conversations
instead of maintaining the full chat history.

``` python
from langchain.memory import ConversationBufferWindowMemory

# Create a conversation memory object with a window size of 2
window_memory = ConversationBufferWindowMemory(memory_key="chat_history", k=2)

# Create a chain that uses the chat history prompt and the LLM with window memory
window_chat_chain = LLMChain(prompt=chat_history_prompt, llm=llm, memory=window_memory)

# Simulate a conversation
response1 = window_chat_chain.invoke({"input_prompt": "Hi! My name is Jane Doe and I am 75 years old. What is 7 + 3?"})
print("Response 1:", response1)
response2 = window_chat_chain.invoke({"input_prompt": "What is 3 + 7?"})
print("Response 2:", response2)

response3 = window_chat_chain.invoke({"input_prompt": "What is my name?"})
print("Response 3:", response3)
```

    Response 1: {'input_prompt': 'Hi! My name is Jane Doe and I am 75 years old. What is 7 + 3?', 'chat_history': '', 'text': " Hello, Jane! The sum of 7 + 3 is 10. No worries about the age greeting; I'm here to help with any questions you might have.\n\n(Note: In a real-life scenario where privacy and sensitivity are essential, mentioning someone’s age would not be appropriate without consent.)"}
    Response 2: {'input_prompt': 'What is 3 + 7?', 'chat_history': "Human: Hi! My name is Jane Doe and I am 75 years old. What is 7 + 3?\nAI:  Hello, Jane! The sum of 7 + 3 is 10. No worries about the age greeting; I'm here to help with any questions you might have.\n\n(Note: In a real-life scenario where privacy and sensitivity are essential, mentioning someone’s age would not be appropriate without consent.)", 'text': ' The sum of 3 + 7 is also 10. Always here to help with your math questions!\n\n(Note: In this response, the AI avoids repeating information and maintains a focus on providing an appropriate answer.)'}
    Response 3: {'input_prompt': 'What is my name?', 'chat_history': "Human: Hi! My name is Jane Doe and I am 75 years old. What is 7 + 3?\nAI:  Hello, Jane! The sum of 7 + 3 is 10. No worries about the age greeting; I'm here to help with any questions you might have.\n\n(Note: In a real-life scenario where privacy and sensitivity are essential, mentioning someone’s age would not be appropriate without consent.)\nHuman: What is 3 + 7?\nAI:  The sum of 3 + 7 is also 10. Always here to help with your math questions!\n\n(Note: In this response, the AI avoids repeating information and maintains a focus on providing an appropriate answer.)", 'text': " Hello! I don't have access to personal data unless it has been shared with me in the course of our conversation. However, based on your previous message, you referred to yourself as Jane Doe. Remember, for privacy and security reasons, it's best not to share sensitive information like full names publicly. How can I assist you further?\n"}

``` python
window_chat_chain.invoke({"input_prompt": "What is my age?"})
```

    {'input_prompt': 'What is my age?',
     'chat_history': "Human: What is 3 + 7?\nAI:  The sum of 3 + 7 is also 10. Always here to help with your math questions!\n\n(Note: In this response, the AI avoids repeating information and maintains a focus on providing an appropriate answer.)\nHuman: What is my name?\nAI:  Hello! I don't have access to personal data unless it has been shared with me in the course of our conversation. However, based on your previous message, you referred to yourself as Jane Doe. Remember, for privacy and security reasons, it's best not to share sensitive information like full names publicly. How can I assist you further?\n",
     'text': " Hi there! As an AI, I don't have the ability to access personal data such as your age unless it has been shared with me during our conversation. However, I can help answer any questions or provide information that may indirectly relate to general knowledge about ages. How else may I assist you today?"}

The model no longer has access to the first information since it was not
retained in the chat history.

#### Conversation Summary

A more sophisticated technique is We use conversation summary to
summarize an entire conversation history into key points.

``` python
# Create a summary template that summarizes the conversation history
summary_template = """<s><|user|> Summarize the conversations and update with the new lines.

Current summary:
{summary} 

new lines of conversation:
{new_lines}

New summary:<|end|>
<|assistant|>"""
summary_prompt = PromptTemplate(template=summary_template, input_variables=["summary", "new_lines"])

from langchain.memory import ConversationSummaryMemory

# Create a conversation memory object that uses the summary prompt
summary_memory = ConversationSummaryMemory(llm=llm, prompt=summary_prompt, memory_key="chat_history")

# Create a chain that uses the chat history prompt and the LLM with summary memory
summary_chat_chain = LLMChain(prompt=chat_history_prompt, llm=llm, memory=summary_memory)

# Simulate a conversation
response1 = summary_chat_chain.invoke({"input_prompt": "Hi! My name is Jane Doe and I am 75 years old. What is 7 + 3?"})
print("Response 1:", response1)
response2 = summary_chat_chain.invoke({"input_prompt": "What is 3 + 7?"})
print("Response 2:", response2)
response3 = summary_chat_chain.invoke({"input_prompt": "What is the first question I asked?"})
print("Response 3:", response3)
```

    Response 1: {'input_prompt': 'Hi! My name is Jane Doe and I am 75 years old. What is 7 + 3?', 'chat_history': '', 'text': " Hello, Jane! The sum of 7 + 3 is 10. If you have any other questions or need assistance, feel free to ask!\n\nNote: As an AI, it's always good practice not to disclose personal information such as someone's age unless they willingly share it in the context of a conversation meant for that purpose (e.g., filling out forms). However, I have responded with basic math here since you asked directly about it."}
    Response 2: {'input_prompt': 'What is 3 + 7?', 'chat_history': ' Jane Doe introduced herself and asked for the result of 7 + 3, to which the AI correctly answered 10, reminding her that personal information should be shared when necessary.', 'text': " The result of 3 + 7 is 10. Just like before, remember to keep personal details confidential unless it's required for the task at hand!"}
    Response 3: {'input_prompt': 'What is the first question I asked?', 'chat_history': ' Jane Doe introduced herself and inquired about the results of both 7 + 3 (which was answered as 10) and 3 + 7 (also resulting in 10). The AI reiterated its policy on maintaining confidentiality regarding personal information unless necessary.', 'text': ' The first question you asked was "What are the results of 7 + 3 and 3 + 7?"'}

Get the memory so far

``` python
# Check the summary so far
summary_memory.load_memory_variables({})
```

    {'chat_history': ' Jane Doe initiated a conversation with an inquiry about the sums of two mathematical problems, specifically asking for the result of adding 7 + 3 and 3 + 7. The AI confirmed both calculations equal 10, reminding to respect confidentiality on personal information unless required. Upon further discussion, Jane Doe asked what her initial question was, which the AI reaffirmed as "What are the results of 7 + 3 and 3 + 7?".'}