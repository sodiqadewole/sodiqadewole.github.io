---
title: "Building a Local Qwen2.5 Code Agent with SmolAgents"
date: 2026-08-31
permalink: /posts/2026/08/local-qwen-code-agent-smolagents/
blog_category: ai-agents
blog_section: Code Agents
blog_summary: "Build a local code-execution agent with SmolAgents and Qwen2.5-0.5B-Instruct, then test it on arithmetic, data processing, text analysis, and algorithmic tasks."
read_time: true
tags:
  - AI Agents
  - Code Agents
  - SmolAgents
  - Qwen
  - Local LLMs
---

Running a code agent locally is a useful way to see where agentic workflows help, and where they still need guardrails. In this walkthrough, we build a small local code agent with Hugging Face SmolAgents and `Qwen/Qwen2.5-0.5B-Instruct`, then try it on arithmetic, structured data, text analysis, prime-number generation, and a small sales-report task.

The goal is not to pretend a 0.5B model behaves like a large frontier model. The useful lesson is more practical: a local model can execute simple Python reasoning, but prompt clarity, sandbox limits, and output validation matter a lot.

## What We Are Building

The setup combines three pieces:

- `Qwen/Qwen2.5-0.5B-Instruct`: a compact instruction-tuned language model that can run locally.
- `TransformersModel`: the SmolAgents wrapper that loads the model from Hugging Face Transformers.
- `CodeAgent`: an agent that writes and executes Python code in a controlled environment.

This gives us a local agent loop:

1. Ask the model to solve a task.
2. Let the agent generate Python code.
3. Execute the generated code.
4. Return the result.

## Setup

The notebook starts by suppressing warnings so the output stays readable.

```python
import warnings

warnings.filterwarnings("ignore")
```

Then we create a `CodeAgent` backed by Qwen2.5-0.5B-Instruct.

```python
from smolagents import CodeAgent, TransformersModel

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

agent = CodeAgent(
    model=TransformersModel(
        model_id,
        temperature=0.7,
        max_new_tokens=1000,
    ),
    tools=[],
)
```

The `tools=[]` argument is important. This agent is not browsing the web or calling external APIs. It is working through generated Python code inside the execution environment that SmolAgents provides.

## Example 1: Arithmetic

The first task is deliberately simple.

```python
result = agent.run("Calculate the sum of numbers from 1 to 100")
print(result)
```

The agent generated the direct Python solution:

```python
total_sum = sum(range(1, 101))
final_answer(total_sum)
```

The output was:

```text
5050
```

This is the best-case path for a local code agent. The task is unambiguous, the code is short, and the result is easy to verify.

## Example 2: Structured Data

Next, the notebook asks the agent to work with a list of student records.

```python
task = """
Given a list of students with their scores:
students = [
    {'name': 'Alice', 'score': 85},
    {'name': 'Bob', 'score': 92},
    {'name': 'Charlie', 'score': 78},
    {'name': 'Diana', 'score': 95},
]

Find the top 2 students by score and return their names and scores.
"""

result = agent.run(task)
print("Task 2 Result:")
print(result)
```

The right computation is simple Python:

```python
students = [
    {"name": "Alice", "score": 85},
    {"name": "Bob", "score": 92},
    {"name": "Charlie", "score": 78},
    {"name": "Diana", "score": 95},
]

top_students = sorted(students, key=lambda student: student["score"], reverse=True)[:2]
print(top_students)
```

Expected result:

```text
[{'name': 'Diana', 'score': 95}, {'name': 'Bob', 'score': 92}]
```

In the notebook run, however, the agent tried to sort a `students` variable that was not defined inside the executed code block. That produced a useful failure mode:

```text
InterpreterError: The variable `students` is not defined.
```

This is a common local-agent issue. The model read the data in the prompt, but the generated executable code did not recreate that data before using it. A stronger prompt would explicitly say: "Define the `students` list inside the code before sorting it."

## Example 3: Text Analysis

The next task asks for a small text-analysis report.

```python
task = """
Analyze this text and provide:
1. Total word count
2. Number of unique words
3. Average word length
4. Most common word, excluding articles like 'the', 'a', 'an'

Text: "The quick brown fox jumps over the lazy dog. The quick fox is clever and nimble.
      The dog is lazy but loyal. Fox and dog are common in stories."
"""

result = agent.run(task)
print("Task 3 Result:")
print(result)
```

A more reliable implementation is to normalize case, remove punctuation, filter articles, and then count words.

```python
from collections import Counter
import re

text = """
The quick brown fox jumps over the lazy dog. The quick fox is clever and nimble.
The dog is lazy but loyal. Fox and dog are common in stories.
"""

words = re.findall(r"\b\w+\b", text.lower())
filtered_words = [word for word in words if word not in {"the", "a", "an"}]

total_word_count = len(words)
unique_word_count = len(set(words))
average_word_length = sum(len(word) for word in words) / total_word_count
most_common_word = Counter(filtered_words).most_common(1)[0]

print(total_word_count)
print(unique_word_count)
print(round(average_word_length, 2))
print(most_common_word)
```

This example shows why validation matters. In the notebook output, the agent produced a runnable fragment, but it counted unique words where total words were requested and returned a noisy most-common-word list rather than one most-common token. For language tasks that look easy, the code can still be subtly wrong.

## Example 4: Prime Numbers

The fourth task asks the agent to generate primes up to 50, then compute summary statistics.

```python
task = """
Write a function to find all prime numbers up to 50.
Return them as a list and also calculate:
1. How many primes are there?
2. What is the sum of all primes?
3. What is the average of all primes?
"""

result = agent.run(task)
print("Task 4 Result:")
print(result)
```

A clean implementation is:

```python
def is_prime(number):
    if number < 2:
        return False

    for divisor in range(2, int(number**0.5) + 1):
        if number % divisor == 0:
            return False

    return True


primes = [number for number in range(2, 51) if is_prime(number)]

print(primes)
print(len(primes))
print(sum(primes))
print(sum(primes) / len(primes))
```

Expected result:

```text
[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
15
328
21.866666666666667
```

The notebook run found the correct prime list, count, and sum, but computed the average incorrectly as `count / len(primes)`, which is always `1.0` when `count == len(primes)`. This is a good reminder: a code agent can produce convincing code and still make a basic semantic mistake.

## Example 5: Sales Data Summary

The last task asks for a compact data pipeline.

```python
task = """
Process this dataset and create a summary report:

sales_data = [
    {'month': 'Jan', 'product': 'A', 'quantity': 100, 'price': 50},
    {'month': 'Jan', 'product': 'B', 'quantity': 150, 'price': 30},
    {'month': 'Feb', 'product': 'A', 'quantity': 120, 'price': 50},
    {'month': 'Feb', 'product': 'B', 'quantity': 100, 'price': 30},
    {'month': 'Mar', 'product': 'A', 'quantity': 200, 'price': 50},
    {'month': 'Mar', 'product': 'B', 'quantity': 180, 'price': 30},
]

Create a report with:
1. Total revenue by product
2. Total revenue by month
3. Best performing product by revenue
4. Which month had the highest revenue
"""

result = agent.run(task)
print("Task 5 Result:")
print(result)
```

The direct Python version is:

```python
from collections import defaultdict

sales_data = [
    {"month": "Jan", "product": "A", "quantity": 100, "price": 50},
    {"month": "Jan", "product": "B", "quantity": 150, "price": 30},
    {"month": "Feb", "product": "A", "quantity": 120, "price": 50},
    {"month": "Feb", "product": "B", "quantity": 100, "price": 30},
    {"month": "Mar", "product": "A", "quantity": 200, "price": 50},
    {"month": "Mar", "product": "B", "quantity": 180, "price": 30},
]

revenue_by_product = defaultdict(int)
revenue_by_month = defaultdict(int)

for row in sales_data:
    revenue = row["quantity"] * row["price"]
    revenue_by_product[row["product"]] += revenue
    revenue_by_month[row["month"]] += revenue

best_product = max(revenue_by_product.items(), key=lambda item: item[1])
best_month = max(revenue_by_month.items(), key=lambda item: item[1])

print(dict(revenue_by_product))
print(dict(revenue_by_month))
print(best_product)
print(best_month)
```

Expected result:

```text
{'A': 21000, 'B': 12900}
{'Jan': 9500, 'Feb': 9000, 'Mar': 15400}
('A', 21000)
('Mar', 15400)
```

The notebook output correctly identified product A and March as winners, but it also attempted an unauthorized `web_search` import and produced incorrect monthly revenue totals in part of the run. That tells us two things:

1. Agent sandboxes matter. If a tool or import is not allowed, the model may still try to call it unless the prompt is explicit.
2. Numeric reports should be recomputed and checked independently before being trusted.

## Practical Prompting Tips

For small local code agents, I get better results when the task includes explicit execution constraints:

```text
Solve this using only Python standard-library code.
Define all input data inside the code block before using it.
Do not import packages that are not needed.
Print the intermediate values and final answer.
Before returning, verify the result with a simple independent check.
```

For data tasks, it also helps to ask for a specific output schema:

```text
Return a dictionary with these keys:
- revenue_by_product
- revenue_by_month
- best_product
- best_month
```

That reduces the chance that the agent mixes explanation, partial calculations, and final results into one hard-to-parse answer.

## What Worked

The local Qwen2.5 code agent handled simple arithmetic cleanly. It also often generated plausible Python for list sorting, text processing, prime-number generation, and dictionary aggregation.

The biggest strength is speed of iteration. You can ask a local model to sketch code, run it, inspect failures, and tighten the prompt without sending data to an external hosted model.

## What Did Not Work Reliably

The notebook also showed several limitations:

- The agent sometimes referenced variables from the prompt without defining them in executable code.
- It sometimes returned text in a format the CodeAgent parser could not execute.
- It occasionally produced correct-looking code with incorrect calculations.
- It attempted unauthorized imports such as `web_search` even though no such tool was configured.
- Longer runs accumulated repeated parsing failures, which made later outputs less reliable.

These are not reasons to avoid local code agents. They are reasons to treat them as code-generation assistants inside a validation loop, not as autonomous authorities.

## When to Use This Setup

This local setup is a good fit for:

- Small programming exercises.
- Quick data transformations.
- Private experiments where you want local execution.
- Teaching agent behavior and failure modes.
- Prototyping prompts before moving to a larger model.

It is less suitable for:

- Production automation without tests.
- Long multi-file software changes.
- Tasks that require external tools unless those tools are explicitly registered.
- Situations where numerical accuracy is critical and unverified.

## Takeaway

A local Qwen2.5-0.5B SmolAgents code agent is capable enough to demonstrate the agent loop: generate code, execute it, observe the result, and iterate. The important engineering habit is to keep the loop grounded. Define inputs inside the generated code, constrain imports, request structured outputs, and verify the final answer.

That turns a small local model from a fragile demo into a useful coding partner for bounded, checkable tasks.