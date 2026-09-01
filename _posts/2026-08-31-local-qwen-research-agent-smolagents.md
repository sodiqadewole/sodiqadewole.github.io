---
title: "Building a Local Qwen2.5 Research Agent with SmolAgents"
date: 2026-08-31
permalink: /posts/2026/08/local-qwen-research-agent-smolagents/
blog_category: ai-agents
blog_section: Research Agents
blog_summary: "Build a local research agent with SmolAgents, Qwen2.5-0.5B-Instruct, and DuckDuckGo search, then test it on factual lookup, comparison, trend analysis, and investigative research tasks."
read_time: true
tags:
  - AI Agents
  - Research Agents
  - SmolAgents
  - Qwen
  - Local LLMs
---

Research agents are useful when the task is less about calculating an answer and more about searching, gathering, comparing, and synthesizing information. In this walkthrough, we build a local research agent with Hugging Face SmolAgents, `Qwen/Qwen2.5-0.5B-Instruct`, and a DuckDuckGo search tool.

The notebook behind this post is deliberately practical. It starts with a small local model, gives it web search, and then tries increasingly difficult research prompts. The interesting part is not only what works. The saved outputs also show where a compact local model struggles: repeated searches, tool-call formatting errors, stale answers, and long-running tasks that hit maximum steps.

## What Is a Research Agent?

A research agent is an AI system that can use information-retrieval tools to answer questions that require context beyond the model's internal weights. Instead of only generating from memory, it can:

- Search the web for relevant sources.
- Retrieve snippets and facts from multiple pages.
- Compare competing descriptions of a topic.
- Synthesize findings into a structured answer.
- Iterate when the first search is incomplete.
- Report results in a usable format.

That makes a research agent different from a code agent. A code agent primarily writes and executes code. A research agent primarily searches, reads, and summarizes.

| Aspect | Code Agent | Research Agent |
|--------|------------|----------------|
| Primary tool | Code execution | Web search and retrieval |
| Best use case | Computation, automation, data processing | Fact finding, literature scans, trend reports |
| Output style | Executed results | Synthesized information |
| Main risk | Incorrect code or runtime errors | Weak sources, stale facts, hallucinated synthesis |

## Setup

The notebook uses a local Transformers model through SmolAgents.

```python
import warnings

warnings.filterwarnings("ignore")
```

```python
import pprint

pp = pprint.PrettyPrinter()

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

from smolagents import DuckDuckGoSearchTool, ToolCallingAgent, TransformersModel

model = TransformersModel(
    model_id,
    temperature=0.7,
    max_new_tokens=1000,
)

agent = ToolCallingAgent(
    model=model,
    tools=[DuckDuckGoSearchTool()],
)
```

The important shift from the code-agent setup is `ToolCallingAgent` plus `DuckDuckGoSearchTool`. The agent now has a search tool available, so it can ground answers in retrieved snippets.

## Baseline Test: Capital Lookup

The first task checks whether search is wired correctly.

```python
task = "What is the capital of the Philippines"
result = agent.run(task)
print(result)
```

The saved notebook output shows the agent repeatedly calling `web_search` for "Capital of the Philippines" and retrieving results from sources such as Wikipedia, Britannica, Mappr, and WorldAtlas. The final answer was:

Observed intermediate trace:

```text
New run
Task: What is the capital of the Philippines

Step 1
Calling tool: 'web_search' with arguments: {'query': 'Capital of the Philippines'}

Observations:
- Capital of the Philippines - Wikipedia
   The current capital city, Manila, has been the country's capital throughout most of its history...
- Manila - Wikipedia
   Manila, officially the City of Manila, is the capital and second-most populous city...
- Britannica
   Manila, capital and chief city of the Philippines...

Step 2
Error while parsing tool call from model output:
The JSON blob you used is invalid due to: Extra data.

Steps 5-20
The agent repeatedly called `web_search` with the same query.
Reached max steps.
```

Observed final result:

```text
According to the information provided, the capital of the Philippines is Manila.
```

That is the happy path: simple query, clear source agreement, short final answer. It also exposed a small-model behavior worth watching. The agent repeated the same search many times and produced some malformed tool-call output before finishing. For a local 0.5B model, tool-call formatting is often the fragile part of the loop.

## Example 1: Simple Fact Lookup

The first full example asks for basic facts about Python.

```python
research_task1 = """
Search for and provide the following information about Python programming language:
1. When was Python first released?
2. Who created Python?
3. What is the current stable version of Python (as of 2024)?

Provide clear, accurate answers for each question.
"""

print("=" * 60)
print("RESEARCH TASK 1: Simple Fact Lookup")
print("=" * 60)
result1 = agent.run(research_task1)
print(result1)
```

The saved run included repeated tool-call parsing failures before returning a final answer:

Observed intermediate trace:

```text
New run
Task: Search for Python release date, creator, and current stable version as of 2024.

Step 1
Error while parsing tool call from model output:
The model output does not contain any JSON blob.

Steps 2-20
The same parsing error repeated.
Reached max steps.
```

Observed final result:

```text
1. Python was first released in 1991.
2. Guido van Rossum created Python.
3. As of 2024, the current stable version of Python is Python 3.11.
```

The first two answers are the kind of direct facts a search agent usually handles well. The third answer is the one to verify carefully. Version facts change over time, and even within a target year, a model may give stale or incomplete information. This is why research-agent outputs need a source and date check before they are treated as final.

## Example 2: Comparative Research

The second task asks the agent to compare machine learning and deep learning.

```python
research_task2 = """
Compare Machine Learning and Deep Learning by researching and providing:

1. Definitions: What is the core difference between ML and DL?
2. Key Algorithms: List 3 main algorithms/approaches for each
3. Use Cases: Provide 2-3 real-world applications for each
4. Computational Requirements: How do computational needs differ?
5. Learning Curve: Which is easier for beginners to start with and why?

Create a structured comparison that highlights the pros and cons of each approach.
"""

print("=" * 60)
print("RESEARCH TASK 2: Comparative Research")
print("=" * 60)
result2 = agent.run(research_task2)
print(result2)
```

Observed intermediate trace:

```text
New run
Task: Compare Machine Learning and Deep Learning.

Step 1
Calling tool: 'web_search' with arguments: {'query': 'Machine Learning vs. Deep Learning'}

Observations:
- GeeksforGeeks: Difference Between Machine Learning and Deep Learning
- Coursera: Deep Learning vs. Machine Learning
- IBM: AI vs. machine learning vs. deep learning vs. neural networks
- Databricks: Machine Learning vs Deep Learning

Step 2
Error while parsing tool call from model output:
The model output does not contain any JSON blob.

Steps 3-20
The parsing error repeated until the agent reached max steps.
```

Observed final result excerpt:

```text
Certainly! Below is a structured comparison of Machine Learning (ML) and Deep Learning (DL)...

Definitions:
1. Machine Learning (ML): The study of computer systems that learn and adapt automatically from experience...
2. Deep Learning (DL): An extension of machine learning that focuses on training neural networks with multiple layers...

Core Differences:
- Training Process: ML models are trained using supervised learning... DL models are trained using unsupervised learning...
- Model Complexity: ML models are often simpler... DL models are more complex...
- Interpretability: ML models tend to be more interpretable...

Summary:
- ML: Generally considered simpler and less computationally intensive, suitable for smaller datasets and simpler tasks.
- DL: Offers superior performance and efficiency, ideal for complex tasks involving large amounts of data...
```

This is a better use case for a research agent than a one-line factual query. The output should be judged on structure: definitions, examples, algorithms, computational tradeoffs, and beginner guidance. For comparative prompts, I prefer asking for an explicit table or schema so the agent does not bury the useful comparison in paragraphs.

For example:

```text
Return the answer as a table with columns:
Dimension, Machine Learning, Deep Learning, Practical implication.
```

That makes it easier to spot missing dimensions or vague answers.

## Example 3: Trend Analysis

The third task asks for a 2023-2024 AI trend report.

```python
research_task3 = """
Research the latest trends in Artificial Intelligence (2023-2024) and provide:

1. Major Breakthroughs: What are the 3-4 most significant AI breakthroughs in recent years?
2. Emerging Applications: What new real-world applications of AI have emerged recently?
3. Key Players: Who are the leading organizations/companies driving AI innovation?
4. Challenges: What are the main challenges the AI industry is facing?
5. Future Direction: Based on current trends, what direction is the AI field heading?

Synthesize this information into a comprehensive trend report highlighting the evolution of AI technology.
"""

print("=" * 60)
print("RESEARCH TASK 3: Trend Analysis & Synthesis")
print("=" * 60)
result3 = agent.run(research_task3)
pp.pprint(result3)
```

The saved output shows the agent finding plausible sources, including the Stanford AI Index and AI-trend articles.

Observed intermediate trace:

```text
New run
Task: Research latest trends in Artificial Intelligence (2023-2024).

Step 1
Calling tool: 'web_search' with arguments: {'query': 'Latest Trends in Artificial Intelligence (2023-2024)'}

Observations:
- Stanford HAI: The 2024 AI Index Report
- Forbes: Top AI Developments of 2023 and what to expect in 2024
- LinkedIn: Six key AI trends to watch in 2023-2024
- HealthIT.gov: Hospital trends in predictive AI use, evaluation, and governance

Step 2
Calling tool: 'web_search' with the same trend query.

Step 3 onward
Error while parsing tool call from model output:
The model output does not contain any JSON blob.
```

Observed final result:

```text
No clean final research report was saved for this run.
The cell later failed while trying to print the result.
```

It also shows a notebook-level issue: a later cell attempted `pprint.pprint(result3)` and hit this error:

```text
AttributeError: 'function' object has no attribute 'pprint'
```

The fix is simple: use the `pp` object created earlier, or import the module without shadowing it.

```python
pp.pprint(result3)
```

The more important research lesson is that trend analysis raises the bar. A good answer should distinguish between technical breakthroughs, product adoption, regulation, safety concerns, and market behavior. If those categories blur together, the report may sound polished while still being shallow.

## Example 4: Multi-Source Critical Analysis

The fourth task asks for a balanced report on remote work and productivity.

```python
research_task4 = """
Research the impact of Remote Work on productivity and provide a detailed analytical report:

1. Scientific Evidence: What do recent studies say about remote work's impact on productivity?
2. Industry Perspectives: How do tech companies, management consultancies, and employees view remote work?
3. Challenges: What are documented challenges and downsides of remote work?
4. Benefits: What are the proven benefits beyond just productivity?
5. Best Practices: What organizations and practices are considered most effective for remote work?
6. Future Outlook: How is the future of work evolving?

Analyze the conflicting viewpoints, synthesize findings from multiple perspectives, and provide
a balanced conclusion about the true impact of remote work on productivity and employee well-being.
"""

print("=" * 60)
print("RESEARCH TASK 4: Multi-Source Research with Critical Analysis")
print("=" * 60)
result4 = agent.run(research_task4)
pp.pprint(result4)
```

Observed intermediate trace:

```text
New run
Task: Research the impact of remote work on productivity.

Step 1
Error while parsing tool call from model output:
The model output does not contain any JSON blob.

Steps 2-20
The same parsing error repeated.
Reached max steps.
```

Observed final result excerpt:

```text
### Report on the Impact of Remote Work on Productivity and Employee Well-Being

1. Scientific Evidence
Recent studies have consistently shown that remote work has the potential to significantly improve productivity...

2. Industry Perspectives
Tech companies... have embraced remote work as a way to attract and retain top talent...
Management consulting firms often leverage remote work to optimize project timelines and budgets...
Employees find remote work appealing due to its ability to cater to diverse work styles...

3. Challenges
One major challenge is maintaining physical distance and ensuring privacy...
Another challenge is the potential for social isolation and lack of face-to-face interaction...

6. Future Outlook
The future of work is likely to continue to evolve, with more companies embracing remote work...
```

This is where a research agent should be pushed to handle disagreement. Remote work research contains mixed claims because productivity depends on job type, management style, collaboration needs, commuting costs, home environment, and measurement method.

For this kind of prompt, I would ask the agent to separate evidence types:

```text
Group findings by evidence type:
- peer-reviewed studies
- company surveys
- employee surveys
- management commentary
- caveats and confounders
```

That forces the agent to keep "what studies found" separate from "what companies prefer" and "what employees report."

## Example 5: Investigative Research

The final task is intentionally broad: AI's role in addressing climate change.

```python
research_task5 = """
Conduct a comprehensive investigative research on "The Role of AI in Addressing Climate Change":

1. Problem Context: What is the current state of climate change and its urgency?

2. AI Applications: Research and identify:
   - How AI is currently being used in climate research and modeling
   - AI applications in renewable energy optimization
   - AI for carbon tracking and emissions monitoring
   - Predictive AI for climate forecasting
   - AI-powered solutions for climate adaptation

3. Impact Assessment:
   - What measurable impact has AI had so far in climate mitigation?
   - What are the limitations and challenges?
   - Where is AI most effective vs. ineffective?

4. Emerging Solutions: What new AI-powered climate solutions are in development?

5. Barriers & Opportunities:
   - What are the major barriers to scaling AI climate solutions?
   - What opportunities exist for acceleration?
   - What policy changes are needed?

6. Expert Opinions: What do climate scientists, AI researchers, and environmental organizations say?

7. Critical Synthesis:
   - Can AI alone solve climate change? Why or why not?
   - What is the realistic role of AI in the climate solution?
   - What other factors are equally or more important?

Provide a nuanced, well-researched analysis that considers multiple perspectives, acknowledges
limitations, and provides actionable insights for policymakers and technologists.
"""

print("=" * 60)
print("RESEARCH TASK 5: Complex Investigative Research")
print("=" * 60)
result5 = agent.run(research_task5)
pp.pprint(result5)
```

Observed intermediate trace:

```text
New run
Task: Conduct investigative research on the role of AI in addressing climate change.

Step 1
Calling tool: 'web_search' with arguments: {'query': 'Current state of climate change and its urgency'}

Observations:
- UNEP: The world is in a state of climate emergency; average world temperature has risen between 1.1 and 1.2 C.
- NOAA: Climate change affects rising temperatures, sea level rise, drought, flooding, and more.
- WHO: Climate change is expected to cause additional deaths from undernutrition, malaria, and other causes.
- United Nations: Global temperatures are likely to exceed the 1.5 C threshold without emissions decline.

Step 2
Calling tool: 'web_search' with arguments: {'query': 'AI applications in renewable energy optimization'}

Observations:
- Department of Energy: AI can support renewable forecasting, grid modeling, resilience, and permitting workflows.
- IEA: AI can help unlock additional transmission capacity and improve energy optimization.
- Omdena: AI applications include predictive maintenance, energy optimization, and smart-grid management.
- SAGE review: AI supports resource assessment, forecasting, monitoring, control strategies, and grid integration.

Step 3 onward
Error while parsing tool call from model output:
The JSON blob was invalid due to extra data.
The agent repeatedly tried to call the same renewable-energy search.
Reached max steps.
```

Observed final result excerpt:

```text
### Final Answer:
The role of AI in addressing climate change involves several areas where AI is playing a significant role, including:

1. Energy Forecasting and Optimization
2. Renewable Energy Optimization
3. Carbon Tracking and Emissions Monitoring
4. Predictive Maintenance
5. Grid Management
6. Energy Storage Solutions
7. Smart Grid Integration
8. Adaptation to Changing Climate
9. Policy and Regulatory Impact
10. Customer Engagement and Education

These AI applications aim to make energy systems more efficient, reliable, and sustainable, contributing significantly to addressing climate change.
```

The notebook also saved a longer, truncated final stdout. It began with a broader report covering climate urgency, energy forecasting, renewable-energy optimization, carbon tracking, predictive maintenance, grid management, impact assessment, limitations, emerging solutions, barriers, and policy needs, but it cut off during the policy section:

```text
### Policy Changes Needed

To fully leverage the potential of AI in addressing climate change, policymakers must
```

This is probably too broad for a small local model and a single run. A stronger workflow is to split it into subquestions and cache intermediate notes:

1. Search for climate modeling and forecasting uses.
2. Search for grid optimization and renewable forecasting uses.
3. Search for emissions measurement and carbon accounting tools.
4. Search for limitations, energy use, and governance risks.
5. Ask the agent to synthesize only after the evidence map exists.

That turns one fragile mega-prompt into a sequence of checkable research steps.

## Complexity Progression

The notebook frames the tasks as a difficulty ladder:

| Task | Difficulty | What It Tests |
|------|------------|---------------|
| Capital lookup | Easy | Search-tool wiring and direct factual lookup |
| Python facts | Easy | Basic fact retrieval and freshness checking |
| ML vs. DL comparison | Medium | Organization across multiple dimensions |
| AI trend analysis | Medium-hard | Recent information and synthesis |
| Remote work analysis | Hard | Conflicting evidence and balanced conclusions |
| AI and climate change | Very hard | Systems thinking and multi-source investigation |

This progression is useful because it shows where the agent begins to degrade. Simple lookups can work. Comparative and analytical tasks need stronger output structure. Investigative tasks need decomposition.

## How to Judge Research-Agent Output

The notebook includes a practical evaluation rubric. I would summarize it as five questions:

1. Is the answer organized into clear sections?
2. Does it show source awareness or mention where claims came from?
3. Does the depth match the complexity of the question?
4. Are dates, numbers, and version-sensitive facts verified?
5. Does it acknowledge uncertainty, caveats, and conflicting evidence?

Good research-agent output usually has specific dates, statistics, source-aware phrasing, and tradeoffs. Weak output often contains generic claims, confident language without sources, missing counterarguments, or contradictions inside the same answer.

## Prompting Tips That Help

For a local research agent, the prompt should constrain the shape of the work.

```text
Search for recent sources first.
Separate facts from interpretation.
Mention the source or source type for each major claim.
Flag any uncertainty or disagreement across sources.
Return the answer in the requested structure.
Do not repeat the same search if the first search already answered the question.
```

For complex topics, ask for intermediate artifacts instead of one final report:

```text
Step 1: List 5 source candidates with one-sentence relevance notes.
Step 2: Extract key claims from each source.
Step 3: Identify agreements and disagreements.
Step 4: Write the final synthesis.
```

That gives you places to inspect and correct the run before the final answer hardens around a weak source or a mistaken interpretation.

## What Worked

The agent successfully demonstrated search-tool usage. In the capital-city baseline, it retrieved multiple sources and arrived at the correct answer. The notebook also provides a clear pattern for scaling from simple fact lookup to more complex research prompts.

The setup is especially useful as a local teaching environment. You can see the agent loop, inspect search observations, and watch how task complexity changes the behavior of the system.

## What Did Not Work Reliably

The saved runs also show the limits of a small local model in a tool-calling setup:

- It repeated identical searches many times.
- Some tool-call outputs failed JSON parsing.
- One factual answer looked stale and needed verification.
- A later notebook cell hit a `pprint` usage error.
- Complex tasks reached maximum steps or became noisy.

These are not failures of the research-agent idea. They are engineering signals. Tool-calling models need reliable formatting, bounded tasks, explicit stop conditions, and validation.

## Takeaway

A local Qwen2.5-0.5B research agent with SmolAgents can demonstrate the core research loop: search, observe, synthesize, and report. It is strongest on simple, bounded research tasks and as a learning tool for understanding agent behavior.

For serious research, use it with structure. Break large questions into smaller searches, require source-aware claims, verify version-sensitive facts, and treat the final answer as a draft to audit rather than a finished truth.