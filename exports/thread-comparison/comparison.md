# RAG graph thread comparison

Source: `http://127.0.0.1:2024`

## Totals (all threads)

| Graph | Threads | Idle | Error | Interrupted | Mean utility | Mean last run (s) | Mean relevant docs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Self-RAG | 8 | 3 | 4 | 1 | 2.33 | 12.3 | 1.38 |
| Query decomposition | 5 | 5 | 0 | 0 | 4.2 | 38.1 | 35.2 |
| HyDE | 5 | 5 | 0 | 0 | 3.6 | 35.0 | 4.0 |

## Totals (idle / completed only)

| Graph | Threads | Mean utility | Mean last run (s) | Mean relevant docs |
|---|---:|---:|---:|---:|
| Self-RAG | 3 | 2.67 | 21.0 | 3.67 |
| Query decomposition | 5 | 4.2 | 38.1 | 35.2 |
| HyDE | 5 | 3.6 | 35.0 | 4.0 |

## Shared questions

### what are the best frameworks for building AI agents?

**Self-RAG** (`01a03545…`, utility=3, relevant=4/5, 30.5s)

Top choices are modular, componentized agent frameworks (examples called out include LangGraph and Mastra) that let you compose Skills, tools, and prompts into agents [1][2]. Use ReACT-style agent patterns together with an evaluation-driven development loop—context engineering, prompt iteration, offline evaluation, and staged production—to iterate prompts and architecture reliably [1]. Vendor guides also recommend skill-based single- or multi-agent designs with strong observability and tool integration as best-practice patterns [2][3][4].

[1] support-agent.pdf, page 2
[2] claude-guide.pdf, page 10
[3] claude-guide.pdf, page 0
[4] openai-guide.pdf, page 2

Sources: support-agent.pdf, claude-guide.pdf, openai-guide.pdf

**Query decomposition** (`01a03546…`, utility=3, relevant=6/10, 55.8s)

Well-regarded options include LangGraph and Mastra for building componentized/modular agent graphs, and the ReAct pattern for LLM-driven reasoning and tool use [1][2]. Adopt componentized architectures with Agent Skills, modular prompts, and workflows (single-agent or multi-agent) to enable composability and easy tool integration [2][1]. Use an evaluation-driven development loop (offline evaluation, prompt/versioning, and production A/B tests), plus observability and guardrails for compliance and safe deployment [1][2][3].

[1] support-agent.pdf, page 2
[2] claude-guide.pdf, page 10
[3] openai-guide.pdf, page 2

Sub-queries: what are the best frameworks for building AI agents?

Sources: support-agent.pdf, claude-guide.pdf, openai-guide.pdf

**HyDE** (`01a03547…`, utility=2, relevant=2/5, 51.2s)

Popular frameworks include LangGraph and Mastra [1], and OpenAI’s Agents SDK is another practical option for implementing agents [2].  
A componentized agent architecture with modular "Agent Skills" makes it easy to add tools and roll out new prompting or capabilities without system‑wide refactoring [1].  
Design choices should reflect the agent pattern you need (single‑agent perceive–decide–act loops or multi‑agent specialization) [1] and the core components of model, tools, and instructions [2].

[1] claude-guide.pdf, page 10  
[2] openai-guide.pdf, page 6

Sources: claude-guide.pdf, openai-guide.pdf

### what are the best ways of building effective AI agents considering the following components (architecture and design, technical implementations) and give real world examples

**Self-RAG** (`01a0353a…`, utility=4, relevant=5/5, 15.3s)

To build effective AI agents, start with a simple architecture, focusing on single-purpose agents that can be scaled as requirements evolve. For example, a customer support agent can handle routine queries effectively before transitioning to a more complex multi-agent system for broader tasks like financial analysis [1]. Additionally, employing modular design principles allows for easy integration of new capabilities and tools, as seen in organizations that utilize agent skills for specialized workflows [1]. 

Sources:
[1] claude-guide.pdf, page 26

Sources: claude-guide.pdf, openai-guide.pdf

**Query decomposition** (`01a0353e…`, utility=4, relevant=40/40, 27.1s)

To build effective AI agents, start with single-purpose agents that can perform specific tasks well, then scale them as needed. Choose the right model based on the task's complexity, balancing capabilities, speed, and cost; for example, a customer support agent may use a lighter model for routine queries, while a financial analysis agent requires a more capable model for complex tasks. Real-world examples include Intercom's Fin AI agent, which achieves high resolution rates in customer support, and Coinbase's agentic systems that handle thousands of messages per hour with high availability [1]. 

Sources:
[1] assistant/docs/claude-guide.pdf, page 6

Sub-queries: best practices for AI agent architecture and design; technical implementations for building AI agents; real world examples of effective AI agents; challenges in building effective AI agents

Sources: claude-guide.pdf, support-agent.pdf, openai-guide.pdf

**HyDE** (`01a0353e…`, utility=4, relevant=5/5, 18.7s)

To build effective AI agents, start with single-purpose agents that can perform specific tasks well, allowing for clear metrics and easier debugging. For example, a customer service agent can handle routine inquiries before evolving into a multi-agent system that manages more complex interactions across different domains. Additionally, employing modular design and integrating specialized skills can enhance capabilities without over-complicating the architecture, as seen in e-commerce platforms that progressively add specialized agents for various customer service functions [1]. 

Sources:
[1] assistant/docs/claude-guide.pdf, page 26

Sources: claude-guide.pdf

**HyDE** (`01a0353e…`, utility=4, relevant=4/5, 38.4s)

To build effective AI agents, start with single-purpose agents that can prove ROI before scaling to more complex systems. For example, an e-commerce platform might begin with a single agent for customer inquiries, then evolve to specialized agents for different query types as needs grow [1]. Additionally, employing modular design and integrating specialized skills can enhance capabilities without over-engineering, allowing for easier updates and maintenance [1]. 

Sources:
[1] assistant/docs/claude-guide.pdf, page 26

Sources: claude-guide.pdf

### What's the best platform to deploy your ai agents?

**Query decomposition** (`01a03538…`, utility=4, relevant=68/120, 61.0s)

The best platform to deploy AI agents is the Claude Developer Platform, which provides access to models, tools, and technical documentation for AI agent implementation. It supports both single-agent systems and multi-agent architectures, making it suitable for various development needs [1]. Additionally, it offers resources for prompt engineering and specialized knowledge integration for agents [1]. 

Sources:
[1] claude-guide.pdf, page 28

Sub-queries: best platforms for deploying AI agents; features of popular AI deployment platforms; cost comparison of AI deployment platforms; user reviews of AI deployment platforms

Sources: claude-guide.pdf, openai-guide.pdf, support-agent.pdf

**HyDE** (`01a03533…`, utility=4, relevant=2/5, 18.6s)

The best platform to deploy your AI agents is the Claude Developer Platform, which provides access to models, tools, and technical documentation necessary for AI agent implementation. It supports both single-agent and multi-agent architectures, allowing for modular integration of specialized skills and workflows [1]. Additionally, it offers resources for building observable systems and advanced architectures [1]. 

Sources:
[1] assistant/docs/claude-guide.pdf, page 28

Sources: claude-guide.pdf


## All threads

### Self-RAG

- `01a03545-f376-78d0-b984-5ef82e67228c` · idle · utility=3 · what are the best frameworks for building AI agents?
- `01a0353a-ce1c-7bf0-9f32-efcd3864c39a` · idle · utility=4 · what are the best ways of building effective AI agents considering the following components (architecture and design, te
- `01a02cd7-eb78-7c12-a9f2-fdfa19c7266c` · idle · utility=1 · who is the president of united states
- `01a02cd0-08d2-73d1-ab0e-6384e24fce34` · interrupted · utility=4 · how do I get started with AI agents?
- `01a02cce-8506-7933-8d1b-716bbbf56f8e` · error · utility=None · (no question)
- `01a02ccd-4a1b-7b70-b2ba-16db5be08310` · error · utility=None · (no question)
- `01a02cc8-9c26-7b62-be67-396ac5c8ec81` · error · utility=1 · how do I get started with AI agents?
- `01a02cbf-6d9b-7062-a15a-eb4757dbd69c` · error · utility=1 · how do I get started with AI agents?

### Query decomposition

- `01a03546-0a36-7c53-9bd5-cdc290c245e0` · idle · utility=3 · what are the best frameworks for building AI agents?
- `01a0353e-228e-7323-848b-3c774756f718` · idle · utility=4 · what are the best ways of building effective AI agents considering the following components (architecture and design, te
- `01a03538-2783-7450-8647-91c19f7f8456` · idle · utility=4 · What's the best platform to deploy your ai agents?
- `01a02fa7-2578-7e91-a8f1-da6f805e2f5e` · idle · utility=5 · what is an AI agent?
- `01a02fa0-4686-7201-97df-4646bdaac19d` · idle · utility=5 · what is an AI agent?

### HyDE

- `01a03553-df3f-7951-8db8-66e2329f16b1` · idle · utility=4 · summarize everything I need to know about the openai agent sdk, it's uses and advantages over the claude agent sdk
- `01a03547-78b9-7e52-a2fd-730a44a974c8` · idle · utility=2 · what are the best frameworks for building AI agents?
- `01a0353e-f563-7d32-be1e-f358b59147a3` · idle · utility=4 · what are the best ways of building effective AI agents considering the following components (architecture and design, te
- `01a0353e-3c31-7f21-8e9e-ead80cb08c2b` · idle · utility=4 · what are the best ways of building effective AI agents considering the following components (architecture and design, te
- `01a03533-f037-7491-9888-dd985f8a3265` · idle · utility=4 · What's the best platform to deploy your ai agents?

