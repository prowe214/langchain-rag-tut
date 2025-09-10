# LangChain RAG AI Tutorial Output

This codebase is the output from following the [RAG tutorial from LangChain](https://js.langchain.com/docs/tutorials/rag).

The tutorial focuses on the following topics:

- **Initializing** an app with models for chat, embeddings, and vector storage
- **Loading** documents from a remote source (blog post)
- **Splitting** documents into chunks
- **Storing** documents in a vector store
- **Retrieving** documents relevant to a user query
- **Generating** a response based on retrieved documents
- **Orchestrating** the control flow via LangGraph
- **Invocation modes** of sync, async, and streaming output
- **System prompt** customization, or using a pre-built prompt
- **Query Analysis** to optimize user search query

## Usage

The pipeline now accepts user queries via command line arguments:

```bash
npx tsx pipeline.ts "Your query here"
```

### Examples

```bash
npx tsx pipeline.ts "What is Task Decomposition?"
npx tsx pipeline.ts "How do agents work in AI systems?"
npx tsx pipeline.ts "What are the common ways to implement reasoning in AI?"
```

### Prerequisites

You need to set up the following environment variables:

- `ANTHROPIC_API_KEY` - Your Anthropic API key for Claude
- `COHERE_API_KEY` - Your Cohere API key for embeddings

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your_anthropic_key_here
COHERE_API_KEY=your_cohere_key_here
```
