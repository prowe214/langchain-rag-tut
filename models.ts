import "dotenv/config";
import { ChatAnthropic } from "@langchain/anthropic";
import { CohereEmbeddings } from "@langchain/cohere";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

export const llm = new ChatAnthropic({
  model: "claude-3-7-sonnet-20250219",
  temperature: 0,
  apiKey: process.env.ANTHROPIC_API_KEY,
});

export const embeddings = new CohereEmbeddings({
  model: "embed-english-v3.0",
});

export const vectorStore = new MemoryVectorStore(embeddings);
