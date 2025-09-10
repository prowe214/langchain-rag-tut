import "dotenv/config";
import { ChatAnthropic } from "@langchain/anthropic";
import { CohereEmbeddings } from "@langchain/cohere";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

let _llm: ChatAnthropic;
let _embeddings: CohereEmbeddings;
let _vectorStore: MemoryVectorStore;

export function getLlm() {
  if (!_llm) {
    _llm = new ChatAnthropic({
      model: "claude-3-7-sonnet-20250219",
      temperature: 0,
      apiKey: process.env.ANTHROPIC_API_KEY,
    });
  }
  return _llm;
}

export function getEmbeddings() {
  if (!_embeddings) {
    _embeddings = new CohereEmbeddings({
      model: "embed-english-v3.0",
    });
  }
  return _embeddings;
}

export function getVectorStore() {
  if (!_vectorStore) {
    _vectorStore = new MemoryVectorStore(getEmbeddings());
  }
  return _vectorStore;
}
