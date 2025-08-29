import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { embeddings, llm, vectorStore } from "./models";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { Document } from "@langchain/core/documents";
import { Annotation, StateGraph } from "@langchain/langgraph";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { z } from "zod";

const pTagSelector = "p";
const cheerioLoader = new CheerioWebBaseLoader(
  "https://lilianweng.github.io/posts/2023-06-23-agent/",
  {
    selector: pTagSelector,
  }
);

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});

const searchSchema = z.object({
  query: z.string().describe("Search query to run."),
  section: z.enum(["beginning", "middle", "end"]).describe("Section to query."),
});

const structuredLlm = llm.withStructuredOutput(searchSchema);

async function main() {
  // load html content
  const docs = await cheerioLoader.load();

  // split massive text content into chunks
  const allSplits = await splitter.splitDocuments(docs);

  const totalDocuments = allSplits.length;
  const third = Math.floor(totalDocuments / 3);

  allSplits.forEach((document, i) => {
    if (i < third) {
      document.metadata.section = "beginning";
    } else if (i < third * 2) {
      document.metadata.section = "middle";
    } else {
      document.metadata.section = "end";
    }
  });

  await vectorStore.addDocuments(allSplits);

  // use custom system prompt
  const template = `Use the following pieces of context to answer the question at the end.
      If you don't know the answer, just say that you don't know, don't try to make up an answer.
      Use three sentences maximum and keep the answer as concise as possible.
      Always say "thanks for asking!" at the end of the answer.

      {context}

      Question: {question}

      Helpful Answer:`;

  const promptTemplate = ChatPromptTemplate.fromMessages([["user", template]]);
  const StateAnnotation = Annotation.Root({
    question: Annotation<string>,
    search: Annotation<z.infer<typeof searchSchema>>,
    context: Annotation<Document[]>,
    answer: Annotation<string>,
  });

  const analyzeQuery = async (state: typeof StateAnnotation.State) => {
    const result = await structuredLlm.invoke(state.question);
    return { search: result };
  };

  // retrieve with query analysis
  const retrieve = async (state: typeof StateAnnotation.State) => {
    const filter = (doc: Document) =>
      doc.metadata.section === state.search.section;
    const retrievedDocs = await vectorStore.similaritySearch(
      state.search.query,
      2,
      filter
    );
    console.log("RETRIEVED DOCUMENTS", retrievedDocs);

    return { context: retrievedDocs };
  };

  //   generate with query analysis
  const generate = async (state: typeof StateAnnotation.State) => {
    const docsContent = state.context
      .map((doc: Document) => doc.pageContent)
      .join("\n");
    const messages = await promptTemplate.invoke({
      question: state.question,
      context: docsContent,
    });

    const response = await llm.invoke(messages);
    return { answer: response.content };
  };

  const graph = new StateGraph(StateAnnotation)
    .addNode("analyzeQuery", analyzeQuery)
    .addNode("retrieve", retrieve)
    .addNode("generate", generate)
    .addEdge("__start__", "analyzeQuery")
    .addEdge("analyzeQuery", "retrieve")
    .addEdge("retrieve", "generate")
    .addEdge("generate", "__end__")
    .compile();

  const inputs = {
    question: "What does the end of the post say about task decomposition?",
  };

  console.log(inputs);
  console.log("\n========\n");
  // Print output in single response
  for await (const chunk of await graph.stream(inputs, {
    streamMode: "updates",
  })) {
    console.log(chunk);
    console.log("\n========\n");
  }
}

main();
