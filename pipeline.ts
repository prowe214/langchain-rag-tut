import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { embeddings, llm, vectorStore } from "./models";
import { pull } from "langchain/hub";
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

const vectorStoreQA = new MemoryVectorStore(embeddings);

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

  await vectorStoreQA.addDocuments(allSplits);

  // embed and store all splits into vector store
  //   await vectorStore.addDocuments(allSplits);

  //   use pre-created langchain prompt for helpful chatbot
  //   const promptTemplate = await pull<ChatPromptTemplate>("rlm/rag-prompt");

  // use custom system prompt
  const template = `Use the following pieces of context to answer the question at the end.
      If you don't know the answer, just say that you don't know, don't try to make up an answer.
      Use three sentences maximum and keep the answer as concise as possible.
      Always say "thanks for asking!" at the end of the answer.

      {context}

      Question: {question}

      Helpful Answer:`;

  const promptTemplate = ChatPromptTemplate.fromMessages([["user", template]]);

  //   const InputStateAnnotation = Annotation.Root({
  //     question: Annotation<string>,
  //   });

  //   const StateAnnotation = Annotation.Root({
  //     question: Annotation<string>,
  //     context: Annotation<Document[]>,
  //     answer: Annotation<string>,
  //   });

  const StateAnnotationQA = Annotation.Root({
    question: Annotation<string>,
    search: Annotation<z.infer<typeof searchSchema>>,
    context: Annotation<Document[]>,
    answer: Annotation<string>,
  });

  const analyzeQuery = async (state: typeof StateAnnotationQA.State) => {
    const result = await structuredLlm.invoke(state.question);
    return { search: result };
  };

  //   retrieve without query analysis
  //   const retrieve = async (state: typeof InputStateAnnotation.State) => {
  //     const retrievedDocs = await vectorStore.similaritySearch(state.question);
  //     return { context: retrievedDocs };
  //   };

  // generate without query analysis
  //   const generate = async (state: typeof StateAnnotation.State) => {
  //     const docsContent = state.context.map((doc) => doc.pageContent).join("\n");
  //     const messages = await promptTemplate.invoke({
  //       question: state.question,
  //       context: docsContent,
  //     });
  //     const response = await llm.invoke(messages);
  //     return { answer: response.content };
  //   };

  // retrieve with query analysis
  const retrieveQA = async (state: typeof StateAnnotationQA.State) => {
    const filter = (doc: Document) =>
      doc.metadata.section === state.search.section;
    const retrievedDocs = await vectorStoreQA.similaritySearch(
      state.search.query,
      2
      //   filter
    );
    console.log("RETRIEVED DOCUMENTS", retrievedDocs);

    return { context: retrievedDocs };
  };

  //   generate with query analysis
  const generateQA = async (state: typeof StateAnnotationQA.State) => {
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

  //   const graph = new StateGraph(StateAnnotation)
  //     .addNode("retrieve", retrieve)
  //     .addNode("generate", generate)
  //     .addEdge("__start__", "retrieve")
  //     .addEdge("retrieve", "generate")
  //     .addEdge("generate", "__end__")
  //     .compile();

  const graphQA = new StateGraph(StateAnnotationQA)
    .addNode("analyzeQuery", analyzeQuery)
    .addNode("retrieveQA", retrieveQA)
    .addNode("generateQA", generateQA)
    .addEdge("__start__", "analyzeQuery")
    .addEdge("analyzeQuery", "retrieveQA")
    .addEdge("retrieveQA", "generateQA")
    .addEdge("generateQA", "__end__")
    .compile();

  const inputs = { question: "What is task decomposition?" };
  const inputsQA = {
    question: "What does the end of the post say about task decomposition?",
  };

  console.log(inputsQA);
  console.log("\n========\n");
  // Print output in single response
  for await (const chunk of await graphQA.stream(inputsQA, {
    streamMode: "updates",
  })) {
    console.log(chunk);
    console.log("\n========\n");
  }
  //     const result = await graphQA.invoke(inputs);
  //   console.log(result.context.slice(0, 2));
  //   console.log(`\nAnswer: ${result.answer}`);

  // Stream output chunks
  //   console.log("inputs:", inputs);
  //   console.log("\n========\n");
  //   for await (const chunk of await graph.stream(inputs, {
  //     streamMode: "updates",
  //   })) {
  //     console.log(chunk);
  //     console.log("\n========\n");
  //   }

  // Stream tokens
  // const stream = await graph.stream(inputs, { streamMode: "messages" });
  // for await (const [message, _metadata] of stream) {
  //   process.stdout.write(message.content + "");
  // }
}

main();
