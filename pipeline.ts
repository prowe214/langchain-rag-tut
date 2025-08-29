import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { llm, vectorStore } from "./models";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { Document } from "@langchain/core/documents";
import {
  Annotation,
  MessagesAnnotation,
  StateGraph,
} from "@langchain/langgraph";
import { z } from "zod";
import { tool } from "@langchain/core/tools";
import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";
import {
  AIMessage,
  BaseMessage,
  HumanMessage,
  isAIMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";

const prettyPrint = (message: BaseMessage) => {
  let txt = `[${message._getType()}]: ${message.content}`;
  if ((isAIMessage(message) && message.tool_calls?.length) || 0 > 0) {
    const tool_calls = (message as AIMessage)?.tool_calls
      ?.map((tc) => `- ${tc.name}(${JSON.stringify(tc.args)})`)
      .join("\n");
    txt += ` \nTools: \n${tool_calls}`;
  }
  console.log(txt);
};

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

const retrieveSchema = z.object({ query: z.string() });

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

  // retrieve with query analysis
  //   const retrieve = async (state: typeof StateAnnotation.State) => {
  //     const filter = (doc: Document) =>
  //       doc.metadata.section === state.search.section;
  //     const retrievedDocs = await vectorStore.similaritySearch(
  //       state.search.query,
  //       2,
  //       filter
  //     );
  //     console.log("RETRIEVED DOCUMENTS", retrievedDocs);

  //     return { context: retrievedDocs };
  //   };

  // retrieve with tool-calling
  const retrieve = tool(
    async ({ query }) => {
      const retrievedDocs = await vectorStore.similaritySearch(query, 2);
      const serialized = retrievedDocs
        .map(
          (doc) => `Source: ${doc.metadata.source}\nContent: ${doc.pageContent}`
        )
        .join("\n");
      return [serialized, retrievedDocs];
    },
    {
      name: "retrieve",
      description: "Retrieve information related to a query",
      schema: retrieveSchema,
      responseFormat: "content_and_artifact",
    }
  );

  //   generate with query analysis
  //   const generate = async (state: typeof StateAnnotation.State) => {
  //     const docsContent = state.context
  //       .map((doc: Document) => doc.pageContent)
  //       .join("\n");
  //     const messages = await promptTemplate.invoke({
  //       question: state.question,
  //       context: docsContent,
  //     });

  //     const response = await llm.invoke(messages);
  //     return { answer: response.content };
  //   };

  // Step 1: Generate an AIMessage that may include a tool-call to be sent.
  async function queryOrRespond(state: typeof MessagesAnnotation.State) {
    const llmWithTools = llm.bindTools([retrieve]);
    const response = await llmWithTools.invoke(state.messages);
    // MessagesState appends messages to state instead of overwriting
    return { messages: [response] };
  }

  //   Step 2: Execute the retrieval
  const tools = new ToolNode([retrieve]);

  const analyzeQuery = async (state: typeof StateAnnotation.State) => {
    const result = await structuredLlm.invoke(state.question);
    return { search: result };
  };

  //   Step 3: Generate a response using the retrieved content
  async function generate(state: typeof MessagesAnnotation.State) {
    // get generated ToolMessages
    let recentToolMessages: BaseMessage[] = [];
    for (let i = state.messages.length - 1; i >= 0; i--) {
      let message = state.messages[i];
      if (message instanceof ToolMessage) {
        recentToolMessages.push(message);
      } else {
        break;
      }
    }

    let toolMessages = recentToolMessages.reverse();

    // Format into prompt
    const docsContent = toolMessages.map((doc) => doc.content).join("\n");
    const systemMessageContent =
      "You are an assistant for question-answering tasks. " +
      "Use the following pieces of retrieved context to answer " +
      "the question. If you don't know the answer, say that you " +
      "don't know. Use three sentences maximum and keep the " +
      "answer concise." +
      "\n\n" +
      `${docsContent}`;

    const conversationMessages = state.messages.filter(
      (message) =>
        message instanceof HumanMessage ||
        message instanceof SystemMessage ||
        (message instanceof AIMessage && message.tool_calls?.length == 0)
    );

    const prompt = [
      new SystemMessage(systemMessageContent),
      ...conversationMessages,
    ];

    // Run
    const response = await llm.invoke(prompt);
    return { messages: [response] };
  }

  //   const graph = new StateGraph(StateAnnotation)
  //     .addNode("analyzeQuery", analyzeQuery)
  //     .addNode("retrieve", retrieve)
  //     .addNode("generate", generate)
  //     .addEdge("__start__", "analyzeQuery")
  //     .addEdge("analyzeQuery", "retrieve")
  //     .addEdge("retrieve", "generate")
  //     .addEdge("generate", "__end__")
  //     .compile();
  const graphBuilder = new StateGraph(MessagesAnnotation)
    .addNode("queryOrRespond", queryOrRespond)
    .addNode("tools", tools)
    .addNode("generate", generate)
    .addEdge("__start__", "queryOrRespond")
    .addConditionalEdges("queryOrRespond", toolsCondition, {
      __end__: "__end__",
      tools: "tools",
    })
    .addEdge("tools", "generate")
    .addEdge("generate", "__end__");

  const graph = graphBuilder.compile();

  const inputs1 = { messages: [{ role: "user", content: "hello" }] };

  for await (const step of await graph.stream(inputs1, {
    streamMode: "values",
  })) {
    const lastMessage = step.messages[step.messages.length - 1];

    prettyPrint(lastMessage);
    console.log("--------\n");
  }
}

main();
