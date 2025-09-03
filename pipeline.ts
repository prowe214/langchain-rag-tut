import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { llm, vectorStore } from "./models";
import {
  MemorySaver,
  MessagesAnnotation,
  StateGraph,
} from "@langchain/langgraph";
import { z } from "zod";
import { tool } from "@langchain/core/tools";
import {
  createReactAgent,
  ToolNode,
  toolsCondition,
} from "@langchain/langgraph/prebuilt";
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

const checkpointer = new MemorySaver();

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

  // Step 1: Generate an AIMessage that may include a tool-call to be sent.
  async function queryOrRespond(state: typeof MessagesAnnotation.State) {
    const llmWithTools = llm.bindTools([retrieve]);
    const response = await llmWithTools.invoke(state.messages);
    // MessagesState appends messages to state instead of overwriting
    return { messages: [response] };
  }

  //   Step 2: Execute the retrieval
  const tools = new ToolNode([retrieve]);

  const agent = createReactAgent({
    llm,
    tools: [retrieve],
  });

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

  const graphWithMemory = graphBuilder.compile({ checkpointer });

  const threadConfig = {
    configurable: {
      thread_id: "abc123",
    },
    streamMode: "values" as const,
  };

  const inputs = {
    messages: [{ role: "user", content: "What is Task Decomposition?" }],
  };

  const inputs2 = {
    messages: [
      {
        role: "user",
        content: "Can you look up some common ways of doing it?",
      },
    ],
  };

  let inputMessage = `What is the standard method for Task Decomposition?
    Once you get the answer, look up common extensions of that method.`;

  let inputs5 = { messages: [{ role: "user", content: inputMessage }] };

  // first query
  for await (const step of await graphWithMemory.stream(inputs, threadConfig)) {
    const lastMessage = step.messages[step.messages.length - 1];

    prettyPrint(lastMessage);
    console.log("--------\n");
  }

  // followup query
  for await (const step of await graphWithMemory.stream(
    inputs2,
    threadConfig
  )) {
    const lastMessage = step.messages[step.messages.length - 1];

    prettyPrint(lastMessage);
    console.log("--------\n");
  }

  // using agent
  for await (const step of await agent.stream(inputs5, {
    streamMode: "values",
  })) {
    const lastMessage = step.messages[step.messages.length - 1];

    prettyPrint(lastMessage);
    console.log("--------\n");
  }
}

main();
