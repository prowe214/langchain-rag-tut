import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { getLlm, getVectorStore } from "./models";
import { z } from "zod";
import { tool } from "@langchain/core/tools";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import {
  AIMessage,
  BaseMessage,
  isAIMessage,
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

const retrieveSchema = z.object({ query: z.string() });

async function main() {
  // Parse command line arguments
  const args = process.argv.slice(2);
  
  if (args.length === 0) {
    console.error("Error: Please provide a query as a command line argument.");
    console.error("Usage: npx tsx pipeline.ts \"Your query here\"");
    process.exit(1);
  }
  
  const userQuery = args[0];
  console.log(`Processing query: "${userQuery}"`);
  
  // Initialize models (this is where API key validation happens)
  const llm = getLlm();
  const vectorStore = getVectorStore();
  
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

  const agent = createReactAgent({
    llm,
    tools: [retrieve],
  });

  // Create input from user query
  const userInput = {
    messages: [{ role: "user", content: userQuery }],
  };

  // Process the user query using the agent
  console.log("Processing with agent...\n");
  for await (const step of await agent.stream(userInput, {
    streamMode: "values",
  })) {
    const lastMessage = step.messages[step.messages.length - 1];

    prettyPrint(lastMessage);
    console.log("--------\n");
  }
}

main();
