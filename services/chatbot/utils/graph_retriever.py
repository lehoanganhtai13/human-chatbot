import ast
from typing import Any, List, Dict, Optional, Set, Type

from concurrent.futures import ThreadPoolExecutor, as_completed
from llama_index.core import PropertyGraphIndex, PromptTemplate
from llama_index.core.schema import TextNode
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.graph_stores.types import PropertyGraphStore
from llama_index.core.indices.property_graph.sub_retrievers.base import BasePGRetriever
from llama_index.core.llms import CustomLLM
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.retrievers import CustomPGRetriever, VectorContextRetriever, TextToCypherRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from pydantic import BaseModel

from chatbot.prompt.graph.summary import PARAPHRASE_PROMPT_TEMPLATE
from chatbot.query.graph_query import CYPHER_QUERY
from chatbot.utils.chat_store import CacheChatStore
from chatbot.utils.models_client import LLMCore

class Entities(BaseModel):
    """List of entity names or keywords to use for lookup in a knowledge graph."""
    names: Optional[List[str]]


class CustomCypherTemplateRetriever(BasePGRetriever):
    """A Cypher retriever that fills in params for a cypher query using an LLM.

    Args:
        graph_store (PropertyGraphStore):
            The graph store to retrieve data from.
        output_cls (Type[BaseModel]):
            The output class to use for the LLM.
            Should contain the params needed for the cypher query.
        cypher_query (str):
            The cypher query to use, with templated params.
        llm (Optional[LLMCore], optional):
            The language model to use.
    """
    def __init__(
        self,
        graph_store: PropertyGraphStore,
        output_cls: Type[BaseModel],
        cypher_query: str,
        llm: Optional[LLMCore] = None,
        **kwargs: Any,
    ) -> None:
        if not graph_store.supports_structured_queries:
            raise ValueError(
                "The provided graph store does not support cypher queries."
            )

        self.llm = llm
        self.output_cls: Type[BaseModel] = output_cls
        self.cypher_query = cypher_query

        super().__init__(
            graph_store=graph_store, include_text=True, include_properties=True
        )

    def retrieve_from_graph(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        try:
            query_str = query_bundle.query_str
            # Extract the prompt template and question from the query
            prompt_template_str = query_str.split("|mixed|")[0]
            question = query_str.split("|mixed|")[1]

            # Extract import entities from the query
            entity_extraction = LLMTextCompletionProgram.from_defaults(
                output_cls=self.output_cls,
                prompt_template_str=prompt_template_str,
                llm=self.llm,
                verbose=True
            )
            entity_name_list = entity_extraction(text=question).names

            if len(entity_name_list) == 0:
                print("No entities extracted from the query")
                return []
            
            print("Extracted entity list for Cypher query: ", entity_name_list)

            # Retrieve nodes from the graph store using parallel Cypher queries based extracted entities
            all_responses = set()  # Use set to avoid duplicates
            
            def query_single_entity(entity_name: str) -> Set[str]:
                response = self._graph_store.structured_query(
                    self.cypher_query,
                    param_map={"names": [entity_name]},
                )
                return {str(response)} if response else set()
            
            # Use ThreadPoolExecutor for parallel execution
            with ThreadPoolExecutor(max_workers=min(len(entity_name_list), 10)) as executor:
                future_to_entity = {
                    executor.submit(query_single_entity, entity): entity 
                    for entity in entity_name_list
                }
                
                for future in as_completed(future_to_entity):
                    try:
                        results = future.result()
                        all_responses.update(results)
                    except Exception as e:
                        print(f"Query failed for entity {future_to_entity[future]}: {e}")

            # Handle case when no results found
            if not all_responses:
                print("No results found in graph for the given entities")
                return []
            
            # Convert results to NodeWithScore format
            return [NodeWithScore(node=TextNode(text=response), score=1.0) for response in all_responses]
        except Exception as e:
            print(f"Error retrieving cypher nodes: {e}")
            return []

    async def aretrieve_from_graph(
        self, query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        try:
            query_str = query_bundle.query_str
            # Extract the prompt template and question from the query
            prompt_template_str = query_str.split("|mixed|")[0]
            question = query_str.split("|mixed|")[1]


            # Extract import entities from the query
            entity_extraction = LLMTextCompletionProgram.from_defaults(
                output_cls=self.output_cls,
                prompt_template_str=prompt_template_str,
                llm=self.llm,
                verbose=True
            )
            entity_name_list = entity_extraction(text=question).names

            if len(entity_name_list) == 0:
                print("No entities extracted from the query")
                return []
            
            print("Extracted entity list for Cypher query: ", entity_name_list)

            # Retrieve nodes from the graph store using parallel Cypher queries based extracted entities
            all_responses = set()  # Use set to avoid duplicates
            
            def query_single_entity(entity_name: str) -> Set[str]:
                response = self._graph_store.structured_query(
                    self.cypher_query,
                    param_map={"names": [entity_name]},
                )
                return {str(response)} if response else set()
            
            # Use ThreadPoolExecutor for parallel execution
            with ThreadPoolExecutor(max_workers=min(len(entity_name_list), 10)) as executor:
                future_to_entity = {
                    executor.submit(query_single_entity, entity): entity 
                    for entity in entity_name_list
                }
                
                for future in as_completed(future_to_entity):
                    try:
                        results = future.result()
                        all_responses.update(results)
                    except Exception as e:
                        print(f"Query failed for entity {future_to_entity[future]}: {e}")

            # Handle case when no results found
            if not all_responses:
                print("No results found in graph for the given entities")
                return []

            # Convert results to NodeWithScore format
            return [NodeWithScore(node=TextNode(text=response), score=1.0) for response in all_responses]
        except Exception as e:
            print(f"Error retrieving cypher nodes: {e}")
            return []
    

# See: https://neo4j.com/labs/genai-ecosystem/llamaindex/ and
# https://docs.llamaindex.ai/en/latest/module_guides/indexing/lpg_index_guide/#schemallmpathextractor and
# https://www.llamaindex.ai/blog/customizing-property-graph-index-in-llamaindex and
# https://python.langchain.com/docs/how_to/graph_prompting/ and
# https://community.aws/content/2kOWDPgScaWwILcSgQfRVpfbTYa/knowledge-graphs-and-generative-ai-graphrag-with-amazon-neptune-and-llamaindex-part-1-natural-language-querying
class CustomSubRetriever(CustomPGRetriever):
    """Custom retriever with entity detection."""
        
    def init(
        self,
        embed_model: Optional[BaseEmbedding] = None,
        llm: Optional[CustomLLM] = None,
        graph_store: Optional[PropertyGraphStore] = None,
        similarity_top_k: int = 4,
        path_depth: int = 1,
        include_text: bool = True,
        **kwargs: Any
    ) -> None:
        """Uses any kwargs passed in from class constructor."""
        self.llm = llm
        self.vector_retriever = VectorContextRetriever(
            self._graph_store,
            include_text=include_text,
            embed_model=embed_model,
            similarity_top_k=similarity_top_k,
            path_depth=path_depth,
        )
        # self.cypher_retriever = TextToCypherRetriever(
        #     llm=llm,
        #     graph_store=self._graph_store,
        #     summarize_response=True,
        #     text_to_cypher_template=TEXT_TO_CYPHER_PROMPT_TEMPLATE,
        #     include_raw_response_as_metadata=True
        # )
        self.cypher_retriever = CustomCypherTemplateRetriever(
            graph_store=self._graph_store,
            llm=self.llm,
            cypher_query=CYPHER_QUERY,
            output_cls=Entities
        )
        self.top_k = similarity_top_k

    def process_cypher_nodes(self, cypher_nodes: List[NodeWithScore], top_k=10) -> List[NodeWithScore]:
        """Process the cypher nodes."""
        texts = []
        for cypher_node in cypher_nodes:
            text_list = ast.literal_eval(cypher_node.text)
            texts_list = [item["c.text"] for item in text_list]
            texts.extend(texts_list)

        # Remove duplicate texts
        seen_texts = set()
        unique_texts = []
        for text in texts:
            if text not in seen_texts:
                seen_texts.add(text)
                unique_texts.append(text)
        unique_texts = unique_texts[:top_k]

        cypher_nodes = [NodeWithScore(node=TextNode(text=text), score=1.0) for text in unique_texts]

        return cypher_nodes
    
    def process_vector_nodes(self, vector_nodes: List[NodeWithScore], top_k=10) -> List[NodeWithScore]:
        """Process the vector nodes."""
        seen_relationships = set()
        seen_contexts = set()
        unique_relationships = []
        unique_contexts = []
        for node in vector_nodes:
            text = node.node.text
            relationships = text.split("\n\n")[1]
            context = text.split("\n\n")[2]

            for relationship in relationships.split("\n"):
                if relationship not in seen_relationships:
                    seen_relationships.add(relationship)
                    unique_relationships.append(relationship)
            if context not in seen_contexts:
                seen_contexts.add(context)
                unique_contexts.append(context)

        unique_texts = unique_relationships[:top_k] + unique_contexts[:top_k]

        vector_nodes = [NodeWithScore(node=TextNode(text=text), score=1.0) for text in unique_texts]
        return vector_nodes

    def custom_retrieve(self, query_str: str) -> List[NodeWithScore]:
        """Define custom retriever with entity detection.

        Could return `str`, `TextNode`, `NodeWithScore`, or a list of those.
        """
        split_query = query_str.split("|mixed|")[1]

        retry_times = 3
        while retry_times > 0:
            try:

                result_nodes = []
                seen_nodes = set()

                # Retrieve nodes using both vector and cypher retrievers in parallel
                with ThreadPoolExecutor(max_workers=2) as executor:
                    vector_future = executor.submit(self.vector_retriever.retrieve, split_query)
                    cypher_future = executor.submit(self.cypher_retriever.retrieve, query_str)

                    for future in as_completed([vector_future, cypher_future]):
                        if future == vector_future:
                            vector_nodes = future.result()
                            processed_vector_nodes = self.process_vector_nodes(vector_nodes, top_k=self.top_k)
                            print(f"Retrieved {len(processed_vector_nodes)} vector nodes.")
                            for node in processed_vector_nodes:
                                if node.text not in seen_nodes:
                                    seen_nodes.add(node.text)
                                    result_nodes.append(node)
                        elif future == cypher_future:
                            cypher_nodes = future.result()
                            
                            # Skip processing if no cypher node is retrieved
                            if len(cypher_nodes) == 0:
                                print(f"Retrieved 0 cypher node.")
                                continue
                            
                            processed_cypher_nodes = self.process_cypher_nodes(cypher_nodes, top_k=self.top_k)
                            print(f"Retrieved {len(processed_cypher_nodes)} cypher nodes.")
                            for node in processed_cypher_nodes:
                                if node.text not in seen_nodes:
                                    seen_nodes.add(node.text)
                                    result_nodes.append(node)
                
                return result_nodes
            except Exception as e:
                print(f"Error retrieving nodes: {e}")
                retry_times -= 1
    

class Retriever():
    """A class to retrieve nodes from the graph store."""
    def __init__(self, sub_retriever: CustomSubRetriever, graph_store: PropertyGraphIndex, llm: CustomLLM, cache_chat_store: CacheChatStore):
        self.sub_retriever = sub_retriever
        self.graph_store = graph_store
        self.llm = llm
        self.cache_chat_store = cache_chat_store

    def query_transformation(self, query: str, user_id: str, assistant_id: str, coversation_history: List[Dict[str, str]], use_llm: bool = False) -> str:
        """Transform the query string into a format that can be used for retrieval."""
        
        if not use_llm:
            # Clean the query
            query = query.replace("?", "").replace("!", "").replace(".", "").replace(",", "").replace(":", "").replace(";", "").replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace("{", "").replace("}", "")
            query = query.lower()

            # Add spaces before and after the query for later pronoun replacement
            query = " " + query + " "      

            # Replace pronouns with user and assistant IDs
            query = query.replace(" am i ", f" is {user_id} ").replace(" are you ", f" are {assistant_id} ").replace(" are we ", f" are {user_id} and {assistant_id} ")
            query = query.replace(" you ", f" {assistant_id} ").replace(" your ", f" {assistant_id}'s ").replace(" you're ", f" {assistant_id} is ").replace(" you've ", f" {assistant_id} has ").replace(" you'll ", f" {assistant_id} will ")
            query = query.replace(" i ", f" {user_id} ").replace(" me ", f" {user_id} ").replace(" my ", f" {user_id}'s ").replace(" i'm ", f" {user_id} is ").replace(" i've ", f" {user_id} has ").replace(" i'll ", f" {user_id} will ")
            query = query.replace(" we'll ", f" {user_id} and {assistant_id} will ").replace(" we ", f" {user_id} and {assistant_id} ").replace(" us ", f" {user_id} and {assistant_id} ").replace(" our ", f" {user_id}'s and {assistant_id}'s ")
            return query

        list_of_conversation_messages = []
        conversation_messages = ""
        
        if len(coversation_history) > 0:
            for message in coversation_history:
                if message["role"] == "user":
                    list_of_conversation_messages.append(f"{user_id} said to {assistant_id}, \"{message['content']}\".")
                else:
                    list_of_conversation_messages.append(f"{assistant_id} replied, \"{message['content']}\"")
            conversation_messages = "\n".join(list_of_conversation_messages)

        # Fill the prompt template with the user and assistant IDs, conversation history, and input sentence
        prompt = PromptTemplate(PARAPHRASE_PROMPT_TEMPLATE).format(
            speaker=user_id,
            listener=assistant_id,
            conversation_history=conversation_messages,
            input_sentence=f"{user_id} said to {assistant_id}, \"{query}\"."
        )

        retry_times = 3
        while retry_times > 0:
            try:
                transformed_query = self.llm.complete(prompt).text
                # Clean the output
                if "Output:" in transformed_query:
                    transformed_query = transformed_query.split("Output:")[1].strip()
                    if "\"" in transformed_query:
                        transformed_query = transformed_query.replace("\"", "")
                return transformed_query
            except Exception as e:
                print(f"Error transforming query: {e}")
                retry_times -= 1

    def retrieve(
            self,
            query: str = None,
            prompt_template_str: str = None,
            user_id: str = None,
            assistant_id: str = None
    ) -> List[NodeWithScore]:
        """Retrieve nodes from the graph store."""

        import time
        start = time.time()

        coversation_history = self.cache_chat_store.get_chat_history(user_id, en_translate=False)
        end_retrieve_history = time.time()

        # Transform the query
        transformed_query = self.query_transformation(
            query,
            user_id=user_id,
            assistant_id=assistant_id,
            coversation_history=coversation_history
        )
        end_transform = time.time()

        print(f"Transformed query: {transformed_query}")

        mixed_query = f"{prompt_template_str}|mixed|{transformed_query}"

        # Add the cypher query validator to the text-to-cypher retriever with latest graph schema
        # self.sub_retriever.cypher_retriever.cypher_validator = cypher_query_corrector

        self.retriever = self.graph_store.as_retriever(sub_retrievers=[self.sub_retriever])
        retrieved_nodes = self.retriever.retrieve(mixed_query)

        # print(f"Total transformation time: {end_transform - start}")

        return retrieved_nodes
    
    async def async_retrieve(
            self,
            query: str = None,
            prompt_template_str: str = None,
            user_id: str = None,
            assistant_id: str = None
    ) -> List[NodeWithScore]:
        """Retrieve nodes from the graph store asynchronously."""
        import asyncio
        import nest_asyncio

        original_loop = asyncio.get_event_loop()
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        asyncio.set_event_loop(loop)

        retrieved_nodes = self.retrieve(query, prompt_template_str, user_id, assistant_id)
        
        asyncio.set_event_loop(original_loop)
        
        return retrieved_nodes
