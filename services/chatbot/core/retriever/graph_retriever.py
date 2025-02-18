from typing import Any, List, Dict, Optional, Tuple

from datetime import datetime
import json
import pytz
import time
from concurrent.futures import ThreadPoolExecutor

from llama_index.core import PropertyGraphIndex, PromptTemplate
from llama_index.core.schema import TextNode
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import CustomLLM
from llama_index.core.retrievers import CustomPGRetriever
from llama_index.core.schema import NodeWithScore

from chatbot.prompt.graph.query_decomposition import EXTRACT_SUBQUERIES_PROMPT_TEMPLATE
from chatbot.prompt.graph.prompt_paraphrasing import PARAPHRASE_PROMPT_TEMPLATE
from chatbot.core.chat_stores import CacheChatStore
from chatbot.core.retriever.local_retriever import LocalContextRetriever
from chatbot.core.retriever.global_retriever import GlobalContextRetriever
from chatbot.core.retriever.exceptions import (
    GraphRetrieveError,
    DecomposeQueryError,
    TransformQueryError
)
from chatbot.core.retriever.triplet_data import EntityInfo, RelationshipInfo, TextChunkInfo
    

# See: https://neo4j.com/labs/genai-ecosystem/llamaindex/ and
# https://docs.llamaindex.ai/en/latest/module_guides/indexing/lpg_index_guide/#schemallmpathextractor and
# https://www.llamaindex.ai/blog/customizing-property-graph-index-in-llamaindex and
# https://python.langchain.com/docs/how_to/graph_prompting/ and
# https://community.aws/content/2kOWDPgScaWwILcSgQfRVpfbTYa/knowledge-graphs-and-generative-ai-graphrag-with-amazon-neptune-and-llamaindex-part-1-natural-language-querying
class CustomSubRetriever(CustomPGRetriever):
    """
    Custom sub-retriever is combined with local and global search techniques of architecture developed based on both Microsoft's GraphRAG and LightRAG.

    Args:
        - embed_model (Optional[BaseEmbedding], optional):
            The embedding model to use. Defaults to None.
        - llm (Optional[CustomLLM], optional):
            The LLM model to use. Defaults to None.
        - top_k_local_entity (int, optional):
            The number of top similar entities to retrieve. Defaults to 10.
        - top_k_global_relation (int, optional):
            The number of top similar relations to retrieve. Defaults to 10.
        - top_k_chunk (int, optional):
            The number of top text chunks to include in the context. Defaults to 4.
        - top_k_community (int, optional):
            The number of top communities to include in the context. Defaults to 4.
            This parameter is not used in the current implementation.
        - relation_path_depth (int, optional):
            The depth of the path to retrieve for each entity. 
            Higher depth will retrieve more relationships (in-network and out-of-network). Defaults to 2.
        - max_num_sub_queries (int, optional):
            The maximum number of subqueries to extract from the original query. Defaults to 4.
        - timezone (str, optional):
            The timezone of the selected city to use for the current timestamp. Defaults to "UTC".
        - debug (bool, optional):
            Whether to print debug information. Defaults to False.
    """
    
    llm: CustomLLM
    local_context_retriever: LocalContextRetriever
    global_context_retriever: GlobalContextRetriever
        
    def init(
        self,
        embed_model: Optional[BaseEmbedding] = None,
        llm: Optional[CustomLLM] = None,
        top_k_local_entity: int = 10,
        top_k_global_relation: int = 10,
        top_k_chunk: int = 4,
        top_k_community: int = 4,
        relation_path_depth: int = 2,
        max_num_sub_queries: int = 4,
        timezone: str = "UTC",
        debug: bool = False,
        **kwargs: Any
    ) -> None:
        self.llm = llm
        self.local_context_retriever = LocalContextRetriever(
            self._graph_store,
            include_text=False,
            include_properties=True, # Include properties to get the description of the retrieved entities as well
            embed_model=embed_model,
            similarity_top_k=top_k_local_entity,
            similarity_score=0.4,   # Remove the noise triplets with low maximum similarity score (50-60% of total triplets)
            path_depth=relation_path_depth,
        )
        self.global_context_retriever = GlobalContextRetriever(
            self._graph_store,
            include_text=False,
            include_properties=True, # Include properties to get the description of the retrieved relations as well
            embed_model=embed_model,
            similarity_top_k=top_k_global_relation,
            similarity_score=0.4,   # Remove the noise relations with low maximum similarity score (50-60% of total relation)
        )
        self.top_k_chunk = top_k_chunk
        self.top_k_community = top_k_community
        self.max_num_sub_queries = max_num_sub_queries
        self.timezone = timezone
        self.debug = debug

    def custom_retrieve(self, query_str: str) -> List[NodeWithScore]:
        """
        Custom retrieval method to retrieve nodes based on combination of techniques from Microsoft's GraphRAG and LightRAG.

        Args:
            query_str (str):
                The query string to retrieve nodes for.

        Returns:
            List[NodeWithScore]:
                The final context nodes aggregated from local and global retrievals.
        """

        try:
            if self.debug:
                self.local_context_retriever.debug = True
                self.global_context_retriever.debug = True

            # Decompose the query into global and local subqueries
            start = time.time()
            subqueries = self.decompose_query(query_str)
            decompose_time = time.time() - start

            # Retrieve nodes using vector retriever for each subquery   
            global_subqueries = subqueries["global_subqueries"]
            local_subqueries = subqueries["local_subqueries"]
            if self.debug:
                print("=== Subqueries ===")
                print("Global subqueries:")
                for subquery in global_subqueries:
                    print(subquery)
                print("\nLocal subqueries:")
                for subquery in local_subqueries:
                    print(subquery)
                print("===================")

            start = time.time()
            with ThreadPoolExecutor() as executor:
                future_global = executor.submit(self.process_search, global_subqueries, "global")
                future_local = executor.submit(self.process_search, local_subqueries, "local")
                
                global_entity_info, global_relationship_info, _ = future_global.result()
                local_entity_info, local_relationship_info, _ = future_local.result()
            total_retrieve_time = time.time() - start

            start = time.time()
            local_selected_entity_list = self.local_context_retriever.get_selected_entities()
            global_selected_entity_list = self.global_context_retriever.get_selected_entities()
            selected_entity_list = local_selected_entity_list
            for entity in global_selected_entity_list:
                if entity not in selected_entity_list:
                    selected_entity_list.append(entity)
            if self.debug:
                print(f"Selected entity list: {selected_entity_list}")

            total_context = []

            # Add timezone to the context
            total_context.append(f"Timezone: {self.timezone}\n")
            if self.debug:
                print(f"Timezone: {self.timezone}\n")

            entity_context = self.build_entity_context(local_entity_info, global_entity_info)
            total_context.append(entity_context)
            if self.debug:
                print(f"Entity context:\n{entity_context}")

            relationship_context = self.build_relationship_context(local_relationship_info, global_relationship_info)
            total_context.append(relationship_context)
            if self.debug:
                print(f"Relationship context:\n{relationship_context}")

            text_chunk_context, entity_chunk_info = self.build_text_chunk_context(selected_entity_list, global_relationship_info)
            total_context.append(text_chunk_context)
            if self.debug:
                print(f"Text chunk context:\n{text_chunk_context}")
                print(f"Entity chunk info:\n{entity_chunk_info}")

            final_context = "\n".join(total_context)
            # if self.debug:
            #     print(f"Final context:\n{final_context}")

            context_node = [NodeWithScore(node=TextNode(text=final_context), score=1.0)]
            build_context_time = time.time() - start

            if self.debug:
                print(f"Decompose query time: {decompose_time}")
                print(f"Total retrieve time: {total_retrieve_time}")
                print(f"Build context time: {build_context_time}")
            
            return context_node
        except Exception as e:
            raise GraphRetrieveError(f"Error retrieving nodes: {e}")

    def process_search(self, list_subqueries: List[str], search_mode: str) -> Tuple[Dict[str, EntityInfo], List[RelationshipInfo], Dict[str, TextChunkInfo]]:
        """
        Process the search for the given list of subqueries based on the search mode.

        Args:
            subquery (str):
                The subquery to search for.
            search_mode (str):
                The search mode to use (global or local).

        Returns:
            Tuple[List[str], List[str], List[str]]:
                A tuple containing the selected entities, relationships, and text chunk information.
        """

        if search_mode == "global":
            return self.global_context_retriever.global_search(list_subqueries)
        elif search_mode == "local":
            return self.local_context_retriever.local_search(list_subqueries)
        else:
            raise ValueError(f"Invalid search mode: {search_mode}")

    def decompose_query(self, query_str: str) -> Dict[str, List[str]]:
        """
        Decompose the original query into global and/or local subqueries.
        
        Args:
            query_str (str):
                The original query string.

        Returns:
            Dict[str, List[str]]:
                A dictionary containing the global and local subqueries.
        """

        prompt = PromptTemplate(EXTRACT_SUBQUERIES_PROMPT_TEMPLATE).format(
            text=query_str,
            max_num_sub_queries=self.max_num_sub_queries
        )

        retry_times = 3
        while retry_times > 0:
            try:
                response = self.llm.complete(prompt=prompt)
                subqueries = json.loads(response.text)
                return subqueries
            except Exception as e:
                retry_times -= 1
                if retry_times == 0:
                    raise DecomposeQueryError(f"Error decomposing query: {e}")
    
    def build_entity_context(
        self,
        local_entity_info: Dict[str, EntityInfo],
        global_entity_info: Dict[str, EntityInfo]
    ) -> str:
        """
        Build the entity context in data table format.
        
        Args:
            local_entity_info (Dict[str, str]):
                The local entity information.
            global_entity_info (Dict[str, str]):
                The global entity information.

        Returns:
            str:
                The entity context in data table format.
        """

        entity_context = (
            "----Entity Report----\n"
            "Entity|Description|Created Timestamp\n"
        )

        # Add local entities first and then global entities since local entities hold local context
        for entity_id, entity_info in local_entity_info.items():
            description = entity_info.description
            timestamp = entity_info.timestamp
            if not entity_id or not description:
                continue
            entity_context += f"{entity_id}|{description}|{timestamp}\n"

        for entity_id, entity_info in global_entity_info.items():
            description = entity_info.description
            timestamp = entity_info.timestamp
            if not entity_id or not description:
                continue
            entity_context += f"{entity_id}|{description}|{timestamp}\n"
        
        return entity_context
    
    def build_relationship_context(
        self,
        local_relationship_info: List[RelationshipInfo],
        global_relationship_info: List[RelationshipInfo]
    ) -> str:
        """
        Build the relationship context in data table format.

        Args:
            local_relationship_info (List[str]):
                The local relationship information.
            global_relationship_info (List[str]):
                The global relationship information.

        Returns:
            str:
                The relationship context in data table format.
        """

        relationship_context = (
            "----Relationship Report----\n"
            "Source Entity|Target Entity|Description|Created Timestamp\n"
        )
        # Add global relationships first and then local relationships since global relationships hold global context
        for relationship_info in global_relationship_info:
            source_entity = relationship_info.source_entity
            target_entity = relationship_info.target_entity
            relationship = relationship_info.relationships
            timestamp = relationship_info.timestamp
            if not source_entity or not target_entity or not relationship:
                continue
            relationship_context += f"{source_entity}|{target_entity}|{relationship}|{timestamp}\n"

        for relationship_info in local_relationship_info:
            source_entity = relationship_info.source_entity
            target_entity = relationship_info.target_entity
            relationship = relationship_info.relationships
            timestamp = relationship_info.timestamp
            if not source_entity or not target_entity or not relationship:
                continue
            relationship_context += f"{source_entity}|{target_entity}|{relationship}|{timestamp}\n"

        return relationship_context

    def build_text_chunk_context(
        self,
        selected_entity_list: List[str],
        global_relationship_info: List[RelationshipInfo]
    ) -> Tuple[str, Dict[str, List[str]]]:
        """
        Build the text chunk context in data table format.

        Args:
            selected_entity_list (List[str]):
                The list of selected entities.
            global_relationship_info (List[RelationshipInfo]):
                The global relationship data.

        Returns:
            Tuple[str, Dict[str, List[str]]]:
                The text chunk context in data table format and the entity chunk information.
        """

        if len(selected_entity_list) == 0:
            return "", {}

        text_chunk_context = (
            "----Text Chunk Report----\n"
            "ID|Text Chunk|Created Timestamp\n"
        )

        # Get all of the relationships related to the selected entities
        data = self._graph_store.structured_query("""
        MATCH (e)-[r]-(other)
        WHERE e.id IN $selected_entities AND type(r) <> "MENTIONS"
        RETURN e.id AS entity_id, 
            e{.*} AS entity_properties, 
            COLLECT({
                relationship_type: type(r),
                relationship_properties: r{.*}
            }) AS relationships
        """, param_map={"selected_entities": selected_entity_list})

        chunk_info = {}
        entity_chunk_info = {}
        for node in data:
            # Add the chunk IDs related to the entity
            if node["entity_id"] not in entity_chunk_info:
                entity_chunk_info[node["entity_id"]] = []
            entity_chunk_info[node["entity_id"]].extend(node["entity_properties"]["triplet_source_id"] or [])

            # Add the relationship types related to the chunk IDs
            for source_id in node["entity_properties"]["triplet_source_id"] or []:
                if source_id not in chunk_info:
                    chunk_info[source_id] = set()
                for relationship in node["relationships"]:
                    if source_id in relationship["relationship_properties"]["triplet_source_id"]:
                        chunk_info[source_id].add(relationship["relationship_type"])

        # Count the number of relationships for each chunk and sort them by the number of relationships descending
        chunk_relationship_counts  = {id: len(relationships) for id, relationships in chunk_info.items()}
        sorted_chunk_ids = sorted(chunk_relationship_counts.keys(), key=lambda k: chunk_relationship_counts[k], reverse=True)

        # Sort the chunk IDs based on the global relationships
        high_priority_chunk_ids = []
        low_priority_chunk_ids = []
        for chunk_id, relationship_info in zip(sorted_chunk_ids, global_relationship_info):
            if chunk_id in relationship_info.chunk_source_id:
                high_priority_chunk_ids.append(chunk_id)
            else:
                low_priority_chunk_ids.append(chunk_id)

        sorted_chunk_ids = high_priority_chunk_ids + low_priority_chunk_ids

        # Fetch chunk texts based on the sorted chunk IDs
        data = self._graph_store.structured_query("""
        MATCH (e:Chunk)
        WHERE e.id IN $source_id
        RETURN e.text AS chunk_text,
            e.id AS chunk_id,
            e.created_timestamp AS created_timestamp
        """, param_map={"source_id": sorted_chunk_ids})

        # Create a map of chunk IDs to chunk data (text and timestamp)
        chunk_data_map = {chunk["chunk_id"]: chunk for chunk in data}

        # Get list of the chunk texts in the order of the sorted chunk IDs
        list_of_chunks = []
        list_of_timestamps = []
        for chunk_id in sorted_chunk_ids:
            # Check if the chunk ID is in the chunk data map
            if chunk_id in chunk_data_map:
                chunk_info = chunk_data_map[chunk_id]
                list_of_chunks.append(chunk_info["chunk_text"])
                list_of_timestamps.append(chunk_info["created_timestamp"])

        text_chunk_context += "\n".join(
            [f"{i}|{chunk}|{timestamp}" for i, (chunk, timestamp) in enumerate(zip(list_of_chunks[:self.top_k_chunk], list_of_timestamps[:self.top_k_chunk]))]
        )
        text_chunk_context += "\n"

        return text_chunk_context, entity_chunk_info
    
    def build_community_context(self, selected_entity_list: List[str], entity_chunk_info: Dict[str, List[str]]) -> str:
        """
        Build the community context in data table format.

        Args:
            selected_entity_list (List[str]):
                The list of selected entities.

        Returns:
            str:
                The community context in data table format.
        """

        if len(selected_entity_list) == 0 or not entity_chunk_info:
            return ""

        community_context = (
            "----Community Report----\n"
            "ID|Community Title|Community Summary\n"
        )

        entity_info = self._graph_store.entity_info
        community_summary = self._graph_store.community_summary

        # Count the number of text chunks mentioning the entities related to each community
        community_text_chunk = {}
        for entity_id in selected_entity_list:
            for community_id in entity_info[entity_id]:
                if community_id not in community_text_chunk:
                    community_text_chunk[community_id] = []
                community_text_chunk[community_id].extend(entity_chunk_info[entity_id]) # Add the text chunk IDs related to the entity

        # Order the communities by the number of text chunks descending
        community_text_chunk = {community_id: len(set(text_chunks)) for community_id, text_chunks in community_text_chunk.items()}
        sorted_community_ids = sorted(community_text_chunk.keys(), key=lambda k: community_text_chunk[k], reverse=True)

        # Build the community context
        for i, community_id in enumerate(sorted_community_ids[:self.top_k_community]):
            community_context += f"{i}|{community_summary[community_id]['title']}|{community_summary[community_id]['summary']}\n"

        return community_context
    
    def get_current_timestamp(self) -> str:
        """
        Get the current timestamp in the selected timezone.

        Returns:
            str: The current timestamp.
        """
        timezone = pytz.timezone(self.timezone)
        return datetime.now(timezone).strftime("%Y-%m-%d %H:%M:%S")
    

class Retriever():
    """A class to retrieve nodes from the graph store."""
    def __init__(self, sub_retriever: CustomSubRetriever, graph_store: PropertyGraphIndex, llm: CustomLLM, cache_chat_store: CacheChatStore):
        self.sub_retriever = sub_retriever
        self.graph_store = graph_store
        self.llm = llm
        self.cache_chat_store = cache_chat_store

    def query_transformation(self, query: str, user_id: str, assistant_id: str, coversation_history: List[Dict[str, str]], use_llm: bool = False) -> str:
        """
        Transform the query string into a format that can be used for retrieval.
        
        Args:
            query (str):
                The input query string.
            user_id (str):
                The user ID (name, nickname, or alias).
            assistant_id (str):
                The assistant ID (name, nickname, or alias).
            coversation_history (List[Dict[str, str]]):
                The conversation history used for query transformation with LLM.
            use_llm (bool, optional):
                Whether to use LLM for query transformation. Defaults to False.

        Returns:
            str:
                The transformed query string.
        """
        
        if not use_llm:
            # Clean the query
            query = query.replace("?", "").replace("!", "").replace(".", "").replace(",", "").replace(":", "").replace(";", "").replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace("{", "").replace("}", "")
            query = query.lower()

            # Add spaces before and after the query for later pronoun replacement
            query = " " + query + " "      

            # Replace pronouns with user and assistant IDs and possessive forms
            query = query.replace(" am i ", f" is {user_id} ").replace(" are you ", f" is {assistant_id} ").replace(" are we ", f" are they ({user_id} and {assistant_id}) ").replace(" have i ", f" has {user_id} ").replace(" have you ", f" has {assistant_id} ")
            query = query.replace(" you ", f" {assistant_id} ").replace(" your ", f" {assistant_id}'s ").replace(" you're ", f" {assistant_id} is ").replace(" you've ", f" {assistant_id} has ").replace(" you'll ", f" {assistant_id} will ")
            query = query.replace(" i ", f" {user_id} ").replace(" me ", f" {user_id} ").replace(" my ", f" {user_id}'s ").replace(" i'm ", f" {user_id} is ").replace(" i've ", f" {user_id} has ").replace(" i'll ", f" {user_id} will ")
            query = query.replace(" we'll ", f" they ({user_id} and {assistant_id}) will ").replace(" we ", f" they ({user_id} and {assistant_id}) ").replace(" us ", f" them ({user_id} and {assistant_id}) ").replace(" our ", f" their ({user_id}'s and {assistant_id}'s) ")
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
                retry_times -= 1
                if retry_times == 0:
                    raise TransformQueryError(f"Error transforming query: {e}")

    def retrieve(
            self,
            query: str = None,
            transform_query_with_llm: bool = False,
            user_id: str = None,
            assistant_id: str = None
    ) -> List[NodeWithScore]:
        """
        Retrieve local and global context nodes from the graph store.

        Args:
            query (str, optional):
                The query string to retrieve. Defaults to None.
            transform_query_with_llm (bool, optional):
                Whether to transform the query with LLM. Defaults to False.
            user_id (str, optional):
                The user ID (name, nickname, or alias). Defaults to None.
            assistant_id (str, optional):
                The assistant ID (name, nickname, or alias). Defaults to None.

        Returns:
            List[NodeWithScore]:
                Final context nodes aggregated from local and global retrievals.
        """

        import time
        start = time.time()

        coversation_history = self.cache_chat_store.get_chat_history(user_id, en_translate=False)

        # Transform the query
        transformed_query = self.query_transformation(
            query,
            user_id=user_id,
            assistant_id=assistant_id,
            coversation_history=coversation_history,
            use_llm=transform_query_with_llm
        )
        end_transform = time.time()

        print(f"Total transformation time: {end_transform - start}")

        print(f"Transformed query: {transformed_query}")

        self.retriever = self.graph_store.as_retriever(sub_retrievers=[self.sub_retriever])
        retrieved_nodes = self.retriever.retrieve(transformed_query)

        return retrieved_nodes
    
    async def async_retrieve(
            self,
            query: str = None,
            transform_query_with_llm: bool = False,
            user_id: str = None,
            assistant_id: str = None
    ) -> List[NodeWithScore]:
        """
        Retrieve local and global context nodes from the graph store asynchronously.

        Args:
            query (str, optional):
                The query string to retrieve. Defaults to None.
            transform_query_with_llm (bool, optional):
                Whether to transform the query with LLM. Defaults to False.
            user_id (str, optional):
                The user ID (name, nickname, or alias). Defaults to None.
            assistant_id (str, optional):
                The assistant ID (name, nickname, or alias). Defaults to None.

        Returns:
            List[NodeWithScore]:
                Final context nodes aggregated from local and global retrievals.
        """
        import asyncio
        import nest_asyncio

        original_loop = asyncio.get_event_loop()
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        asyncio.set_event_loop(loop)

        retrieved_nodes = self.retrieve(
            query=query,
            transform_query_with_llm=transform_query_with_llm,
            user_id=user_id,
            assistant_id=assistant_id
        )
        
        asyncio.set_event_loop(original_loop)
        
        return retrieved_nodes
