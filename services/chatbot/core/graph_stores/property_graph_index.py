import asyncio
import json
from typing import Any, List, Optional, Sequence, Type
from tqdm import tqdm

from datetime import datetime
from geopy.geocoders import Nominatim
import pytz
from timezonefinder import TimezoneFinder

from llama_index.core import PropertyGraphIndex
from llama_index.core.callbacks import CallbackManager
from llama_index.core.data_structs import IndexLPG
from llama_index.core.embeddings.utils import EmbedType
from llama_index.core.ingestion.pipeline import arun_transformations, run_transformations
from llama_index.core.llms import LLM
from llama_index.core.schema import BaseNode, Document, MetadataMode, TransformComponent
from llama_index.core.settings import Settings
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.graph_stores.types import (
    LabelledNode,
    Relation,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    TRIPLET_SOURCE_KEY
)
from llama_index.core.vector_stores.types import BasePydanticVectorStore

from chatbot.prompt.graph.entity_deduplication import DEDUPLICATE_ENTITIES_PROMPT_TEMPLATE
from chatbot.prompt.graph.description_summarization import (
    DESCRIPTION_SUMMARIZE_PROMPT_TEMPLATE, 
    CHECK_DESCRIPTION_INCLUSION_PROMPT_TEMPLATE
)
from chatbot.core.graph_stores.exceptions import (
    GraphEntityDeduplicationError, GraphGetCurrentTimeError, GraphGetTimezoneError,
    GraphProcessExtractedTripletsError, GraphDeduplicateLlamaNodesError, GraphAddTimestampsError,
    GraphEmbedNodesError, GraphUpsertEntitiesAndRelationsError
)
from chatbot.core.graph_stores.falkordb_property_graph_store import CustomFalkorDBPropertyGraphStore
from chatbot.core.graph_stores.geographical_data import GeographicalData


class CustomPropertyGraphIndex(PropertyGraphIndex):
    """
    This is a custom class to handle the extraction of triplets from nodes and insert them into the graph store.
    It is implemented to modify indexing stage based on the structure of Microsoft GraphRAG and LightRAG architectures.
    """

    def __init__(
        self,
        nodes: Optional[Sequence[BaseNode]] = None,
        llm: Optional[LLM] = None,
        kg_extractors: Optional[List[TransformComponent]] = None,
        property_graph_store: Optional[CustomFalkorDBPropertyGraphStore] = None,
        geographical_data: Optional[GeographicalData] = None,
        description_summarization_max_retries: int = 3,
        # vector related params
        vector_store: Optional[BasePydanticVectorStore] = None,
        use_async: bool = True,
        embed_model: Optional[EmbedType] = None,
        embed_kg_nodes: bool = True,
        # parent class params
        callback_manager: Optional[CallbackManager] = None,
        transformations: Optional[List[TransformComponent]] = None,
        storage_context: Optional[StorageContext] = None,
        time_aware: bool = False,
        show_progress: bool = False,
        **kwargs: Any,
    ):
        self._property_graph_store: CustomFalkorDBPropertyGraphStore = property_graph_store
        self.geographical_data: Optional[GeographicalData] = self.get_city_timezone(geographical_data) if geographical_data else None
        self.description_summarization_max_retries: int = description_summarization_max_retries
        self.time_aware: bool = time_aware
        super().__init__(
            nodes=nodes,
            llm=llm,
            property_graph_store=property_graph_store,
            kg_extractors=kg_extractors,
            vector_store=vector_store,
            use_async=use_async,
            embed_model=embed_model,
            embed_kg_nodes=embed_kg_nodes,
            callback_manager=callback_manager,
            transformations=transformations,
            storage_context=storage_context,
            show_progress=show_progress,
            **kwargs,
        )
    
    @classmethod
    def from_documents(
        cls: Type["CustomPropertyGraphIndex"],
        documents: Sequence[Document],
        property_graph_store: CustomFalkorDBPropertyGraphStore,
        vector_store: Optional[BasePydanticVectorStore] = None,
        llm: Optional[LLM] = None,
        kg_extractors: Optional[List[TransformComponent]] = None,
        use_async: bool = True,
        embed_model: Optional[EmbedType] = None,
        embed_kg_nodes: bool = True,
        storage_context: Optional[StorageContext] = None,
        show_progress: bool = False,
        callback_manager: Optional[CallbackManager] = None,
        transformations: Optional[List[TransformComponent]] = None,
        geographical_data: Optional[GeographicalData] = None,
        description_summarization_max_retries: int = 3,
        time_aware: bool = False,
        **kwargs: Any,
    ) -> "CustomPropertyGraphIndex":
        """
        Create index of the entities extracted from the documents and store them in the graph store.

        Args:
            documents (Optional[Sequence[BaseDocument]]): List of documents to
                build the index from.
        """

        storage_context = storage_context or StorageContext.from_defaults()
        docstore = storage_context.docstore
        callback_manager = callback_manager or Settings.callback_manager
        transformations = transformations or Settings.transformations

        with callback_manager.as_trace("index_construction"):
            for doc in documents:
                docstore.set_document_hash(doc.get_doc_id(), doc.hash)

            # Chunking the document using default Chunk Splitter (size=1024, overlap=200) 
            # ignore since the max length of the document is 600
            nodes = run_transformations(
                documents,  # type: ignore
                transformations,
                show_progress=show_progress,
                **kwargs,
            )

            # Return the initialized class instance with the extracted nodes
            return cls(
                nodes=nodes,
                property_graph_store=property_graph_store,
                vector_store=vector_store,
                llm=llm,
                kg_extractors=kg_extractors,
                use_async=use_async,
                embed_model=embed_model,
                embed_kg_nodes=embed_kg_nodes,
                storage_context=storage_context,
                callback_manager=callback_manager,
                transformations=transformations,
                geographical_data=geographical_data,
                description_summarization_max_retries=description_summarization_max_retries,
                time_aware=time_aware,
                show_progress=show_progress,
                **kwargs,
            )
    
    @classmethod
    def from_existing(
        cls: Type["CustomPropertyGraphIndex"],
        property_graph_store: CustomFalkorDBPropertyGraphStore,
        vector_store: Optional[BasePydanticVectorStore] = None,
        geographical_data: Optional[GeographicalData] = None,
        description_summarization_max_retries: int = 3,
        time_aware: bool = False,
        # general params
        llm: Optional[LLM] = None,
        kg_extractors: Optional[List[TransformComponent]] = None,
        # vector related params
        use_async: bool = True,
        embed_model: Optional[EmbedType] = None,
        embed_kg_nodes: bool = True,
        # parent class params
        callback_manager: Optional[CallbackManager] = None,
        transformations: Optional[List[TransformComponent]] = None,
        storage_context: Optional[StorageContext] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> "PropertyGraphIndex":
        """Create an index from an existing property graph store (and optional vector store)."""

        # Return the initialized class instance
        return cls(
            nodes=[],  # no nodes to insert
            property_graph_store=property_graph_store,
            vector_store=vector_store,
            llm=llm,
            kg_extractors=kg_extractors,
            use_async=use_async,
            embed_model=embed_model,
            embed_kg_nodes=embed_kg_nodes,
            callback_manager=callback_manager,
            transformations=transformations,
            storage_context=storage_context,
            geographical_data=geographical_data,
            description_summarization_max_retries=description_summarization_max_retries,
            time_aware=time_aware,
            show_progress=show_progress,
            **kwargs,
        )
    
    @property
    def property_graph_store(self) -> CustomFalkorDBPropertyGraphStore:
        """Get the FalkorDB property graph store."""
        return self._property_graph_store

    @property_graph_store.setter
    def property_graph_store(self, value) -> None:
        """Set the FalkorDB property graph store."""
        self._property_graph_store = value

    def _build_index_from_nodes(
        self, nodes: Optional[Sequence[BaseNode]], **build_kwargs: Any
    ) -> IndexLPG:
        """Build index from nodes."""
        nodes = self._insert_nodes(nodes or [], **build_kwargs)

        # this isn't really used or needed
        return IndexLPG()

    def _insert_nodes(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        """
        Insert parsed document nodes into the graph store.
        Key points:
        - Extract triplets from nodes.
        - Insert triplets into the graph store.
        - For duplicate nodes, add the source id (if not already present) and summarize the description to the existing node.
        - Embed the description of the entity node for further retrieval.
        - Deduplicate entities based on the similarity threshold and decision from the LLM.

        Args:
            nodes (Sequence[BaseNode]): List of nodes to insert.

        Returns:
            Sequence[BaseNode]: List of nodes inserted.
        """

        if len(nodes) == 0:
            return nodes

        # Run transformations on the document nodes to extract triplets
        if self._use_async:
            nodes = asyncio.run(
                arun_transformations(
                    nodes, self._kg_extractors, show_progress=self._show_progress
                )
            )
        else:
            nodes = run_transformations(
                nodes, self._kg_extractors, show_progress=self._show_progress
            )

        # Ensure all nodes have nodes and/or relations in metadata
        assert all(
            node.metadata.get(KG_NODES_KEY) is not None
            or node.metadata.get(KG_RELATIONS_KEY) is not None
            for node in nodes
        )

        try:
            kg_nodes_to_insert: List[LabelledNode] = []
            kg_rels_to_insert: List[Relation] = []
            for node in nodes:
                # Remove nodes and relations from metadata
                kg_nodes = node.metadata.pop(KG_NODES_KEY, [])
                kg_rels = node.metadata.pop(KG_RELATIONS_KEY, [])

                # Add source id to properties (both entities and relations) 
                # and add created timestamp of the relations only
                for kg_node in kg_nodes:
                    kg_node.properties[TRIPLET_SOURCE_KEY] = [node.id_]

                    if self.time_aware:
                        if self.geographical_data:
                            kg_node.properties["created_timestamp"] = self.get_city_timestamp(self.geographical_data)
                        else:
                            kg_node.properties["created_timestamp"] = ""

                for kg_rel in kg_rels:
                    kg_rel.properties[TRIPLET_SOURCE_KEY] = [node.id_]

                    if self.time_aware:
                        if self.geographical_data:
                            kg_rel.properties["created_timestamp"] = self.get_city_timestamp(self.geographical_data)
                        else:
                            kg_rel.properties["created_timestamp"] = ""

                # Add nodes and relations to insert lists
                kg_nodes_to_insert.extend(kg_nodes)
                kg_rels_to_insert.extend(kg_rels)

            # Filter out duplicate kg nodes
            kg_node_ids = {node.id for node in kg_nodes_to_insert}
            existing_kg_nodes = self.property_graph_store.get(ids=list(kg_node_ids))
            existing_kg_node_ids = {node.id for node in existing_kg_nodes}

            _kg_nodes_to_insert = []
            processed_node_ids = set()
            iterable = tqdm(kg_nodes_to_insert, desc="Processing the extracted triplets") if self._show_progress else kg_nodes_to_insert
            for node in iterable:
                # If the node does not exist in the existing nodes (from both the graph store and the nodes 
                # created in the current batch), add it to the list of nodes to insert
                if node.id not in existing_kg_node_ids:
                    _kg_nodes_to_insert.append(node)
                    existing_kg_node_ids.add(node.id)
                    existing_kg_nodes.append(node)      # Add the node the existing nodes list to insert the source id if 
                                                        # the entity appears again in the same batch

                    processed_node_ids.add(node.id)     # Add the node id to the processed node ids set to avoid duplication 
                                                        # in the future in case the entity appears again in the same batch
                else:
                    # If the node already exists, add the source id to the existing node
                    existing_node = next(
                        (n for n in existing_kg_nodes if n.id == node.id), None
                    )
                    if existing_node is not None:
                        # Ensure the TRIPLET_SOURCE_KEY list exists
                        if TRIPLET_SOURCE_KEY not in existing_node.properties:
                            existing_node.properties[TRIPLET_SOURCE_KEY] = []

                        # Add the source id to the existing node if it does not already exist
                        for source_id in node.properties[TRIPLET_SOURCE_KEY]:
                            if source_id not in existing_node.properties[TRIPLET_SOURCE_KEY]:
                                existing_node.properties[TRIPLET_SOURCE_KEY].append(source_id)

                        # Combine the description of both existing and new nodes
                        existing_node.properties["description"] = self.summarize_description(
                            current_desc=existing_node.properties.get("description", ""),
                            new_desc=node.properties.get("description", "")
                        )

                        # Update timestamp of the existing node
                        existing_node.properties["created_timestamp"] = self.get_city_timestamp(self.geographical_data)
                        
                        # Add the existing node to the list of nodes to insert if it has not been processed before; 
                        # otherwise, replace the existing node in the list
                        if node.id not in processed_node_ids:
                            processed_node_ids.add(node.id)
                        else:
                            # If the node has been processed before, take out the node in `_kg_nodes_to_insert` list to avoid duplication
                            _kg_nodes_to_insert = [n for n in _kg_nodes_to_insert if n.id != node.id]
                        _kg_nodes_to_insert.append(existing_node)

            kg_nodes_to_insert = _kg_nodes_to_insert
        except Exception as e:
            raise GraphProcessExtractedTripletsError(f"Error processing extracted triplets: {e}")

        try:
            # Filter out duplicate llama nodes
            existing_nodes = self.property_graph_store.get_llama_nodes(
                [node.id_ for node in nodes]
            )
            existing_node_hashes = {node.hash for node in existing_nodes}
            _nodes = []
            for node in nodes:
                if node.hash not in existing_node_hashes:
                    _nodes.append(node)
                else:
                    print("Node already exists: ", node.get_content(metadata_mode=MetadataMode.ALL))
            nodes = _nodes
        except Exception as e:
            raise GraphDeduplicateLlamaNodesError(f"Error deduplicating Llama nodes: {e}")
        
        # Add created_at timestamp to the llama nodes
        try:
            for node in nodes:
                node.metadata["created_timestamp"] = self.get_city_timestamp(self.geographical_data)
        except Exception as e:
            raise GraphAddTimestampsError(f"Error adding timestamps to nodes: {e}")

        try:
            # Embed nodes (if needed)
            if self._embed_kg_nodes:
                # Embed llama-index nodes
                node_texts = [
                    node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes
                ]

                if self._use_async:
                    embeddings = asyncio.run(
                        self._embed_model.aget_text_embedding_batch(
                            node_texts, show_progress=self._show_progress
                        )
                    )
                else:
                    embeddings = self._embed_model.get_text_embedding_batch(
                        node_texts, show_progress=self._show_progress
                    )

                for node, embedding in zip(nodes, embeddings):
                    node.embedding = embedding

                # Embed knowledge graph nodes
                kg_node_texts = [
                    str(kg_node.properties["description"]) if "description" in kg_node.properties else str(kg_node.model_dump()["name"])
                    for kg_node in kg_nodes_to_insert
                ]

                if self._use_async:
                    kg_embeddings = asyncio.run(
                        self._embed_model.aget_text_embedding_batch(
                            kg_node_texts, show_progress=self._show_progress
                        )
                    )
                else:
                    kg_embeddings = self._embed_model.get_text_embedding_batch(
                        kg_node_texts, show_progress=self._show_progress,
                    )

                for kg_node, embedding in zip(kg_nodes_to_insert, kg_embeddings):
                    kg_node.embedding = embedding

                # Embed knowledge graph entity names in standard format
                entity_names = [
                    str(kg_node.model_dump()["name"]).lower().replace("_", " ").replace("-", " ")
                    for kg_node in kg_nodes_to_insert
                ]

                if self._use_async:
                    entity_name_embeddings = asyncio.run(
                        self._embed_model.aget_text_embedding_batch(
                            entity_names, show_progress=self._show_progress
                        )
                    )
                else:
                    entity_name_embeddings = self._embed_model.get_text_embedding_batch(
                        entity_names, show_progress=self._show_progress,
                    )

                # Embed knowledge graph relations
                kg_rels_texts = [
                    str(kg_rel.properties["description"]) if "description" in kg_rel.properties else str(kg_rel.model_dump()["label"]).lower().replace("_", " ")
                    for kg_rel in kg_rels_to_insert
                ]
                
                if self._use_async:
                    rel_embeddings = asyncio.run(
                        self._embed_model.aget_text_embedding_batch(
                            kg_rels_texts, show_progress=self._show_progress
                        )
                    )
                else:
                    rel_embeddings = self._embed_model.get_text_embedding_batch(
                        kg_rels_texts, show_progress=self._show_progress,
                    )
        except Exception as e:
            raise GraphEmbedNodesError(f"Error embedding nodes: {e}")

        try:
            # If graph store doesn't support vectors, or the vector index was provided, use it
            if self.vector_store is not None and len(kg_nodes_to_insert) > 0:
                self._insert_nodes_to_vector_index(kg_nodes_to_insert)

            if len(nodes) > 0:
                self.property_graph_store.upsert_llama_nodes(nodes)

            if len(kg_nodes_to_insert) > 0:
                self.property_graph_store.upsert_nodes(
                    nodes=kg_nodes_to_insert, entity_name_embeddings=entity_name_embeddings, show_progress=self._show_progress
                )

            # Important: upsert relations after nodes
            if len(kg_rels_to_insert) > 0:
                self.property_graph_store.upsert_relations(
                    relations=kg_rels_to_insert, embeddings=rel_embeddings, show_progress=self._show_progress
                )
        except Exception as e:
            raise GraphUpsertEntitiesAndRelationsError(f"Error upserting entities and relations: {e}")
        
        # Deduplicate entities
        self.deduplicate_entities(entity_nodes=kg_nodes_to_insert)
            
        # Refresh schema if needed
        if self.property_graph_store.supports_structured_queries:
            self.property_graph_store.get_schema(refresh=True)

        return nodes

    def deduplicate_entities(self, entity_nodes: List[LabelledNode], similarity_threshold: float = 0.85) -> None:
        """
        Deduplicate entities in a list of lists provided as input.
        Method to deduplicate entities are as follows:
        - Find entities with similar names based on the similarity threshold.
        - Use the LLM to decide whether to merge the entities or not.
        
        Args:
            entity_nodes (List[LabelledNode]): A list of entity nodes to deduplicate.
            similarity_threshold (float): The similarity threshold to consider two entities as duplicates.
        """
        try:
            # Entity deduplication
            duplicate_entities_list = []
            entity_properties_dict = {}
            iterable = tqdm(entity_nodes, desc="Finding duplicate entities") if self._show_progress else entity_nodes
            for entity in iterable:
                data = self.property_graph_store.structured_query(
                    """
                    MATCH (e1:__Entity__ {id: $entity_id})
                    MATCH (e2:__Entity__)
                    WHERE e1.id <> e2.id AND e1.name_embedding IS NOT NULL AND e2.name_embedding IS NOT NULL
                    WITH e1, e2, 1 - vec.cosineDistance(e1.name_embedding, e2.name_embedding) AS similarity_score
                    WHERE similarity_score > $similarity_threshold
                    RETURN e2.id AS similar_entity_id, 
                        similarity_score, 
                        e2{.* , name_embedding: Null, embedding: Null, name: Null, id: Null} AS entity_properties
                    """,
                    param_map={
                        "entity_id": entity.id,
                        "similarity_threshold": similarity_threshold
                    },
                )
                if data:
                    # Add list of duplicate entities to the dictionary
                    duplicate_entities_list.append([entity.id] + [record["similar_entity_id"] for record in data])
                    # Add entity properties to the dictionary
                    entity_properties_dict[entity.id] = entity.properties
                    for record in data:
                        entity_properties_dict[record["similar_entity_id"]] = record["entity_properties"]
        except Exception as e:
            raise GraphEntityDeduplicationError(f"Error finding duplicate entities: {e}")

        if not duplicate_entities_list:
            print("No duplicate entities found.")
            return

        response_text = self._llm.complete(prompt=DEDUPLICATE_ENTITIES_PROMPT_TEMPLATE.format(
            input_entity_lists=duplicate_entities_list
        )).text

        try:
            response = json.loads(response_text)
        except json.JSONDecodeError as e:
            raise GraphEntityDeduplicationError(f"Error parsing JSON response: {e}")

        removed_entities = []
        if self._show_progress:
            iterable = tqdm(zip(response, duplicate_entities_list), 
                            total=len(response), 
                            desc="Merging duplicate entities")
        else:
            iterable = zip(response, duplicate_entities_list)
        try:
            for result, duplicate_entities in iterable:
                processed_entities = []

                if result["confirm"] == "merge":
                    merged_entity = result["final_entity"]
                    other_entities = [entity for entity in duplicate_entities if entity != merged_entity]

                    # Add the merged entity to the list of processed entities 
                    # to avoid the entity being processed again
                    for entity in other_entities:
                        if entity not in removed_entities:
                            processed_entities.append(entity)
                            removed_entities.append(entity)

                    other_entities = processed_entities

                    if other_entities:
                        # Update the properties of the merged entity
                        for entity in other_entities:
                            source_ids = entity_properties_dict[entity].get(TRIPLET_SOURCE_KEY, [])
                            merged_source_ids = entity_properties_dict[merged_entity].get(TRIPLET_SOURCE_KEY, [])
                            for source_id in source_ids:
                                if source_id not in merged_source_ids:
                                    merged_source_ids.append(source_id)

                            other_description = entity_properties_dict[entity].get("description", "")
                            merged_description = entity_properties_dict[merged_entity].get("description", "")
                            merged_description = self.summarize_description(
                                current_desc=merged_description, new_desc=other_description
                            )

                        # Collect the outgoing relationship types of the duplicate entities
                        outgoing_rel_types_query = """
                        UNWIND $other_entities AS duplicate_id
                        MATCH (e:__Entity__ {id: duplicate_id})-[r]->(target)
                        RETURN DISTINCT type(r) AS rel_type
                        """
                        outgoing_rel_types = self.property_graph_store.structured_query(
                            query=outgoing_rel_types_query,
                            param_map={"other_entities": other_entities}
                        )

                        # Collect the incoming relationship types of the duplicate entities
                        incoming_rel_types_query = """
                        UNWIND $other_entities AS duplicate_id
                        MATCH (source)-[r]->(e:__Entity__ {id: duplicate_id})
                        RETURN DISTINCT type(r) AS rel_type
                        """
                        incoming_rel_types = self.property_graph_store.structured_query(
                            query=incoming_rel_types_query,
                            param_map={"other_entities": other_entities}
                        )

                        # Merge outgoing relationships into the merged entity
                        for rel_type in outgoing_rel_types:
                            rel_type = rel_type["rel_type"]
                            query = f"""
                            UNWIND $other_entities AS duplicate_id
                            MATCH (e:__Entity__ {{id: duplicate_id}})-[r:`{rel_type}`]->(target)
                            MATCH (merged:__Entity__) WHERE merged.id = $merged_entity
                            MERGE (merged)-[new_r:`{rel_type}`]->(target)
                            SET new_r = properties(r)
                            DELETE r
                            """
                            self.property_graph_store.structured_query(
                                query=query,
                                param_map={
                                    "merged_entity": merged_entity,
                                    "other_entities": other_entities
                                }
                            )

                        # Merge incoming relationships into the merged entity
                        for rel_type in incoming_rel_types:
                            rel_type = rel_type["rel_type"]
                            query = f"""
                            UNWIND $other_entities AS duplicate_id
                            MATCH (source)-[r:`{rel_type}`]->(e:__Entity__ {{id: duplicate_id}})
                            MATCH (merged:__Entity__) WHERE merged.id = $merged_entity
                            MERGE (source)-[new_r:`{rel_type}`]->(merged)
                            SET new_r = properties(r)
                            DELETE r
                            """
                            self.property_graph_store.structured_query(
                                query,
                                {"merged_entity": merged_entity, "other_entities": other_entities}
                            )
                            
                        # Update the triplet_source_id and description properties for the merged entity
                        self.property_graph_store.structured_query(
                            """
                            MATCH (merged:__Entity__) WHERE merged.id = $merged_entity
                            SET merged.triplet_source_id = $merged_source_ids,
                                merged.description = $merged_description
                            """,
                            param_map={
                                "merged_entity": merged_entity,
                                "merged_source_ids": merged_source_ids,
                                "merged_description": merged_description
                            },
                        )

                        # Delete duplicate entities
                        self.property_graph_store.structured_query(
                            """
                            UNWIND $other_entities AS duplicate_id
                            MATCH (e:__Entity__ {id: duplicate_id})
                            DELETE e
                            """,
                            param_map={"other_entities": other_entities},
                        )
        except Exception as e:
            raise GraphEntityDeduplicationError(f"Error merging duplicate entities: {e}")

    def summarize_description(self, current_desc: str, new_desc: str) -> str:
        """
        Summarize the description of the new node and add it to the existing node.
        
        Args:
            current_desc (str): The current description of the existing node.
            new_desc (str): The new description of the node to be added.

        Returns:
            str: The summarized description of the existing node.
        """
        max_iterations = self.description_summarization_max_retries
        status_check = "NO"
        while status_check == "NO" and max_iterations > 0:
            # Check if the new description is included in the existing description
            response = self._llm.complete(prompt=CHECK_DESCRIPTION_INCLUSION_PROMPT_TEMPLATE.format(
                old_description=current_desc,
                new_description=new_desc
            )).text
            json_result = json.loads(response)

            status_check = json_result["result"]
            if status_check == "NO":
                # Add new description and summarize the existing description
                summarized_desc = self._llm.complete(prompt=DESCRIPTION_SUMMARIZE_PROMPT_TEMPLATE.format(
                    old_description=current_desc,
                    new_description=new_desc,
                    missing_information=json_result["missing_information"]
                )).text
                current_desc = summarized_desc
            
            max_iterations -= 1
        
        return current_desc
    
    def get_city_timestamp(self, data: GeographicalData) -> str:
        """
        Get the current timestamp in the given country and city.
        
        Args:
            data (GeographicalData): The geographical data containing the city and country.

        Returns:
            str: The current stamp in the given country and city.
        """
        try:
            tz = pytz.timezone(data.timezone)
            return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            raise GraphGetCurrentTimeError(f"Error getting current time: {e}")

    def get_city_timezone(self, data: GeographicalData) -> GeographicalData:
        """
        Get the timezone of the given city and country.

        Args:
            data (GeographicalData): The geographical data containing the city and country.

        Returns:
            GeographicalData: The geographical data with the timezone.
        """
        try:
            # Get coordinates from city and country
            geolocator = Nominatim(user_agent="geoapi")
            location = geolocator.geocode(f"{data.city}, {data.country}")

            if location:
                # Get timezone from coordinates
                tf = TimezoneFinder()
                data.timezone = tf.timezone_at(lng=location.longitude, lat=location.latitude)
            else:
                raise GraphGetTimezoneError(f"Could not find location for {data.city}, {data.country}")
             
            return data
        except Exception as e:
            raise GraphGetTimezoneError(f"Error getting timezone: {e}")
