from collections import OrderedDict
from typing import Dict, List, Tuple
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from llama_index.core.graph_stores.types import LabelledNode, KG_SOURCE_REL, TRIPLET_SOURCE_KEY
from llama_index.core.retrievers import VectorContextRetriever
from llama_index.core.schema import QueryBundle

from chatbot.core.graph_stores import CustomFalkorDBPropertyGraphStore
from chatbot.core.retriever.triplet_data import EntityInfo, RelationshipInfo, TextChunkInfo


class LocalContextRetriever(VectorContextRetriever):
    """
    A custom retriever that uses a vector store to retrieve local context from a graph store.
    This class modifies the retrieve_from_graph method to handle the retrieved nodes with multiple source (chunk) IDs.
    """

    debug: bool = False # Set to True to print debug logs
    selected_entities: List[str] = [] # List of selected entities from both global and local search
    _graph_store: CustomFalkorDBPropertyGraphStore = None # Change from PropertyGraphStore to CustomFalkorDBPropertyGraphStore

    def get_selected_entities(self):
        """Return the list of selected entities from local search."""
        return self.selected_entities
    
    def local_search(self, list_queries: List[str]) -> Tuple[Dict[str, EntityInfo], List[RelationshipInfo], Dict[str, TextChunkInfo]]:
        """
        Retrieve nodes from the graph store using local search.

        Args:
            list_query_bundle (List[str]):
                A list of subqueries to search for.

        Returns:
            Tuple[Dict[str, EntityInfo], List[RelationshipInfo], Dict[str, TextChunkInfo]]:
                A tuple containing the ordered selected entities, relationships, and text chunk information.
        """

        if len(list_queries) == 0:
            return {}, [], {}

        search_results = []
        selected_entity_nodes = []
        scores = []
        seen_entities = set()

        start = time.time()
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.search_process, subquery) for subquery in list_queries]
            for future in as_completed(futures):
                search_results.append(future.result())

        for entities, llama_entity_nodes, entity_node_scores in search_results:
            for entity, node, score in zip(entities, llama_entity_nodes, entity_node_scores):
                if entity not in seen_entities:
                    seen_entities.add(entity)
                    selected_entity_nodes.append(node)
                    scores.append(score)
        query_entities_time = time.time() - start

        # Get triplets for selected entities
        triplets = self._graph_store.get_rel_map(
            selected_entity_nodes, depth=self._path_depth, ignore_rels=[KG_SOURCE_REL]
        )

        self.selected_entities = [node.id for node in selected_entity_nodes]

        selected_entity_info: Dict[str, EntityInfo] = {}
        relationship_info: List[RelationshipInfo] = []
        seen_relations = set()
        text_chunk_info: Dict[str, TextChunkInfo] = {}

        start = time.time()
        for triplet in triplets:
            source_entity = triplet[0]
            relation = triplet[1]
            target_entity = triplet[2]

            # Create information dictionaries for entities
            if source_entity.id not in selected_entity_info:
                entity_info = EntityInfo(
                    description=source_entity.properties["description"],
                    timestamp=source_entity.properties["created_timestamp"]
                )
                selected_entity_info[source_entity.id] = entity_info
            if target_entity.id not in selected_entity_info:
                entity_info = EntityInfo(
                    description=target_entity.properties["description"],
                    timestamp=target_entity.properties["created_timestamp"]
                )
                selected_entity_info[target_entity.id] = entity_info
            
            # Create information list for relationships
            relation_description = relation.properties["description"]
            if relation_description not in seen_relations:
                rel_info = RelationshipInfo(
                    source_entity=relation.source_id,
                    target_entity=relation.target_id,
                    relationships=relation_description,
                    relationship_type=relation.label,
                    chunk_source_id=relation.properties.get(TRIPLET_SOURCE_KEY, []),
                    timestamp=relation.properties["created_timestamp"]
                )
                relationship_info.append(rel_info)
                seen_relations.add(relation_description)

            # Create list of entities and relations for each text chunk
            for source_id in source_entity.properties.get(TRIPLET_SOURCE_KEY, []):
                if source_id not in text_chunk_info:
                    text_chunk_info[source_id] = TextChunkInfo()
                if source_entity.id not in text_chunk_info[source_id].entities:
                    text_chunk_info[source_id].entities.append(source_entity.id)

            for source_id in target_entity.properties.get(TRIPLET_SOURCE_KEY, []):
                if source_id not in text_chunk_info:
                    text_chunk_info[source_id] = TextChunkInfo()
                if target_entity.id not in text_chunk_info[source_id].entities:
                    text_chunk_info[source_id].entities.append(target_entity.id)

            relation_source_id = relation.properties.get(TRIPLET_SOURCE_KEY, [])
            if relation_source_id:
                if relation_description not in text_chunk_info[relation_source_id[0]].relations:
                    text_chunk_info[relation_source_id[0]].relations.append(relation_description)

        processing_triplets_time = time.time() - start

        # Sort selected_entity_info based on self.selected_entities
        ordered_selected_entity_info = OrderedDict()
        for entity_id in self.selected_entities:
            # Add only the selected entities to the ordered_selected_entity_info first
            if entity_id in selected_entity_info:
                ordered_selected_entity_info[entity_id] = selected_entity_info[entity_id]
        for entity_id, entity_info in selected_entity_info.items():
            # Add the remaining entities to the ordered_selected_entity_info
            if entity_id not in ordered_selected_entity_info:
                ordered_selected_entity_info[entity_id] = EntityInfo(
                    description=entity_info.description, timestamp=entity_info.timestamp
                )

        if self.debug:
            print(f"List of local entities: {list(seen_entities)}")
            print(f"Time taken for local entities search: {query_entities_time} seconds")
            print(f"Time taken for processing triplets: {processing_triplets_time} seconds")

        return ordered_selected_entity_info, relationship_info, text_chunk_info
    
    def search_process(self, subquery: str) -> Tuple[List[str], List[LabelledNode], List[float]]:
        """
        A process to search for entities in the graph store.

        Args:
            subquery (str):
                The subquery to search for.

        Returns:
            Tuple[List[str], List[LabelledNode], List[float]]:
                A tuple containing the entity names, entity nodes, and similarity scores.
        """
        query_bundle = QueryBundle(query_str=subquery)
        entities, llama_entity_nodes, entity_node_scores = self.retrieve_entities(query_bundle)
        return entities, llama_entity_nodes, entity_node_scores
    
    def retrieve_entities(self, query_bundle: QueryBundle) -> Tuple[List[str], List[LabelledNode], List[float]]:
        """
        Retrieve relevant entities from a query bundle.

        Args:
            query_bundle (QueryBundle):
                The query bundle to use for retrieval.

        Returns:
            Tuple[List[str], List[LabelledNode], List[float]]:
                A tuple containing the entity names, entity nodes, and similarity scores.
        """
        entity_names = []
        llama_nodes = []
        node_scores = []

        vector_store_query = self._get_vector_store_query(query_bundle)
        result = self._graph_store.entity_search(vector_store_query)
        if len(result) != 2:
            raise ValueError("No nodes returned by vector_query")
        
        entities, scores = result
        for entity, score in zip(entities, scores):
            if score > self._similarity_score:
                entity_names.append(entity.name)
                llama_nodes.append(entity)
                node_scores.append(score)
        
        return entity_names, llama_nodes, node_scores
    