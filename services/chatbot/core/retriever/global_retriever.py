from typing import Dict, List, Tuple
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from llama_index.core.graph_stores.types import Relation, TRIPLET_SOURCE_KEY
from llama_index.core.retrievers import VectorContextRetriever
from llama_index.core.schema import QueryBundle

from chatbot.core.graph_stores import CustomFalkorDBPropertyGraphStore
from chatbot.core.retriever.triplet_data import EntityInfo, RelationshipInfo, TextChunkInfo


class GlobalContextRetriever(VectorContextRetriever):
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
    
    def global_search(self, list_queries: List[str]) -> Tuple[Dict[str, EntityInfo], List[RelationshipInfo], Dict[str, TextChunkInfo]]:
        """
        Retrieve nodes from the graph store using global search.

        Args:
            list_query_bundle (List[str]):
                A list of subqueries to search for.

        Returns:
            Tuple[Dict[str, EntityInfo], List[RelationshipInfo], Dict[str, TextChunkInfo]]:
                A tuple containing the selected entities, relationships, and text chunk information.
        """

        if len(list_queries) == 0:
            return {}, [], {}

        search_results = []
        selected_relation_nodes = []
        scores = []
        seen_relations = set()

        start = time.time()
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.search_process, subquery) for subquery in list_queries]
            for future in as_completed(futures):
                search_results.append(future.result())

        for relations, llama_relation_nodes, relation_node_scores in search_results:
            for relation, node, score in zip(relations, llama_relation_nodes, relation_node_scores):
                if relation not in seen_relations:
                    seen_relations.add(relation)
                    selected_relation_nodes.append(node)
                    scores.append(score)
        query_relations_time = time.time() - start

        selected_entity_info: Dict[str, EntityInfo] = {}
        relationship_info: List[RelationshipInfo] = []
        text_chunk_info: Dict[str, TextChunkInfo] = {}
        seen_entities = set()

        for relation_node in selected_relation_nodes:
            source_entity = relation_node.source_id
            target_entity = relation_node.target_id
            relationships = relation_node.properties["description"]
            label = relation_node.label
            chunk_source_id = relation_node.properties.get(TRIPLET_SOURCE_KEY, [])
            timestamp = relation_node.properties["created_timestamp"]

            if source_entity not in seen_entities:
                seen_entities.add(source_entity)
            if target_entity not in seen_entities:
                seen_entities.add(target_entity)

            if (source_entity, target_entity, relationships, timestamp) not in relationship_info:
                rel_info = RelationshipInfo(
                    source_entity=source_entity,
                    target_entity=target_entity,
                    relationships=relationships,
                    relationship_type=label,
                    chunk_source_id=chunk_source_id,
                    timestamp=timestamp
                )
                relationship_info.append(rel_info)

        start = time.time()

        data = self._graph_store.structured_query(
            """
            UNWIND $entity_ids AS entity_id
            MATCH (e)
            WHERE e.id = entity_id
            RETURN e.id AS entity_id,
                e.description AS description,
                e.triplet_source_id AS triplet_source_id,
                e.created_timestamp AS created_timestamp
            """,
            param_map={"entity_ids": list(seen_entities)},
        )

        for record in data:
            entity_id = record["entity_id"]
            description = record["description"]
            source_ids = record[TRIPLET_SOURCE_KEY]
            created_timestamp = record["created_timestamp"]
            selected_entity_info[entity_id] = EntityInfo(
                description=description,
                timestamp=created_timestamp
            )

            for source_id in source_ids:
                if source_id not in text_chunk_info:
                    text_chunk_info[source_id] = TextChunkInfo()
                text_chunk_info[source_id].entities.append(entity_id)

        for relation_node in selected_relation_nodes:
            if relation_node.properties["description"] not in text_chunk_info[relation_node.properties[TRIPLET_SOURCE_KEY][0]].relations:
                text_chunk_info[relation_node.properties[TRIPLET_SOURCE_KEY][0]].relations.append(relation_node.properties["description"])

        processing_relations_time = time.time() - start

        self.selected_entities = list(seen_entities)

        if self.debug:
            print(f"Time taken for global relations search: {query_relations_time:.2f} seconds")
            print(f"Time taken for processing relations: {processing_relations_time:.2f} seconds")
        
        return selected_entity_info, relationship_info, text_chunk_info
    
    def search_process(self, subquery: str) -> Tuple[List[str], List[Relation], List[float]]:
        """
        A process to search for relations in the graph store.

        Args:
            subquery (str):
                The subquery to search for.

        Returns:
            Tuple[List[str], List[LabelledNode], List[float]]:
                A tuple containing the relation names, relation nodes, and similarity scores.
        """
        query_bundle = QueryBundle(query_str=subquery)
        relations, llama_relation_nodes, relation_node_scores = self.retrieve_relations(query_bundle)
        return relations, llama_relation_nodes, relation_node_scores
                
    
    def retrieve_relations(self, query_bundle: QueryBundle) -> Tuple[List[str], List[Relation], List[float]]:
        """
        Retrieve relevant relations from a query bundle.

        Args:
            query_bundle (QueryBundle):
                The query bundle to use for retrieval.

        Returns:
            Tuple[List[str], List[Relation], List[float]]:
                A tuple containing the relation names, relation nodes, and similarity scores.
        """
        relationship_descriptions = []
        llama_nodes = []
        node_scores = []

        vector_store_query = self._get_vector_store_query(query_bundle)
        result = self._graph_store.relation_search(vector_store_query)
        if len(result) != 2:
            raise ValueError("No nodes returned by vector_query")
        
        relations, scores = result
        for relation, score in zip(relations, scores):
            if score > self._similarity_score:
                relationship_descriptions.append(relation.properties["description"])
                llama_nodes.append(relation)
                node_scores.append(score)
        
        return relationship_descriptions, llama_nodes, node_scores
    