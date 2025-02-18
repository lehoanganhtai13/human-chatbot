import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd
from graspologic.partition import hierarchical_leiden, HierarchicalCluster
from networkx import Graph
from tqdm import tqdm

from llama_index.core import PromptTemplate
from llama_index.graph_stores.falkordb import FalkorDBPropertyGraphStore
from llama_index.graph_stores.falkordb.falkordb_property_graph import remove_empty_values
from llama_index.core.graph_stores.types import (
    ChunkNode,
    EntityNode,
    LabelledNode,
    Relation,
    Triplet,
    VectorStoreQuery
)

from chatbot.prompt.graph.community_summarization import COMMUNITY_SUMMARIZE_PROMPT_TEMPLATE
from chatbot.core.graph_stores.exceptions import GraphUpsertNodesError, InitEntityRelationNodesError
from chatbot.core.model_clients import LLMCore


class CustomFalkorDBPropertyGraphStore(FalkorDBPropertyGraphStore):

    community_summary: Dict[int, List[str]] = {}
    entity_info: Dict [str, List[str]] = None
    nx_graph: Graph = None
    llm: LLMCore = None
    max_cluster_size: int = 10

    def entity_search(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> Tuple[List[LabelledNode], List[float]]:
        """
        This is the custom function to query the graph store with a vector store query to search for entities.
        The difference is that this function uses cosine similarity score to rank the results.
        """
        conditions = None
        if query.filters:
            conditions = [
                f"e.{filter.key} {filter.operator.value} {filter.value}"
                for filter in query.filters.filters
            ]
        filters = (
            f" {query.filters.condition.value} ".join(conditions).replace("==", "=")
            if conditions is not None
            else "1 = 1"
        )

        data = self.structured_query(
            f"""MATCH (e:`__Entity__`)
            WHERE e.embedding IS NOT NULL AND ({filters})
            WITH e, 1 - vec.cosineDistance(e.embedding, vecf32($embedding)) AS score
            ORDER BY score DESC LIMIT $limit
            RETURN e.id AS name,
                [l in labels(e) WHERE l <> '__Entity__' | l][0] AS type,
                e{{.* , name_embedding: Null, embedding: Null, name: Null, id: Null}} AS properties,
                score
            """,
            param_map={
                "embedding": query.query_embedding,
                "dimension": len(query.query_embedding),
                "limit": query.similarity_top_k,
            },
        )
        data = data if data else []

        nodes = []
        scores = []
        for record in data:
            node = EntityNode(
                name=record["name"],
                label=record["type"],
                properties=remove_empty_values(record["properties"]),
            )
            nodes.append(node)
            scores.append(record["score"])

        return (nodes, scores)
    
    def relation_search(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> Tuple[List[Relation], List[float]]:
        """
        This is the custom function to query the graph store with a vector store query to search for relations.
        The difference is that this function uses cosine similarity score to rank the results.
        """
        conditions = None
        if query.filters:
            conditions = [
                f"r.{filter.key} {filter.operator.value} {filter.value}"
                for filter in query.filters.filters
            ]
        filters = (
            f" {query.filters.condition.value} ".join(conditions).replace("==", "=")
            if conditions is not None
            else "1 = 1"
        )

        data = self.structured_query(
            f"""MATCH (source)-[r]->(target)
            WHERE r.embedding IS NOT NULL AND ({filters}) AND type(r) <> 'MENTIONS'
            WITH r, source, target, 1 - vec.cosineDistance(r.embedding, vecf32($embedding)) AS score
            ORDER BY score DESC LIMIT $limit
            RETURN r{{.* , embedding: Null}} AS properties,
                type(r) AS label,
                score,
                source.id AS source_id,
                target.id AS target_id
            """,
            param_map={
                "embedding": query.query_embedding,
                "dimension": len(query.query_embedding),
                "limit": query.similarity_top_k,
            },
        )
        data = data if data else []

        relations = []
        scores = []
        for record in data:
            relation = Relation(
                source_id=record["source_id"],
                target_id=record["target_id"],
                label=record["label"],
                properties=remove_empty_values(record["properties"]),
            )
            relations.append(relation)
            scores.append(record["score"])

        return (relations, scores)

    def get_community_summaries(self, llm: LLMCore) -> Dict[str, str]:
        """Returns the community summaries, building them if not already done."""
        if not self.community_summary:
            self.build_communities(llm)
        return self.community_summary

    def build_communities(self, llm: LLMCore):
        """Builds communities from the graph and summarizes them."""
        self.llm = llm
        self.nx_graph = self.create_nx_graph()
        community_hierarchical_clusters = hierarchical_leiden(
            graph=self.nx_graph, max_cluster_size=self.max_cluster_size
        )
        self.entity_info, community_info, detail_delimiter = self.collect_community_info(
            self.nx_graph, community_hierarchical_clusters
        )
        self.summarize_communities(community_info, detail_delimiter)

    def collect_community_info(self, nx_graph: Graph, clusters: List[HierarchicalCluster]) -> Tuple[Dict, Dict]:
        """
        Collect information for each node based on their community,
        allowing entities to belong to multiple communities.
        """
        entity_info = defaultdict(set)
        community_info = defaultdict(list)
        detail_delimiter = "->"

        for item in clusters:
            node = item.node
            cluster_id = item.cluster

            # Update entity_info
            entity_info[node].add(cluster_id)

            for neighbor in nx_graph.neighbors(node):
                edge_data = nx_graph.get_edge_data(node, neighbor)
                if edge_data:
                    detail = f"{node} {detail_delimiter} {neighbor} {detail_delimiter} {edge_data['relationship']} {detail_delimiter} {edge_data['description']}"
                    community_info[cluster_id].append(detail)

        # Convert sets to lists for easier serialization if needed
        entity_info = {k: list(v) for k, v in entity_info.items()}

        return dict(entity_info), dict(community_info), detail_delimiter

    def summarize_communities(self, community_info: dict, detail_delimiter: str = "->"):
        """Generate and store summaries for each community."""
        for community_id, details in community_info.items():
            summary_context = self.build_community_summary_context(details, detail_delimiter)
            title, summary = self.generate_community_summary(summary_context)
            self.community_summary[community_id] = {}
            self.community_summary[community_id]["title"] = title
            self.community_summary[community_id]["summary"] = summary
            

    def build_community_summary_context(self, community_detail: List[str], detail_delimiter: str = "->") -> str:
        """Build the context for the community summary using the entity and relationship details."""
        seen_nodes = set()
        entities_info = []
        relationships_info = []
        for detail in community_detail:
            source = detail.split(detail_delimiter)[0].strip()
            target = detail.split(detail_delimiter)[1].strip()
            relationship_desc = detail.split(detail_delimiter)[3].strip()
            relationships_info.append(f"{source}|{target}|{relationship_desc}")

            if source not in seen_nodes:
                seen_nodes.add(source)
                entity_desc = self.get(ids=[source])[0].properties["description"]
                entities_info.append(f"{source}|{entity_desc}")
            if target not in seen_nodes:
                seen_nodes.add(target)
                entity_desc = self.get(ids=[target])[0].properties["description"]
                entities_info.append(f"{target}|{entity_desc}")

        entity_report = (
            "-Entities Reports-" + "\n\n" + "Entity|Description" + "\n" + "\n".join(entities_info) + "\n\n"
        )
        relationship_report = (
            "-Relationships Reports-" + "\n\n" + "Source|Target|Description" + "\n" + "\n".join(relationships_info) + "\n\n"
        )
        return entity_report + relationship_report

    def generate_community_summary(self, report: str):
        """Generate summary from Entity-Relationship Report using an LLM."""
        response = self.llm.complete(
            prompt=PromptTemplate(COMMUNITY_SUMMARIZE_PROMPT_TEMPLATE).format(input_text=report)
        ).text
        result = json.loads(response)
        title = result["title"]
        summary = result["summary"]
        return title, summary

    def create_nx_graph(self) -> Graph:
        """Converts internal graph representation to NetworkX graph."""
        nx_graph = nx.Graph()
        triplets = self.get_triplets()
        for entity1, relation, entity2 in triplets:
            nx_graph.add_node(entity1.name)
            nx_graph.add_node(entity2.name)
            nx_graph.add_edge(
                relation.source_id,
                relation.target_id,
                relationship=relation.label,
                description=relation.properties["description"],
            )
        return nx_graph
    
    def save_community_data(self, directory: str):
        """Save community data to a directory."""
        # Save entity info
        df_entity_info = pd.DataFrame({
            "entity": list(self.entity_info.keys()),
            "community_ids": [json.dumps(ids) for ids in self.entity_info.values()]
        })
        df_entity_info.to_parquet(f"{directory}/entity_info.parquet", index=False)

        # Save community summary
        df_community_summary = pd.DataFrame.from_dict(self.community_summary, orient='index').reset_index()
        df_community_summary.columns = ["community_id", "title", "summary"]
        df_community_summary.to_parquet(f"{directory}/community_summary.parquet", index=False)

        # Save nx graph
        graph = self.nx_graph
        edges_df = nx.to_pandas_edgelist(graph)
        nodes_data = dict(graph.nodes(data=True))
        nodes_df = pd.DataFrame.from_dict(nodes_data, orient="index").reset_index()
        nodes_df.columns = ["node_id"] + list(nodes_df.columns[1:])
        nodes_df.to_parquet(f"{directory}/nodes.parquet", index=False)
        edges_df.to_parquet(f"{directory}/edges.parquet", index=False)

    def load_community_data(self, directory: str):
        """Load community data from a directory."""
        # Load entity info
        df_entity_info = pd.read_parquet(f"{directory}/entity_info.parquet")
        self.entity_info = {
            row['entity']: json.loads(row['community_ids'])
            for _, row in df_entity_info.iterrows()
        }

        # Load community summary
        df_community_summary = pd.read_parquet(f"{directory}/community_summary.parquet")
        community_summary = df_community_summary.set_index('community_id').to_dict(orient='index')
        self.community_summary = {
            key: {
                'title': value['title'],
                'summary': value['summary']
            }
            for key, value in community_summary.items()
        }

        # Load nx graph
        nodes_df = pd.read_parquet(f"{directory}/nodes.parquet")
        edges_df = pd.read_parquet(f"{directory}/edges.parquet")
        graph = nx.from_pandas_edgelist(edges_df, source="source", target="target", edge_attr=True)
        for _, row in nodes_df.iterrows():
            graph.add_node(row["node_id"], **row.drop("node_id").to_dict())
        self.nx_graph = graph

    def get(
        self,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ):
        """
        Custom function to get nodes from the graph store based on properties and IDs.
        The difference is that this function handles property "triplet_source_id" to check whether 
        if the chunk (source text unit) id contained in the list of ids.
        """
        cypher_statement = "MATCH (e) "

        params = {}
        if properties or ids:
            cypher_statement += "WHERE "

        if ids:
            cypher_statement += "e.id in $ids "
            params["ids"] = ids

        if properties:
            prop_list = []
            for i, prop in enumerate(properties):
                if prop == "triplet_source_id":
                    prop_list.append(f"ANY(x IN e.`{prop}` WHERE x = $property_{i})")
                else:
                    prop_list.append(f"e.`{prop}` = $property_{i}")
                params[f"property_{i}"] = properties[prop]
            cypher_statement += " AND ".join(prop_list)

        return_statement = """
        WITH e RETURN e.id AS name,
               [l in labels(e) WHERE l <> '__Entity__' | l][0] AS type,
               e{.* , name_embedding: Null, embedding: Null, id: Null} AS properties
        """
        cypher_statement += return_statement
        response = self.structured_query(cypher_statement, param_map=params)
        response = response if response else []

        nodes = []
        for record in response:
            # text indicates a chunk node
            # none on the type indicates an implicit node, likely a chunk node
            if "text" in record["properties"] or record["type"] is None:
                text = record["properties"].pop("text", "")
                nodes.append(
                    ChunkNode(
                        id_=record["name"],
                        text=text,
                        properties=remove_empty_values(record["properties"]),
                    )
                )
            else:
                nodes.append(
                    EntityNode(
                        name=record["name"],
                        label=record["type"],
                        properties=remove_empty_values(record["properties"]),
                    )
                )

        return nodes

    def get_rel_map(
        self,
        graph_nodes: List[LabelledNode],
        depth: int = 2,
        limit: int = 30,
        ignore_rels: Optional[List[str]] = None,
    ) -> List[Triplet]:
        """
        This is a custom function to get triplets from the graph store within the depth-specified path from the given nodes.

        Args:
            graph_nodes (List[LabelledNode]): A list of nodes from which to start the search.
            depth (int, optional): The depth of the search path. Defaults to 2.
            limit (int, optional): The maximum number of relationships to return. Defaults to 30.
            ignore_rels (Optional[List[str]], optional): A list of relationship types to ignore. Defaults to None.

        Returns:
            List[Triplet]: A list of triplets, where each triplet consists of a source node, a relationship, and a target node.
        """
        triples = []

        ids = [node.id for node in graph_nodes]

        response = self.structured_query(
            f"""
            WITH $ids AS id_list
            UNWIND range(0, size(id_list) - 1) AS idx
            MATCH (e:`__Entity__`)
            WHERE e.id = id_list[idx]
            MATCH p=(e)-[r*1..{depth}]-(other)
            WHERE ALL(rel in relationships(p) WHERE type(rel) <> 'MENTIONS')
            UNWIND relationships(p) AS rel
            WITH distinct rel, idx
            WITH startNode(rel) AS source,
                type(rel) AS type,
                endNode(rel) AS endNode,
                rel{{.*}} AS relationship_properties,
                idx
            LIMIT $limit
            RETURN source.id AS source_id, [l in labels(source) WHERE l <> '__Entity__' | l][0] AS source_type,
                source{{.* , name_embedding: Null, embedding: Null, id: Null}} AS source_properties,
                type,
                relationship_properties,
                endNode.id AS target_id, [l in labels(endNode) WHERE l <> '__Entity__' | l][0] AS target_type,
                endNode{{.* , name_embedding: Null, embedding: Null, id: Null}} AS target_properties,
                idx
            ORDER BY idx
            LIMIT $limit
            """,
            param_map={"ids": ids, "limit": limit},
        )
        response = response if response else []

        ignore_rels = ignore_rels or []
        for record in response:
            if record["type"] in ignore_rels:
                continue

            try:
                source = EntityNode(
                    name=record["source_id"],
                    label=record["source_type"],
                    properties=remove_empty_values(record["source_properties"]),
                )
                target = EntityNode(
                    name=record["target_id"],
                    label=record["target_type"],
                    properties=remove_empty_values(record["target_properties"]),
                )
                rel = Relation(
                    source_id=record["source_id"],
                    target_id=record["target_id"],
                    label=record["type"],
                    properties=remove_empty_values(record["relationship_properties"]),
                )
            except Exception as e:
                raise InitEntityRelationNodesError(f"Error initializing Entity/Relation nodes from record: {record}") from e
            triples.append([source, rel, target])

        return triples

    def get_triplets(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[Triplet]:
        """
        Custom function to get triplets from the graph store based on entity names, relation names, properties, and IDs.
        The difference is that this function gets the properties of the relation as well.
        """
        
        # Find nodes labeled as "__Entity__" only
        cypher_statement = "MATCH (e:`__Entity__`) "

        params = {}
        if entity_names or properties or ids:
            cypher_statement += "WHERE "

        if entity_names:
            cypher_statement += "e.name in $entity_names "
            params["entity_names"] = entity_names

        if ids:
            cypher_statement += "e.id in $ids "
            params["ids"] = ids

        if properties:
            prop_list = []
            for i, prop in enumerate(properties):
                prop_list.append(f"e.`{prop}` = $property_{i}")
                params[f"property_{i}"] = properties[prop]
            cypher_statement += " AND ".join(prop_list)

        return_statement = f"""
        WITH e
        CALL {{
            WITH e
            MATCH (e)-[r{':`' + '`|`'.join(relation_names) + '`' if relation_names else ''}]->(t:__Entity__)
            RETURN e.name AS source_id, [l in labels(e) WHERE l <> '__Entity__' | l][0] AS source_type,
                   e{{.* , name_embedding: Null, embedding: Null, name: Null}} AS source_properties,
                   type(r) AS type,
                   t.name AS target_id, [l in labels(t) WHERE l <> '__Entity__' | l][0] AS target_type,
                   t{{.* , name_embedding: Null, embedding: Null, name: Null}} AS target_properties,
                   r{{.*}} AS relationship_properties
            UNION ALL
            WITH e
            MATCH (e)<-[r{':`' + '`|`'.join(relation_names) + '`' if relation_names else ''}]-(t:__Entity__)
            RETURN t.name AS source_id, [l in labels(t) WHERE l <> '__Entity__' | l][0] AS source_type,
                   e{{.* , name_embedding: Null, embedding: Null, name: Null}} AS source_properties,
                   type(r) AS type,
                   e.name AS target_id, [l in labels(e) WHERE l <> '__Entity__' | l][0] AS target_type,
                   t{{.* , name_embedding: Null, embedding: Null, name: Null}} AS target_properties,
                   r{{.*}} AS relationship_properties
        }}
        RETURN source_id, source_type, type, target_id, target_type, source_properties, target_properties, relationship_properties"""
        cypher_statement += return_statement

        data = self.structured_query(cypher_statement, param_map=params)
        data = data if data else []

        triples = []
        for record in data:
            source = EntityNode(
                name=record["source_id"],
                label=record["source_type"],
                properties=remove_empty_values(record["source_properties"]),
            )
            target = EntityNode(
                name=record["target_id"],
                label=record["target_type"],
                properties=remove_empty_values(record["target_properties"]),
            )
            rel = Relation(
                source_id=record["source_id"],
                target_id=record["target_id"],
                label=record["type"],
                properties=remove_empty_values(record["relationship_properties"]),
            )
            triples.append([source, rel, target])
        return triples
    
    def upsert_nodes(self, nodes: List[LabelledNode], entity_name_embeddings: List[float] = None, show_progress: bool = False) -> None:
        """
        This is a custom function to upsert nodes into the graph store.
        The difference is that this function will check whether if chunk's id contained in the list "triplet_source_id" 
        to add the relation `MENTIONS` between entity and chunk.
        """
        # Lists to hold separated types
        entity_dicts: List[dict] = []
        chunk_dicts: List[dict] = []
    
        # Sort by type
        for item in nodes:
            if isinstance(item, EntityNode):
                entity_dicts.append({**item.dict(), "id": item.id})
            elif isinstance(item, ChunkNode):
                chunk_dicts.append({**item.dict(), "id": item.id})
            else:
                raise ValueError(f"Unsupported node type: {type(item)}")
    
        if chunk_dicts:
            iterable = tqdm(chunk_dicts, desc="Upserting text chunks") if show_progress else chunk_dicts
            for chunk in iterable:
                self.structured_query(
                    """
                    MERGE (c:Chunk {id: $data.id})
                    SET c.text = $data.text
                    WITH c
                    SET c += $data.properties
                    WITH c, $data.embedding AS embedding
                    WHERE embedding IS NOT NULL
                    SET c.embedding = vecf32(embedding)
                    RETURN count(*)
                    """,
                    param_map={"data": chunk},
                )
    
        if entity_dicts:
            if not entity_name_embeddings:
                raise GraphUpsertNodesError("Missing entity name embeddings")
            # Add entity name embeddings
            for entity_dict, embedding in zip(entity_dicts, entity_name_embeddings):
                entity_dict["name_embedding"] = embedding

            iterable = tqdm(entity_dicts, desc="Upserting entities") if show_progress else entity_dicts
            for entity_dict in iterable:
                self.structured_query(
                    f"""
                    MERGE (e:`__Entity__` {{id: $data.id}})
                    SET e += $data.properties
                    SET e.name = $data.name
                    WITH e
                    SET e:{entity_dict["label"]}
                    WITH e
                    CALL {{
                        WITH e
                        WITH e
                        WHERE $data.embedding IS NOT NULL
                        SET e.embedding = vecf32($data.embedding)
                        RETURN count(*) AS count
                    }}
                    WITH e
                    CALL {{
                        WITH e
                        WITH e
                        WHERE $data.name_embedding IS NOT NULL
                        SET e.name_embedding = vecf32($data.name_embedding)
                        RETURN count(*) AS count
                    }}
                    WITH e
                    UNWIND $data.properties.triplet_source_id AS source_id
                    MATCH (c:Chunk {{id: source_id}})
                    WHERE source_id IS NOT NULL
                    MERGE (e)<-[:MENTIONS]-(c)
                    """,
                    param_map={"data": entity_dict},
                )

    def upsert_relations(self, relations: List[Relation], embeddings: List[List[float]], show_progress: bool = False) -> None:
        """
        This is a custom function to upsert relations into the graph store.
        The difference is that this function will insert the relation embedding if it is provided.

        Args:
            relations (List[Relation]): A list of relations to insert.
            embeddings (List[List[float]]): A list of relation embeddings to insert.
        """
        params = []
        for relation, embedding in zip(relations, embeddings):
            param = {
                "source_id": relation.source_id,
                "target_id": relation.target_id,
                "label": relation.label,
                "properties": relation.properties,
                "embedding": embedding
            }
            params.append(param)

        iterable = tqdm(params, desc="Upserting relations") if show_progress else params
        for param in iterable:
            self.structured_query(
                f"""
                MERGE (source {{id: $data.source_id}})
                ON CREATE SET source:Chunk
                MERGE (target {{id: $data.target_id}})
                ON CREATE SET target:Chunk
                WITH source, target
                CREATE (source)-[r:`{param["label"]}`]->(target)
                SET r += $data.properties
                WITH r
                WHERE $data.embedding IS NOT NULL
                SET r.embedding = vecf32($data.embedding)
                RETURN count(*)
                """,
                param_map={"data": param},
            )
