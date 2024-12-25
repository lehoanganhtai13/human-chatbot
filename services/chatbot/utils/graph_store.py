from typing import Any, Dict, List, Tuple, Sequence
import re
import json

from llama_index.core import PropertyGraphIndex, Document
from llama_index.core.schema import TransformComponent
from llama_index.graph_stores.falkordb import FalkorDBPropertyGraphStore
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.graph_stores.types import EntityNode, Relation

from chatbot.query.graph_query import NEO4J_CREATE_VECTOR_INDEX_QUERY, NEO4J_DEDUPLICATION_AND_MERGE_QUERY
from chatbot.utils.models_client import EmbedderCore, LLMCore


def parse_dynamic_triplets_with_props(
    llm_output: str,
) -> List[Tuple[EntityNode, Relation, EntityNode]]:
    """
    This is a custom function to parse the LLM output and convert it into a list of entity-relation-entity triplets.
    Parse the LLM output and convert it into a list of entity-relation-entity triplets.
    This function is flexible and can handle various output formats.

    Args:
        llm_output (str): The output from the LLM, which may be JSON-like or plain text.

    Returns:
        List[Tuple[EntityNode, Relation, EntityNode]]: A list of triplets.
    """
    triplets = []

    # Clean and standardize input
    llm_output = llm_output.strip()
    if llm_output.startswith("```json"):
        # Remove markdown code block markers
        llm_output = llm_output.replace("```json", "").replace("```", "").strip()

    try:
        # Attempt to parse the output as JSON
        data = json.loads(llm_output)
        for item in data:
            head = item.get("head")
            head_type = item.get("head_type")
            head_props = item.get("head_props", {})
            relation = item.get("relation")
            relation_props = item.get("relation_props", {})
            tail = item.get("tail")
            tail_type = item.get("tail_type")
            tail_props = item.get("tail_props", {})

            if head and head_type and relation and tail and tail_type:
                head_node = EntityNode(
                    name=head, label=head_type, properties=head_props
                )
                tail_node = EntityNode(
                    name=tail, label=tail_type, properties=tail_props
                )
                relation_node = Relation(
                    source_id=head_node.id,
                    target_id=tail_node.id,
                    label=relation,
                    properties=relation_props,
                )
                triplets.append((head_node, relation_node, tail_node))
    except json.JSONDecodeError:
        # Flexible pattern to match the key-value pairs for head, head_type, head_props, relation, relation_props, tail, tail_type, and tail_props
        pattern = r'[\{"\']head[\}"\']\s*:\s*[\{"\'](.*?)[\}"\']\s*,\s*[\{"\']head_type[\}"\']\s*:\s*[\{"\'](.*?)[\}"\']\s*,\s*[\{"\']head_props[\}"\']\s*:\s*\{(.*?)\}\s*,\s*[\{"\']relation[\}"\']\s*:\s*[\{"\'](.*?)[\}"\']\s*,\s*[\{"\']relation_props[\}"\']\s*:\s*\{(.*?)\}\s*,\s*[\{"\']tail[\}"\']\s*:\s*[\{"\'](.*?)[\}"\']\s*,\s*[\{"\']tail_type[\}"\']\s*:\s*[\{"\'](.*?)[\}"\']\s*,\s*[\{"\']tail_props[\}"\']\s*:\s*\{(.*?)\}\s*'

        # Find all matches in the output
        matches = re.findall(pattern, llm_output)

        for match in matches:
            (
                head,
                head_type,
                head_props,
                relation,
                relation_props,
                tail,
                tail_type,
                tail_props,
            ) = match

            # Use more robust parsing for properties
            def parse_props(props_str: str) -> Dict[str, Any]:
                try:
                    # Handle mixed quotes and convert to a proper dictionary
                    props_str = props_str.replace("'", '"')
                    return json.loads(f"{{{props_str}}}")
                except json.JSONDecodeError:
                    return {}

            head_props_dict = parse_props(head_props)
            relation_props_dict = parse_props(relation_props)
            tail_props_dict = parse_props(tail_props)

            head_node = EntityNode(
                name=head, label=head_type, properties=head_props_dict
            )
            tail_node = EntityNode(
                name=tail, label=tail_type, properties=tail_props_dict
            )
            relation_node = Relation(
                source_id=head_node.id,
                target_id=tail_node.id,
                label=relation,
                properties=relation_props_dict,
            )
            triplets.append((head_node, relation_node, tail_node))
    return triplets


class FalkorDBGraphStore:
    """A class to interact with a FalkorDB graph database.
    
    Attributes:
    - url (str): The URL of the FalkorDB database.
    - database (str): The name of the FalkorDB database. Default is "falkor".
    - llm (LLMCore): The LLM model to use for text embedding.
    - embedder (EmbedderCore): The embedder model to use for node embedding.
    - graph_extractor (TransformComponent): The graph extractor to use for entity extraction.
    - documents (Sequence[Document]): The documents to build the graph from.
    - build (bool): Whether to build the graph from the provided documents. Default is False.
    - force_build (bool): Whether to force rebuilding the graph even if it already exists. Default is False.
    - show_progress (bool): Whether to show progress during graph construction. Default is True.

    Examples:
    ```python
    from llama_index.core import Document
    from llama_index.core.indices.property_graph import DynamicLLMPathExtractor

    from utils.graph_store import FalkorDBGraphStore
    from utils.models_client import EmbedderCore, LLMCore

    documents = [
        Document(text="Albert Einstein worked at the Institute for Advanced Study in Princeton."),
        Document(text="Isaac Newton was a mathematician and physicist.")
    ]

    embedder = EmbedderCore()
    llm = LLMCore()
    graph_extractor = DynamicLLMPathExtractor()

    graph_store = FalkorDBGraphStore(
        url="falkor://localhost:6379",
        database="falkor",
        llm=llm,
        embedder=embedder,
        graph_extractor=graph_extractor,
        documents=documents,
        build=True,
        force_build=False,
        show_progress=True
    )
    ```
    """
    def __init__(
            self,
            url: str = "falkor://localhost:6379",
            database: str = "falkor",
            llm: LLMCore = None,
            embedder: EmbedderCore = None,
            graph_extractor: TransformComponent = None,
            documents: Sequence[Document] = [],
            build: bool = False,
            force_build: bool = False,
            show_progress: bool = True
    ):
        self.property_graph_store = FalkorDBPropertyGraphStore(
            url=url, database=database,
        )
        self.llm = llm
        self.embedder = embedder
        self.graph_extractor = graph_extractor

        if build:
            if not documents:
                raise ValueError("No documents provided to build the graph.")
            
            if len(self.property_graph_store.get_schema()["relationships"]) > 0 and not force_build:
                print("Graph already exists. Set force_build to True to rebuild the graph. Loading the existing graph...")
                # Load the existing graph
                self.index = self.load_graph()
                return None
            
            import asyncio
            import nest_asyncio

            original_loop = asyncio.get_event_loop()
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            asyncio.set_event_loop(loop)

            # Clear the graph store and reconstruct the graph memory store
            self.clear_graph()
            self.index = self.construct_graph(
                documents=documents,
                show_progress=show_progress
            )

            asyncio.set_event_loop(original_loop)
        else:
            self.index = self.load_graph()

    def load_graph(self):
        """Load the graph store from an existing graph database."""
        try:
            index = PropertyGraphIndex.from_existing(
                property_graph_store=self.property_graph_store,
                llm=self.llm,
                embed_model=self.embedder,
                kg_extractors=[self.graph_extractor],
                embed_kg_nodes=True,
                use_async=True
            )
        except Exception as e:
            raise f"Error loading the graph: {e}"
        return index 

    def construct_graph(
            self,
            documents: Sequence[Document],
            show_progress: bool = True,
    ):
        retry_times = 3
        for i in range(retry_times):
            try:
                index = PropertyGraphIndex.from_documents(
                    documents=documents,
                    kg_extractors=[self.graph_extractor],
                    llm=self.llm,
                    embed_model=self.embedder,
                    property_graph_store=self.property_graph_store,
                    show_progress=show_progress,
                )
                break
            except Exception as e:
                if i == retry_times - 1:
                    raise f"Error constructing the graph: {e}"
        return index

    def insert_document(
            self,
            document: Document,
    ):
        """Insert a document into the graph database."""
        self.index.insert(document)

    async def async_insert_document(
            self,
            document: Document,
    ):
        """Insert a document into the graph database asynchronously."""
        self.insert_document(document=document)

    def clear_graph(self):
        """Clear the graph store."""
        return self.property_graph_store.structured_query("MATCH (n) DELETE n")
    
    async def async_clear_graph(self):
        """Clear the graph store asynchronously."""
        return self.clear_graph()
    
    def delete_graph(self):
        """Delete the graph store."""
        return self.property_graph_store._graph.delete()
    
    async def async_delete_graph(self):
        """Delete the graph store asynchronously."""
        return self.delete_graph()
    
    def get_schema_info(self):
        """Get the schema information of the graph database."""
        entity_list = {}
        seen_values = set()
        label_list = list(self.property_graph_store.get_schema()["node_props"])
        for label in label_list:
            if label == "Chunk":
                continue

            entity_list[label] = []
            entities_result = self.property_graph_store.structured_query(
                query=f"MATCH (n:{label}) RETURN n.name as name"
            )
            for entity in entities_result:
                entity_name = entity["name"]
                entity_list[label].append(entity_name)
                seen_values.add(entity_name)

        return entity_list

    def get_schema_info_str(self):
        """Get the schema information of the graph database as a string."""
        schema_info = self.get_schema_info()
        schema_str = ""
        for entity_type in schema_info:
            schema_str += f"{entity_type}: {schema_info[entity_type]}\n"

        return schema_str


class Neo4jGraphStore:
    """A class to interact with a Neo4j graph database.
    
    Attributes:
    - url (str): The URL of the Neo4j database.
    - username (str): The username to connect to the Neo4j database. Default is "neo4j".
    - password (str): The password to connect to the Neo4j database. Default is "password".
    - database (str): The name of the Neo4j database. Default is "neo4j" for community edition.
    - llm (LLMCore): The LLM model to use for text embedding.
    - embedder (EmbedderCore): The embedder model to use for node embedding.
    - embedding_dim (int): The dimension of the node embeddings. Default is 1024.
    - graph_extractor (TransformComponent): The graph extractor to use for entity extraction.
    - documents (Sequence[Document]): The documents to build the graph from.
    - build (bool): Whether to build the graph from the provided documents. Default is False.
    - force_build (bool): Whether to force rebuilding the graph even if it already exists. Default is False.
    - is_deduplication (bool): Whether to deduplicate nodes in the graph when building or inserting documents. Default is False.
    - similarity_threshold (float): The similarity threshold for deduplication. Default is 0.9.
    - word_edit_distance (int): The word edit distance between nodes allowed for deduplication. Default is 5.
    - similarity_metric (str): The similarity metric for deduplication. Default is "COSINE". Support "COSINE" and "EUCLIDEAN".
    - show_progress (bool): Whether to show progress during graph construction. Default is True.

    Examples:
    ```python
    from llama_index.core import Document
    from llama_index.core.indices.property_graph import DynamicLLMPathExtractor

    from utils.graph_store import Neo4jGraphStore
    from utils.models_client import EmbedderCore, LLMCore

    documents = [
        Document(text="Albert Einstein worked at the Institute for Advanced Study in Princeton."),
        Document(text="Isaac Newton was a mathematician and physicist.")
    ]

    embedder = EmbedderCore()
    llm = LLMCore()
    graph_extractor = DynamicLLMPathExtractor()

    graph_store = Neo4jGraphStore(
        url="bolt://localhost:7687",
        username="neo4j",
        password="password",
        database="neo4j",
        llm=llm,
        embedder=embedder,
        embedding_dim=1024,
        graph_extractor=graph_extractor,
        documents=documents,
        build=True,
        force_build=False,
        is_deduplication=False,
        similarity_threshold=0.9,
        word_edit_distance=5,
        similarity_metric="COSINE",
        show_progress=True
    )
    ```
    """
    def __init__(
            self,
            url: str = "bolt://localhost:7687",
            username: str = "neo4j",
            password: str = "password",
            database: str = "neo4j",
            llm: LLMCore = None,
            embedder: EmbedderCore = None,
            embedding_dim: int = 1024,
            graph_extractor: TransformComponent = None,
            documents: Sequence[Document] = [],
            build: bool = False,
            force_build: bool = False,
            is_deduplication: bool = False,
            similarity_threshold: float = 0.9,
            word_edit_distance: int = 5,
            similarity_metric: str = "COSINE", # Support "COSINE" and "EUCLIDEAN"
            show_progress: bool = True
    ):
        self.property_graph_store = Neo4jPropertyGraphStore(
            url=url,
            username=username,
            password=password,
            database=database
        )
        self.llm = llm
        self.embedder = embedder
        self.graph_extractor = graph_extractor
        self.embedding_dim = embedding_dim

        if build:
            if not documents:
                raise ValueError("No documents provided to build the graph.")
            
            if len(self.property_graph_store.get_schema()["relationships"]) > 0 and not force_build:
                print("Graph already exists. Set force_build to True to rebuild the graph. Loading the existing graph...")
                # Load the existing graph
                self.index = self.load_graph()
                return None
            
            import asyncio
            import nest_asyncio

            original_loop = asyncio.get_event_loop()
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            asyncio.set_event_loop(loop)

            # Clear the graph store and reconstruct the graph memory store
            self.clear_graph()
            self.index = self.construct_graph(
                documents=documents,
                is_deduplication=is_deduplication,
                similarity_threshold=similarity_threshold,
                word_edit_distance=word_edit_distance,
                similarity_metric=similarity_metric,
                show_progress=show_progress
            )

            asyncio.set_event_loop(original_loop)
        else:
            self.index = self.load_graph()

    def load_graph(self):
        """Load the graph store from an existing graph database."""
        try:
            index = PropertyGraphIndex.from_existing(
                property_graph_store=self.property_graph_store,
                llm=self.llm,
                embed_model=self.embedder,
                kg_extractors=[self.graph_extractor],
                embed_kg_nodes=True,
                use_async=True
            )
        except Exception as e:
            raise f"Error loading the graph: {e}"
        return index    

    def construct_graph(
            self,
            documents: Sequence[Document],
            is_deduplication: bool = False,
            similarity_threshold: float = 0.9,
            word_edit_distance: int = 5,
            similarity_metric: str = "COSINE",
            show_progress: bool = True
    ) -> PropertyGraphIndex:
        """Construct a graph in the graph database."""
        retry_times = 3
        for i in range(retry_times):
            try:
                index = PropertyGraphIndex.from_documents(
                    documents=documents,
                    kg_extractors=[self.graph_extractor],
                    llm=self.llm,
                    embed_model=self.embedder,
                    property_graph_store=self.property_graph_store,
                    show_progress=show_progress,
                )

                if is_deduplication:
                    self.deduplicate_nodes(
                        self.property_graph_store,
                        similarity_threshold=similarity_threshold,
                        word_edit_distance=word_edit_distance,
                        similarity_metric=similarity_metric,
                        show_progress=show_progress
                    )
                break
            except Exception as e:
                if i == retry_times - 1:
                    raise f"Error constructing the graph: {e}"
        return index

    def deduplicate_nodes(
            self,
            graph: Neo4jPropertyGraphStore,
            similarity_threshold: float = 0.9,
            word_edit_distance: int = 5,
            similarity_metric: str = "COSINE",
            show_progress: bool = True
    ) -> None: 
        """Deduplicate the nodes in the graph database."""
        if show_progress:
            print("Deduplicating nodes...")

        # Create a vector index for the nodes
        graph.structured_query(
            query=NEO4J_CREATE_VECTOR_INDEX_QUERY,
            param_map={"dimensions": self.embedding_dim, "similarity_function": similarity_metric} # Suport "cosine" and "euclidean"
        )

        # Deduplicate the nodes
        data = graph.structured_query(
            query=NEO4J_DEDUPLICATION_AND_MERGE_QUERY,
            param_map={"cutoff": similarity_threshold, "distance": word_edit_distance}
        )
        if show_progress:
            print(f"Combined {len(data)} nodes.")
            if len(data) > 0:
                for row in data:
                    print("Combined nodes:", row["combinedResult"])

    def insert_document(
            self,
            document: Document,
            is_deduplication: bool = False,
            similarity_threshold: float = 0.9,
            word_edit_distance: int = 5,
            similarity_metric: str = "COSINE",
            show_progress: bool = True
    ):
        """Insert a document into the graph database."""
        self.index.insert(document)
        if is_deduplication:
            self.deduplicate_nodes(
                self.property_graph_store,
                similarity_threshold=similarity_threshold,
                word_edit_distance=word_edit_distance,
                similarity_metric=similarity_metric,
                show_progress=show_progress
            )

    async def async_insert_document(
            self,
            document: Document,
            is_deduplication: bool = False,
            similarity_threshold: float = 0.9,
            word_edit_distance: int = 5,
            similarity_metric: str = "COSINE",
            show_progress: bool = True
    ):
        """Insert a document into the graph database asynchronously."""
        self.insert_document(
            document=document,
            is_deduplication=is_deduplication,
            similarity_threshold=similarity_threshold,
            word_edit_distance=word_edit_distance,
            similarity_metric=similarity_metric,
            show_progress=show_progress
        )

    def clear_graph(self):
        """Clear the graph store."""
        return self.property_graph_store.structured_query("MATCH (n) DETACH DELETE n")
    
    async def async_clear_graph(self):
        """Clear the graph store asynchronously."""
        return self.clear_graph()
    
    def get_schema_info(self):
        """Get the schema information of the graph database."""
        entity_list = {}
        seen_values = set()
        node_entities = list(self.property_graph_store.get_schema()["node_props"])
        for entity_type in node_entities:
            if entity_type == "Chunk":
                continue

            entity_list[entity_type] = []
            for node in list(self.property_graph_store.get_schema()["node_props"][entity_type]):
                if "values" in list(node.keys()) and node["property"] == "name":
                    for value in node["values"]:
                        if value not in seen_values:
                            seen_values.add(value)
                            entity_list[entity_type].append(value)

        return entity_list
    
    def get_schema_info_str(self):
        """Get the schema information of the graph database as a string."""
        schema_info = self.get_schema_info()

        schema_str = ""
        for entity_type in schema_info:
            schema_str += f"{entity_type}: {schema_info[entity_type]}\n"

        return schema_str
