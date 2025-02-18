from typing import Optional, Sequence

from llama_index.core import Document
from llama_index.core.schema import TransformComponent

from chatbot.core.model_clients import EmbedderCore, LLMCore
from chatbot.core.graph_stores.exceptions import GraphLoadError, GraphConstructionError
from chatbot.core.graph_stores.falkordb_property_graph_store import CustomFalkorDBPropertyGraphStore
from chatbot.core.graph_stores.geographical_data import GeographicalData
from chatbot.core.graph_stores.property_graph_index import CustomPropertyGraphIndex


class FalkorDBGraphStore:
    """
    A class to interact with a FalkorDB graph database for storing and querying graphs.
    
    Attributes:
    - url (str): The URL of the FalkorDB database.
    - database (str): The name of the FalkorDB database. Default is "falkor".
    - llm (LLMCore): The LLM model to use for text embedding.
    - embedder (EmbedderCore): The embedder model to use for node embedding.
    - graph_extractor (TransformComponent): The graph extractor to use for entity extraction.
    - documents (Sequence[Document]): The documents to build the graph from.
    - build (bool): Whether to build the graph from the provided documents. Default is False.
    - force_build (bool): Whether to force rebuilding the graph even if it already exists. Default is False.
    - time_aware (bool): Whether the graph is time-aware. This will automatically add a timestamp to each inserted document. Default is False.
    - geographical_data (Optional[GeographicalData]): The geographical data to use for getting the timestamp of a location.
    - show_progress (bool): Whether to show progress during graph construction. Default is True.

    Examples:
    ```python
    from llama_index.core import Document

    from core.graph_stores import FalkorDBGraphStore, GraphExtractor
    from core.graph_stores.geographical_data import GeographicalData
    from core.model_clients import EmbedderCore, LLMCore

    documents = [
        Document(text="Albert Einstein worked at the Institute for Advanced Study in Princeton."),
        Document(text="Isaac Newton was a mathematician and physicist.")
    ]

    embedder = EmbedderCore()
    llm = LLMCore()
    graph_extractor = GraphExtractor()

    geo_data = GeographicalData()

    graph_store = FalkorDBGraphStore(
        url="falkor://localhost:6379",
        database="falkor",
        llm=llm,
        embedder=embedder,
        graph_extractor=graph_extractor,
        description_summarization_max_retries=3,
        geographical_data=geo_data,
        documents=documents,
        time_aware=True,
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
        description_summarization_max_retries: int = 3, # Number of retries for description summarization
        max_cluster_size: int = 10, # Comumunity context is currently deprecated
        documents: Sequence[Document] = [],
        build: bool = False,
        force_build: bool = False,
        time_aware: bool = False,
        geographical_data: Optional[GeographicalData] = None,
        show_progress: bool = True
    ):
        self.property_graph_store = CustomFalkorDBPropertyGraphStore(
            url=url, database=database,
        )
        self.property_graph_store.max_cluster_size = max_cluster_size
        self.llm = llm
        self.embedder = embedder
        self.graph_extractor = graph_extractor

        if build:
            if not documents:
                raise ValueError("No documents provided to build the graph.")
            
            if len(self.property_graph_store.get_schema()["relationships"]) > 0 and not force_build:
                print("Graph already exists. Set force_build to True to rebuild the graph. Loading the existing graph...")

                # Load the existing graph
                self.index = self.load_graph(
                    description_summarization_max_retries=description_summarization_max_retries,
                    time_aware=time_aware,
                    geo_data=geographical_data
                )
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
                description_summarization_max_retries=description_summarization_max_retries,
                time_aware=time_aware,
                geo_data=geographical_data,
                show_progress=show_progress
            )

            asyncio.set_event_loop(original_loop)
        else:
            self.index = self.load_graph()

    def load_graph(
        self,
        description_summarization_max_retries: int = 3,
        time_aware: bool = False,
        geo_data: Optional[GeographicalData] = None
    ) -> CustomPropertyGraphIndex:
        """Load the graph store from an existing graph database."""
        try:
            index = CustomPropertyGraphIndex.from_existing(
                property_graph_store=self.property_graph_store,
                llm=self.llm,
                embed_model=self.embedder,
                kg_extractors=[self.graph_extractor],
                description_summarization_max_retries=description_summarization_max_retries,
                geographical_data=geo_data,
                embed_kg_nodes=True,
                use_async=True,
                time_aware=time_aware
            )
        except Exception as e:
            raise GraphLoadError(f"Error loading the graph: {e}")
        return index

    def construct_graph(
            self,
            documents: Sequence[Document],
            description_summarization_max_retries: int = 3,
            time_aware: bool = False,
            geo_data: Optional[GeographicalData] = None,
            show_progress: bool = True
    ) -> CustomPropertyGraphIndex:
        try:
            index = CustomPropertyGraphIndex.from_documents(
                documents=documents,
                kg_extractors=[self.graph_extractor],
                llm=self.llm,
                embed_model=self.embedder,
                property_graph_store=self.property_graph_store,
                description_summarization_max_retries=description_summarization_max_retries,
                geographical_data=geo_data,
                time_aware=time_aware,
                show_progress=show_progress
            )
        except Exception as e:
            raise GraphConstructionError(f"Error constructing the graph: {e}")
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
