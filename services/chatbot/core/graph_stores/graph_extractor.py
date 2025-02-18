from typing import Any, Callable, Dict, List, Optional, Sequence, Union, Tuple

from llama_index.core.indices.property_graph import DynamicLLMPathExtractor
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.graph_stores.types import KG_NODES_KEY, KG_RELATIONS_KEY, PromptTemplate

from chatbot.prompt.graph.triplet_extraction import CONTINUE_PROMPT, LOOP_PROMPT
from chatbot.core.model_clients import LLMCore
from chatbot.core.chat_stores import CacheChatStore


class GraphExtractor(DynamicLLMPathExtractor):

    _llm: Optional[LLMCore] = None
    entity_extraction_max_retries: int = 1
    _show_progress: bool = False

    def __init__(
        self,
        llm: Optional[LLMCore] = None,
        extract_prompt: Optional[Union[str, PromptTemplate]] = None,
        entity_extraction_max_retries: int = 1,
        parse_fn: Optional[Callable] = None,
        max_triplets_per_chunk: int = 10,
        num_workers: int = 4,
        allowed_entity_types: Optional[List[str]] = None,
        allowed_entity_props: Optional[Union[List[str], List[Tuple[str, str]]]] = None,
        allowed_relation_types: Optional[List[str]] = None,
        allowed_relation_props: Optional[
            Union[List[str], List[Tuple[str, str]]]
        ] = None,
        show_progress: bool = False,
    ):
        """
        The GraphExtractor class is used to extract triplets from a document using an LLM model.

        Args:
            llm (Optional[LLMCore]): The LLM model client to use for extraction.
            extract_prompt (Optional[Union[str, PromptTemplate]]): The prompt to use for extraction.
            entity_extraction_max_retries (int): The maximum number of retries for entity extraction.
            parse_fn (Optional[Callable]): The function to use for parsing the LLM response.
            max_triplets_per_chunk (int): The maximum number of triplets to extract per chunk.
            num_workers (int): The number of workers to use for extraction.
            allowed_entity_types (Optional[List[str]]): The allowed entity types to extract.
            allowed_entity_props (Optional[Union[List[str], List[Tuple[str, str]]]): The allowed entity properties to extract.
            allowed_relation_types (Optional[List[str]]): The allowed relation types to extract.
            allowed_relation_props (Optional[Union[List[str], List[Tuple[str, str]]]): The allowed relation properties to extract.
        """
        super().__init__(
            llm=llm,
            extract_prompt=extract_prompt,
            parse_fn=parse_fn,
            max_triplets_per_chunk=max_triplets_per_chunk,
            num_workers=num_workers,
            allowed_entity_types=allowed_entity_types,
            allowed_entity_props=allowed_entity_props,
            allowed_relation_types=allowed_relation_types,
            allowed_relation_props=allowed_relation_props,
        )

        # Set the other attributes not set in the parent class
        self._llm = llm
        self.entity_extraction_max_retries = entity_extraction_max_retries

        # Set the progress flag
        self._show_progress = show_progress

    @property
    def llm(self) -> LLMCore:
        """Get the LLM model client."""
        return self._llm
    
    @llm.setter
    def llm(self, value) -> None:
        """Set the LLM model client."""
        self._llm = value

    async def _aextract(self, node: BaseNode) -> BaseNode:
        """
        Asynchronously extract triples from a single document node.

        Args:
            node (BaseNode): The node to process.

        Returns:
            BaseNode: The processed document node with extracted information.
        """
        text = node.get_content(metadata_mode=MetadataMode.LLM)
        try:
            # Set the function name for storing the call history
            self.llm.set_function_name("extract_triplets")

            # Extract triplets from the text with the Instruction prompt
            if (
                self.allowed_entity_props is not None
                and self.allowed_relation_props is not None
            ):
                llm_response = await self._apredict_with_props(text)
            else:
                llm_response = await self._apredict_without_props(text)

            triplets = self.parse_fn(llm_response)

            for i in range(self.entity_extraction_max_retries):
                # Retry to maximize the extracted triplet count
                llm_response = self.llm.complete(CONTINUE_PROMPT).text

                # Parse the new triplets and add them to the list
                triplets.extend(self.parse_fn(llm_response))

                # If there are no new triplets, break the loop
                if llm_response == "NO NEW TRIPLETS":
                    break

                # If the maximum number of retries is reached, break the loop early
                if i >= self.entity_extraction_max_retries - 1:
                    break

                # Check if there are more triplets to extract after retrying
                checking_status = self.llm.complete(LOOP_PROMPT).text

                if self._show_progress:
                    print(f"Checking status: {checking_status}")

                if checking_status == "NO":
                    break

            if self._show_progress:
                print("=== Triplets ===")
                print(f"Extracted {len(triplets)} triplets:")
                for triplet in triplets:
                    print(f"- {triplet}")
                print("=== End of Triplets ===")
            
            # Clean up the call history and function name for the next extraction
            self.llm.clean_call_history()
            self.llm.clear_function_name()
        except Exception as e:
            print(f"Error during extraction: {e!s}")
            triplets = []

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])

        metadata = node.metadata.copy()
        for subj, rel, obj in triplets:
            subj.properties.update(metadata)
            obj.properties.update(metadata)
            rel.properties.update(metadata)

            existing_nodes.extend([subj, obj])
            existing_relations.append(rel)

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations

        return node
    