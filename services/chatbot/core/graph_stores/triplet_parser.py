import json
import re
from typing import Any, Dict, List, Tuple

from llama_index.core.graph_stores.types import EntityNode, Relation


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
