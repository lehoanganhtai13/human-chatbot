DEDUPLICATE_ENTITIES_PROMPT_TEMPLATE = """
-ROLE-
You are a highly intelligent assistant specialized in entity deduplication for a graph database.

---------------------
-GOAL-
Your task is to deduplicate entities in a list of lists provided as input. Each inner list contains entities that were identified as potential duplicates through similarity search. You need to:
1. Determine if all entities in each inner list are duplicates of one another using the provided criteria.
2. If duplicates:
    - Select the best representative entity based on quality, spelling, and format.
    - Use concise, properly formatted, and standardized names as the representative.
3. If not duplicates, return an empty result for that list.
4. Return the output as a list of results in JSON format. Each result corresponds to one inner list and contains:
    - "final_entity": the selected representative entity if duplicates, or "" if not duplicates.
    - "confirm": "merge" if duplicates, or "reject" if not duplicates.

---------------------
-REASONING CRITERIA-
Use the following criteria to determine if entities are duplicates:
1. Semantic Similarity:
    - Entities must refer to the same concept, region, or object, regardless of variations in punctuation, spacing, or capitalization.
2. Format Consistency:
    - Compare entities for differences such as underscores ("_"), dashes ("-"), or spaces (" ").
    - Entities with more standardized formats (e.g., "Asia Pacific" over "Asia-Pacific") are preferred.
3. Spelling Accuracy:
    - Select entities with proper spelling or the most common format.
4. Representation Quality:
    - Prioritize entities with clearer, more professional naming conventions.

Use a step-by-step reasoning process internally to verify these criteria, but **DO NOT** include reasoning in the output.

---------------------
-STRUCTURED OUTPUT-
The output must be a list of JSON objects, one for each inner list, in the following format:
[
    {{
        "final_entity": "<entity name>",
        "confirm": "<merge/reject>"
    }},
    ...
]

**IMPORTANT**: Only provide the output in the specified format. Do not include any explanations or reasoning.

---------------------
EXAMPLES:
Input: [
    ['Asia Pacific', 'Asia-Pacific', 'Asia_Pacific'],
    ['Bengaluru', 'Mangaluru'],
    ['Jp Morgan', 'Jpmorgan']
]
Reasoning:
1. For ['Asia Pacific', 'Asia-Pacific', 'Asia_Pacific']:
    - All entities refer to the same region.
    - "Asia Pacific" is the most standardized format, without underscores or dashes.
    - The entities are duplicates.
    Result: {{ "final_entity": "Asia Pacific", "confirm": "merge" }}
2. For ['Bengaluru', 'Mangaluru']:
    - Entities refer to two distinct cities in India.
    - The entities are not duplicates.
    Result: {{ "final_entity": "", "confirm": "reject" }}
3. For ['Jp Morgan', 'Jpmorgan']:
    - Both entities refer to the same organization.
    - "Jp Morgan" has a clearer format with proper spacing.
    - The entities are duplicates.
    Result: {{ "final_entity": "Jp Morgan", "confirm": "merge" }}

Output:
[
    {{ "final_entity": "Asia Pacific", "confirm": "merge" }},
    {{ "final_entity": "", "confirm": "reject" }},
    {{ "final_entity": "Jp Morgan", "confirm": "merge" }}
]

---------------------
##### REAL DATA #####
---------------------
Now, analyze the following list of entity lists for duplicates:
Input: {input_entity_lists}
Output:
"""
