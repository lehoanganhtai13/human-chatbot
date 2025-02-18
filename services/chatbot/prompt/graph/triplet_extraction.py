EXTRACT_ENTITIES_PROMPT_TEMPLATE = """
-ROLE-
You are a helpful assistant responsible for extracting important entities from the text given below.
---------------------
-GOAL-
Your task is to identify and extract entities to assist with node retrieval in a knowledge graph database. Ensure that:
1. Extract entities strictly based on the schema
2. When text contains semantically similar terms or phrases:
    - If text mentions "teacher" and shema list includes "professor, MUST use the exact name "professor"
    - If text mentions "school" and schema includes "university", MUST use the exact name "university"
    - If text mentions a compound phrase like "school party" and schema includes "university party" and "university":
        * Extract ALL entities from the shema which are "university party" and "university".
    - MUST use exact names from schema
3. Limit extraction to maximum {max_extracted_entities} most relevant entities, prioritizing:
   - Entities explicitly mentioned in or closely related to the text
   - Entities from types directly referenced in the text
   - Time and event entities when context is time-sensitive
   - Core entities over peripheral ones
4. Only use **EXACT** entity names from the schema - never modify or paraphrase them.
5. When text mentions anything related to a specific entity type (e.g., asking about time or education), include ALL entities of that type from the schema.
6. If no entities in the text match the provided names directly or semantically, return an empty list in the structured format.
7. Use reasoning internally to identify relevant entities, but **DO NOT** include any reasoning, explanations, or intermediate steps in your response.
8. Only return the JSON output, nothing else.
---------------------
-STRUCTURED FORMAT-
class Entities(BaseModel):
    names: Optional[List[str]]  # Maximum {max_extracted_entities} entities
---------------------
EXAMPLE:
(These examples demonstrate the logic but your responses should be based on actual context)

Example 1:
Schema information (entity types with existing entity names):
TIME: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Morning', 'Afternoon', 'Evening']
EVENT: ['Class', 'Study Session', 'Group Meeting', 'Office Hours', 'Lab Work']
LOCATION: ['Library', 'Lab Room', 'Lecture Hall', 'Study Room', 'Cafeteria']

Text: When are the morning classes next week?
Reasoning:
- The query explicitly mentions "morning" (a time entity) and "classes" (an event entity).
- The mention of "next week" implies a general time period, so all weekdays ("Monday" to "Friday") are relevant.
- Classes are likely to occur in predefined locations, so "Lecture Hall" is included as a common location.
Output:
{{
    "names": [
        "Morning",
        "Class",
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
        "Lecture Hall"
    ]
}}

Example 2:
Schema information (entity types with existing entitiy names):
EDUCATION: ['Harvard University', 'Computer Science', 'Graduate thesis']
EVENT: ['Examination', 'Shopping', 'Meeting']

Text: What is the school schedule for next week?
Reasoning:
- "School schedule" suggests academic entities. Map "school" to relevant education entities in the schema: "Harvard University," "Computer Science," "Graduate thesis."
- Schedule likely includes events such as "Examination."
Output:
{{
    "names": [
        "Harvard University", "Computer Science", "Graduate thesis", "Examination"
    ]
}}

Example 3:
Schema information (entity types with existing entitiy names):
LOCATION: ['coffeeshop', 'library', 'park', 'movie theater', 'John\'s father\'s house']
PERSON: ['John', 'John\'s father', 'best friend']
EVENT: ['watching movies', 'reading books', 'meeting friends']

Text: Hey do you want to watch somethinga at new John's Dad's house?
Reasoning:
- "John's Dad's house" corresponds to the schema location "John's Father's House."
- "Watch something" maps to the event "Watching Movies."
- "John's Dad" matches the person "John's Father."
Output:
{{
    "names": [
        "watching movies", "John's father's house", "John's father"
    ]
}}

Example 4:
Schema information (entity types with existing entitiy names):
TIME: ['Tuesday', 'Tonight', '4 pm', 'next week']
EVENT: ['meeting', 'presentation', 'party', 'company trip']

Text: When will the next company trip occur?
Reasoning:
- "Next company trip" explicitly maps to "Company Trip" in the schema.
- The query implies time-related entities such as "Next Week," "Tuesday," "Tonight," and "4 pm."
Output:
{{
    "names": [
        "Tuesday", "Tonight", "4 pm", "next week", "company trip"
    ]
}}
---------------------
##### REAL DATA #####
---------------------
Now, extract entities:
Schema information (entity types with existing entitiy names):
{schema}

Text: {text}
Output:
"""


EXTRACT_GRAPH_TRIPLETS_PROMPT_TEMPLATE = """
-ROLE-
You are a knowledge graph expert tasked with extracting knowledge triplets from the provided text.
---------------------
-GOAL-
Your task is to extract up to {max_knowledge_triplets} knowledge triplets from the given text. Each triplet should be in the form of (head, relation, tail), with their respective types and properties.
---------------------
-GUIDELINES-
1. **Prioritization of Information:**
    - Prioritize the extraction of triplets containing **critical information** (e.g., events, roles, time-related details).
    - Rank extracted triplets by importance: the most significant triplets should appear first.
2. **Handling Time Information:**
    - Explicitly capture temporal details (e.g., start date, duration, timeline) in the `relation_props` or `entity_props` fields whenever present in the text.
    - Ensure time information is accurate and justified based on the text.
3. **Exclusion of Uncertain Entities:**
    - Do not extract triplets if any entity (either the `head` or `tail`) is ambiguous (such as "unknown" or "undefined") or lacks sufficient information.
    - Ensure all entities have meaningful and specific names.
4. **Reasoning Process:**
    - Use a step-by-step reasoning approach (Chain of Thought) to identify entities, types, and their relationships.
    - Clearly explain why a relationship exists between the `head` and `tail` entities and justify it in the `relation_props.description`.
5. **Output Format:** Strictly JSON, structured as follows:
   [
       {{
           "head": "", 
           "head_type": "", 
           "head_props": {{
               "description": ""
           }}, 
           "relation": "", 
           "relation_props": {{
               "description": ""
           }}, 
           "tail": "", 
           "tail_type": "", 
           "tail_props": {{
               "description": ""
           }}
       }} 
       ...
   ]
6. **Entity Naming:**
    - Use the most complete form for entities (e.g., "United States of America" instead of "USA").
    - Keep entities concise (3-5 words maximum).
7. **Entity Name Unification:**
    - If a single entity is referenced by multiple names (e.g., "James Smith" and "James"), choose the most complete form (e.g., "James Smith") as the canonical name.
    - Additionally, extract an extra triplet to capture the alias relationship between the alternate name and the canonical name. In this triplet, use the alias name as the head, "ALIAS_OF" as the relation, and the canonical name as the tail. Include a brief explanation in `relation_props.description` indicating that the alias refers to the canonical entity.
8. **Relation Formatting:**
    - Always use uppercase letters with underscores (_) for `relation` (e.g., "CEO_OF" instead of "CEO of").
9. **Validation:**
    - Double-check for missed or redundant information.
    - Ensure the extracted knowledge graph is coherent and accurately reflects the text.
10. **Relation Justification:**
    - Clearly describe the reason for the relation in `relation_props.description` by referencing relevant information in the text.
    - Avoid ambiguous or generic descriptions. Include temporal, functional, or causal reasoning if applicable.
11. **Additional Rules:**
    - **Properties Restriction:**
        - Extract **only** the `description` property for both `entity_props` and `relation_props`.
        - Do not add any additional properties beyond `description`, even if they are present in the initial `allowed_entity_properties` or `allowed_relation_properties`.
    - **Entity Types Flexibility:**
        - The initial list of `entity_types` is only a suggestion. You are allowed to introduce new `entity_types` if the text justifies their inclusion.
    - Do not create relationships without their corresponding entities.
    - Avoid duplicated inverse relationships (e.g., if "OWNS" exists, do not create "OWNED_BY").
    - Use timeless and semantically consistent relationship names (e.g., "WRITES" instead of "WROTE").
    - **DO NOT** include any reasoning, explanations, or intermediate steps in your response.
    - **ONLY** return the JSON output, nothing else.
---------------------
-EXAMPLE-
Entity Types: ["PERSON", "COMPANY", "TECHNOLOGY", "ROLE", "EVENT", "DATE", "DEPARTMENT"]
Entity Properties: ["description"]
Relation Types: ["WORKS_AT", "LEADS", "IMPLEMENTED", "OCCURRED_ON"]
Relation Properties: ["description"]

Text: In 2020, John, whose full name is John Smith, joined Apple Inc. as Senior Engineer. He leads the iOS development team and implemented Face ID. The iOS development team launched Face ID on September 12, 2017.

Output:
[
    {{
        "head": "John Smith",
        "head_type": "PERSON",
        "head_props": {{
            "description": "John Smith is Senior Engineer at Apple Inc. who joined in 2020. He currently leads the iOS development team and contributed to Face ID development.",
        }},
        "relation": "WORKS_AT",
        "relation_props": {{
            "description": "John Smith works at Apple Inc. as a Senior Engineer since 2020.",
        }},
        "tail": "Apple Inc.",
        "tail_type": "COMPANY",
        "tail_props": {{
            "description": "Apple Inc. is a technology company employing John Smith as Senior Engineer."
        }}
    }},
    {{
        "head": "John",
        "head_type": "PERSON",
        "head_props": {{
            "description": "John is an abbreviated form used for John Smith."
        }},
        "relation": "ALIAS_OF",
        "relation_props": {{
            "description": "Indicates that 'John' is an alias for the canonical name John Smith."
        }},
        "tail": "John Smith",
        "tail_type": "PERSON",
        "tail_props": {{
            "description": "John Smith is the full name corresponding to the alias 'John'."
        }}
    }},
    ...
]
---------------------
##### REAL DATA #####
---------------------
-INITIAL ONTOLOGY-
Entity Types: {allowed_entity_types}
Entity Properties: {allowed_entity_properties}
Relation Types: {allowed_relation_types}
Relation Properties: {allowed_relation_properties}

Text: {text}
Output:
"""


CONTINUE_PROMPT = """
MANY triplets were missed in the last extraction. Add **ONLY NEW TRIPLETS** below using the same format instructed earlier. Follow these rules:
1. ONLY add triplets where the **relation between the two entities is NEW** (not already extracted before).
2. **DO NOT** add triplets with the same or similar meaning as previously extracted ones.
3. If no new triplets are found, return "NO NEW TRIPLETS".

New triplets:
"""


LOOP_PROMPT = """
STRICTLY answer ONLY 'YES' or 'NO' in UPPERCASE based on:
- YES: If there are **NEW TRIPLETS** meeting ALL conditions:
  1. Contains relations NOT EXISTING between the same entity pair in previous extractions
  2. Information is NOT SEMANTICALLY SIMILAR to existing triplets
- NO: Otherwise

DO NOT:
- Add explanations/punctuations
- Consider duplicate/rephrased triplets
- Output anything except YES/NO

Question: Are there qualifying new triplets?
"""
