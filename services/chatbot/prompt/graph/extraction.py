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
7. **DO NOT** include any additional information or explanation in your response.
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

Text: 
Output: What is the school schedule for next week?
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
1. **Output Format:** Strictly JSON, structured as follows:
   [
       {{
           "head": "", 
           "head_type": "", 
           "head_props": {{}}, 
           "relation": "", 
           "relation_props": {{}}, 
           "tail": "", 
           "tail_type": "", 
           "tail_props": {{}}
       }},
       ...
   ]
2. **Entity Naming:**
    - Use the most complete form for entities (e.g., "United States of America" instead of "USA").
    - Keep entities concise (3-5 words maximum).
3. **Triplet Extraction:**
    - Break down complex phrases into multiple triplets.
    - Ensure the extracted knowledge graph is coherent and easily understandable.
4. **Relation Formatting:**
    - Always use uppercase letters with underscores (_) for `relation` (e.g., "CEO_OF" instead of "CEO of").
5. **Ontology Adherence:**
    - Extract triplets based on the initial ontology, but introduce new types if necessary based on the text's context.
6. **Validation:**
    - Double-check for important or missed information.
    - Pay special attention to proper names, events, time, and dates. For example, from "I want to play games with you guys at 7 pm," extract entities like "7 pm," "play games," and "games."
7. **Additional Rules:**
    - Do not create relationships without their corresponding entities.
    - Do not allow duplicated inverse relationships, for example, if you have a relationship "OWNS" from Person to House, do not create another relationship "OWNED_BY" from House to Person.
    - Relationship names must be timeless. For example "WROTE" and "WRITTEN" means the same thing, if the source and target entities are the same. Remove similar scenarios.
---------------------
-EXAMPLE-
Entity Types: ["PERSON", "COMPANY", "TECHNOLOGY", "ROLE", "DEPARTMENT"]
Entity Properties: ["description", "role"]
Relation Types: ["WORKS_AT", "LEADS", "IMPLEMENTED"]
Relation Properties: ["description", "since", "position"]

Text: John Smith joined Apple Inc. as Senior Engineer in 2020. He leads the iOS development team and has implemented several key features.

Output:
[
    {{
        "head": "John Smith",
        "head_type": "PERSON",
        "head_props": {{
            "description": "John Smith is Senior Engineer at Apple Inc. who joined in 2020. He currently leads the iOS development team and has contributed to implementing several key features. He holds a leadership position in iOS development.",
            "role": "Senior Engineer",
            ...
        }},
        "relation": "WORKS_AT",
        "relation_props": {{
            "description": "John Smith is employed as Senior Engineer since 2020, leading the iOS development team. He has made significant contributions through implementation of key features. His position involves both technical leadership and development responsibilities.",
            "since": "2020",
            ...
        }},
        "tail": "Apple Inc.",
        "tail_type": "COMPANY",
        "tail_props": {{
            "description": "Apple Inc. is a technology company that employs John Smith as Senior Engineer in their iOS development team. The company has an iOS development division where key features are being implemented under John Smith's leadership.",
            ...
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
