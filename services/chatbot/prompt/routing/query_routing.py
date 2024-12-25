QUERY_ROUTING_PROMPT_TEMPLATE = """
-ROLE-
You are an intelligent assistant responsible for determining the appropriate retrieval type for a given query.
---------------------
-GOAL-
Classify the query into one of the following categories based on the retrieval type needed:
1. **Personal Memory Retrieval**: The query references personal or family-related past events, current experiences, or information specifically related to the assistant’s personal memories, identity, or information about the user or any other individual. This includes:
   - Queries about the assistant’s own personal information or past events.
   - Queries about personal or family-related memories or experiences.
   - Queries referencing the user or another specific individual’s information, past events, or personal context.
2. **Domain-Specific Data Retrieval**: The query involves general information or topics related to specialized domains such as medical, education, marketing, or other professional fields that are not tied to the assistant’s, user’s, or any individual’s personal or familial context.
3. **No Retrieval**: The query is a simple greeting or social chit-chat with no reference to specific past or current events, personal contexts, or specialized domain knowledge.
---------------------
-INSTRUCTIONS-
1. Analyze the query step by step using reasoning to identify whether it pertains to:
   - Personal memories, assistant-specific information, or references to the user/another individual (Retrieval 1).
   - Domain-specific topics unrelated to any personal or individual context (Retrieval 2).
   - General social conversation or greetings (Retrieval 0).
2. Follow the structured reasoning process (Chain of Thought) in examples to infer the correct category.
3. Return the result in strict JSON format with the key "retrieval_type".
4. The value of "retrieval_type" must be one of the following:
   - 0 for "No Retrieval"
   - 1 for "Personal Memory Retrieval"
   - 2 for "Domain-Specific Data Retrieval".
5. **DO NOT** include any additional information or explanations in your response.
---------------------
-STRUCTURED FORMAT-
Output:
{{
    "retrieval_type": 0 | 1 | 2
}}
---------------------
-EXAMPLES WITH CHAIN OF THOUGHT-

Example 1:
Text: "Hi there, how's your day?"
Reasoning: 
- This is a simple social greeting without referencing any specific events, personal information, or domain-specific topics.
Output:
{{
    "retrieval_type": 0
}}

Example 2:
Text: "Do you remember our trip to the beach last summer?"
Reasoning:
- The query references a specific past event ("trip to the beach last summer") that is personal in nature.
Output:
{{
    "retrieval_type": 1
}}

Example 3:
Text: "What are the symptoms of Alzheimer's disease?"
Reasoning:
- This query requests information on a specialized topic (medical domain) unrelated to any personal context.
Output:
{{
    "retrieval_type": 2
}}

Example 4:
Text: "Who are you?"
Reasoning:
- The query asks for information about the assistant’s identity, which falls under personal context.
Output:
{{
    "retrieval_type": 1
}}
---------------------
##### REAL DATA #####
---------------------
Now, classify the query:
Text: {text}
Output: 
"""
