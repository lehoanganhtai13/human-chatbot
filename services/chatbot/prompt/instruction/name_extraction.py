ASSISTANT_NAME_EXTRACTION_PROMPT = """
-ROLE-
You are responsible for extracting the name of the assistant from given instructions.
---------------------
-GOAL-
Extract ONLY the explicit name of the assistant if mentioned in the instructions. Return empty string if no name is specified.
---------------------
-GUIDELINES-
1. Look for phrases like "You are [NAME]" or similar that directly identify the assistant.
2. Only extract explicit names, not roles or titles.
3. If multiple names appear, only extract the one that refers to the assistant.
4. Return an empty string if no explicit assistant name is found.
5. Return result in strict JSON format with "name" key.
6. **DO NOT** include any explanations or additional text.
---------------------
-EXAMPLES-
Example 1:
Instructions: You are Minh, a dedicated researcher living in Ho Chi Minh City.
Output:
{{
    "name": "Minh"
}}

Example 2:
Instructions: You are a compassionate and experienced professional specializing in Alzheimer's care.
Output:
{{
    "name": ""
}}

Example 3:
Instructions: As Dr. Smith's assistant, you help patients understand their treatment plans.
Output:
{{
    "name": ""
}}
---------------------
##### REAL DATA #####
---------------------
Now, extract the name of the assistant from the following instructions:
Instructions:
{instruction}
Output:
"""
