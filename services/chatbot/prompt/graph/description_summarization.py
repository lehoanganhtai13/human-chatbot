DESCRIPTION_SUMMARIZE_PROMPT_TEMPLATE = """
-ROLE-
You are an advanced language model tasked with summarizing and combining descriptions, with a specific focus on including any missing critical information identified between the old and new descriptions.
---------------------
-GOAL-
Summarize the old and new descriptions into a single, concise description. Ensure the summarized description:
1. Retains all critical information from both descriptions.
2. Focuses specifically on any missing information identified.
3. Combines relevant information and eliminates redundancy while maintaining semantic clarity.
---------------------
-GUIDELINES-
1. **Missing Information Focus:**
   - Start by incorporating any critical missing information explicitly provided (pre-checked).
   - Ensure that these details are seamlessly integrated into the summarized description.
2. **Prioritization of Information:**
   - Include the **most significant details** from both descriptions, focusing on roles, events, dates, and unique contributions.
   - Avoid duplicating information or introducing trivial details.
3. **Clarity and Conciseness:**
   - Limit the final description to **3-4 sentences**, ensuring it is both clear and concise.
   - Use professional and neutral language with grammatical correctness.
4. **Reasoning Approach:**
   - During reasoning (examples), identify how missing information and existing details are combined logically.
   - For real data output, omit reasoning and directly return the summarized description.
5. **Output Format:**
   - Provide **only the summarized description** in the final output.
---------------------
-EXAMPLES-
Example 1:
Old Description: John Smith is a Senior Engineer at Apple Inc. who joined in 2020. He leads the iOS development team and contributed to Face ID development.

New Description: John Smith, a Senior Engineer at Apple Inc., has been instrumental in leading the iOS development team since 2020 and played a key role in launching innovative features like Face ID.

Missing Information: "instrumental in leading," "launching innovative features."

Reasoning:
1. Common details: "John Smith," "Senior Engineer," "Apple Inc.," "iOS development team," "Face ID," and "2020."
2. Missing details provided: "instrumental in leading" and "launching innovative features."
3. Summarized result: Combines all critical details from both descriptions while integrating the missing elements.

Summarized Description: John Smith, a Senior Engineer at Apple Inc. since 2020, leads the iOS development team and has been instrumental in launching innovative features like Face ID.

Example 2:
Old Description: Face ID is a facial recognition technology launched by Apple’s iOS development team on September 12, 2017.

New Description: Apple's iOS development team introduced Face ID, a groundbreaking facial recognition feature, on September 12, 2017, revolutionizing user authentication.

Missing Information: "groundbreaking feature," "revolutionizing user authentication."

Reasoning:
1. Common details: "Face ID," "iOS development team," "September 12, 2017," and "facial recognition technology."
2. Missing details provided: "groundbreaking feature" and "revolutionizing user authentication."
3. Summarized result: Combines critical details from both descriptions while emphasizing missing elements.

Summarized Description: Face ID, introduced by Apple's iOS development team on September 12, 2017, is a groundbreaking facial recognition technology that revolutionized user authentication.
---------------------
##### REAL DATA #####
---------------------
Old Description: {old_description}

New Description: {new_description}

Missing Information: {missing_information}

Summarized Description:
"""


CHECK_DESCRIPTION_INCLUSION_PROMPT_TEMPLATE = """
-ROLE-
You are an advanced reasoning engine tasked with analyzing and comparing descriptions for knowledge graph entities or relationships.
---------------------
-GOAL-
Your task is to determine if the new description is fully contained within the old description. If every critical piece of information in the new description appears in the old description, return "YES". Otherwise, return "NO" and explicitly state what critical information is missing.
---------------------
-GUIDELINES-
1. **Key Comparison Guidelines:**
    - Focus **only** on critical information explicitly stated in the new description. 
    - Do **not** include information that is only present in the old description but absent in the new description.
    - Compare each piece of information in the new description carefully to the old description; any missing element should be noted.
    - If multiple pieces of information are missing, rank them in order of importance.

2. **Reasoning Process:**
    - Use a step-by-step reasoning process (Chain of Thought) to verify each piece of information in the new description exist in the old description:
        - Step 1: Identify all relevant facts from the **new description only**.
        - Step 2: Compare each fact to the old description, checking for exact or partial matches.
        - Step 3: For any information present in the new description but missing from the old description, record it as missing.
        - Step 4: Consolidate all missing details and finalize the output (if any).

    - Follow the examples below to understand how reasoning is applied, but exclude the reasoning text in the final output.

3. **Output Format:** Strictly JSON, structured as follows, without adding any extra text or syntax around it. :
   {{
       "result": "YES" or "NO",
       "missing_information": [
           "<missing information 1>",
           "<missing information 2>",
           ...
       ]
   }}

4. **Rules for Missing Information:**
    - If "result" is "YES", "missing_information" must be an empty array.
    - If "result" is "NO", list all critical missing pieces from the **new description only** in concise form (10-15 words each).

5. **Validation:**
    - Double-check that all critical information in the new description has been accounted for.
    - Ensure the final output aligns strictly with the JSON format and reflects the new description accurately.
---------------------
-EXAMPLES-
Example 1:
Old Description: "John Smith works at Apple Inc. as a Senior Engineer. He joined in 2020."
New Description: "John Smith is a Senior Engineer at Apple Inc. since 2020."

Reasoning:
1. The new description states that John Smith is a Senior Engineer at Apple Inc., which matches the old description.
2. Both descriptions include the temporal information "since 2020," so there is no missing information.
3. All critical elements of the new description are fully contained within the old description.

Output:
{{
    "result": "YES",
    "missing_information": []
}}

Example 2:
Old Description: "John Smith works at Apple Inc. as a Senior Engineer. He joined in 2020."
New Description: "John Smith is a Senior Engineer at Apple Inc. leading the iOS team."

Reasoning:
1. The old description states that John Smith is a Senior Engineer at Apple Inc., which matches part of the new description.
2. The new description adds the information that John Smith is leading the iOS team, which is not mentioned in the old description.
3. This is a critical detail missing from the old description.

Output:
{{
    "result": "NO",
    "missing_information": [
        "John Smith leads the iOS team"
    ]
}}

Example 3:
Old Description: "Face ID was implemented by the iOS development team on September 12, 2017."
New Description: "The iOS development team launched Face ID in 2017."

Reasoning:
1. The old description states that Face ID was implemented by the iOS team on a specific date (September 12, 2017).
2. The new description generalizes the time to "2017," but this information is already covered in the old description.
3. No additional critical information is introduced in the new description, and all details are contained within the old description.

Output:
{{
    "result": "YES",
    "missing_information": []
}}
---------------------
##### REAL DATA #####
---------------------
Old Description: {old_description}
New Description: {new_description}

Output:
"""
