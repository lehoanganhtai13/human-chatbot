COMMUNITY_SUMMARIZE_PROMPT_TEMPLATE = """
-ROLE-
You are an AI assistant that helps a human analyst to perform general information discovery. Information discovery is the process of identifying and assessing relevant information associated with certain entities (e.g., organizations and individuals) within a network.
---------------------
-GOAL-
Write a comprehensive report of a community, given a list of entities that belong to the community as well as their relationships and optional associated claims. The report will be used to inform decision-makers about information associated with the community and their potential impact. The content of this report includes an overview of the community's key entities, their legal compliance, technical capabilities, reputation, and noteworthy claims.
---------------------
-REPORT STRUCTURE-

The report should include the following sections:

- TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.

Return output as a well-formed **JSON-formatted** string with the following format:
{{
    "title": <report_title>,
    "summary": <executive_summary>
}}
---------------------
-EXAMPLES-

Text:
-Entities Reports-

Entity|Description
Verdant Oasis Plaza|Verdant Oasis Plaza is the location of the Unity March
Harmony Assembly|Harmony Assembly is an organization that is holding a march at Verdant Oasis Plaza

-Relationships Reports-

Source|Target|Description
Verdant Oasis Plaza|Unity March|Verdant Oasis Plaza is the location of the Unity March
Verdant Oasis Plaza|Harmony Assembly|Harmony Assembly is holding a march at Verdant Oasis Plaza
Verdant Oasis Plaza|Unity March|The Unity March is taking place at Verdant Oasis Plaza
Verdant Oasis Plaza|Tribune Spotlight|Tribune Spotlight is reporting on the Unity march taking place at Verdant Oasis Plaza
Verdant Oasis Plaza|Bailey Asadi|Bailey Asadi is speaking at Verdant Oasis Plaza about the march
Harmony Assembly|Unity March|Harmony Assembly is organizing the Unity March

Output:
{{
    "title": "Verdant Oasis Plaza and Unity March",
    "summary": "The community revolves around the Verdant Oasis Plaza, which is the location of the Unity March. The plaza has relationships with the Harmony Assembly, Unity March, and Tribune Spotlight, all of which are associated with the march event.",
}}
---------------------
##### REAL DATA #####
---------------------
Use the following text for your answer. Do not make anything up in your answer.

Text:
{input_text}

Output:
"""
