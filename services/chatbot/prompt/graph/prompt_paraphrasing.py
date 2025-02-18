PARAPHRASE_PROMPT_TEMPLATE = """
-ROLE-
You are a language model assistant tasked with rewriting sentences.
---------------------
-GOAL-
Rewrite the quoted sentence in the **input sentence** using the given conversation history (if provided) or context from the preceding text. Ensure that:  
1. All pronouns (e.g., you, I, we) and possessive adjectives (e.g., your, his, her) in the quoted sentence are replaced with specific names, objects, or entities.  
2. If the conversation history or context does not provide specific entities, infer from the input sentence while ensuring no pronouns or possessive adjectives remain.  
3. Ensure the rewritten sentence maintains correct grammar.
4. **DO NOT** include any explanations, reasoning, or comments. Only return the rewritten sentence from quoted sentence as the output.
---------------------
-EXAMPLES-  
Example 1:  
Conversation history:
Input sentence: {speaker} said to {listener}, "Where are we now?"
Output: Where are {listener} and {speaker} now?

Example 2:  
Conversation history: {listener} said to {speaker}, "I just got a new book." 
Input sentence: {listener} said to {speaker}, "Where is it?"
Output: Where is the book?

Example 4:  
Conversation history: {speaker} said to {listener}, "I'm struggling with my homework."
Input sentence: {speaker} said to {listener}, "Do you know how to solve it, help me?"
Output: Does {listener} know how to solve {speaker}'s homework?
---------------------
##### REAL DATA #####
---------------------
Rewrite the following sentence based on the given conversation history (if provided) or the context provided by the words preceding the quoted sentence:  
Conversation history: {conversation_history}
Input sentence: {input_sentence}
Output:
"""
