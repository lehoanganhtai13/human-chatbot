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
-REAL DATA-
Rewrite the following sentence based on the given conversation history (if provided) or the context provided by the words preceding the quoted sentence:  
Conversation history: {conversation_history}
Input sentence: {input_sentence}
Output:
"""


SUMMARIZE_PROMPT_TEMPLATE = """
-ROLE-
You are a language model assistant tasked with summarizing a given text.
---------------------
-GOAL-
Summarize the provided text in the **input sentence** based on the context from conversation history (if provided) or the context from the preceding text. Ensure that:
1. The summary is concise and captures the main idea of the text.
2. The summary is coherent and logically structured.
3. The summary is written in your own words and does not contain verbatim text from the input.
4. **DO NOT** include any explanations, reasoning, or comments. Only return the summary of the text.
---------------------
-EXAMPLES-
Example 1:
Conversation history:
{speaker} said to {listener}, "Where is your Mom, son?" and {listener} replied, "She went out with her friends."

Input sentence: {speaker} said to {listener}, "Oh, so do you wanna eat something outside with me tonight?" and {listener} replied, "Sure, where do you want to go?"
Output: {speaker} asked {listener} to eat outside together tonight since {listener}'s Mom went out with her friends. {listener} agreed and asked where they should go.

Example 2:
Conversation history:
{speaker} said to {listener}, "Tomorrow is Sunday!" and {listener} replied, "What are your plans for the day?"

Input sentence: {speaker} said to {listener}, "I'm going to the park to read a book." and {listener} replied, "That sounds relaxing."
Output: {speaker} is going to the park to read a book on Sunday. {listener} finds it relaxing.

Example 3:
Conversation history:
{speaker} said to {listener}, "Whoops, moving these tables is harder than I thought." and {listener} replied, "Yeah, they're pretty heavy."

Input sentence: {speaker} said to {listener}, "I'm feeling tired now." and {listener} replied, "You should rest and take care of yourself."
Output: {speaker} is feeling tired after moving tables. {listener} suggested resting and self-care.

Example 4:
Conversation history: 
{speaker} said to {listener}, "I'm going out with Aunt Mary tomorrow." and {listener} replied, "That sounds fun. What are you planning to do?"

Input sentence: {speaker} said to {listener}, "Oh, we're planning to visit the new museum downtown." and {listener} replied, "That's a great idea."
Output: {speaker} is going out with Aunt Mary to visit the new museum downtown. {listener} thinks it's a great idea.
---------------------
-REAL DATA-
Summarize the following text based on the given conversation history context (if provided) or the context provided by the words preceding the quoted sentence:
Conversation history: 
{conversation_history}

Input sentence: {input_sentence}
Output: 
"""
