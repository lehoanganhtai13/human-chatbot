MESSAGE_SUMMARIZE_PROMPT_TEMPLATE = """
-ROLE-
You are a language model assistant tasked with summarizing a given text.
---------------------
-GOAL-
Summarize the provided text in the **input sentence** based on the context from conversation history (if provided) or the context from the preceding text. Ensure that:
1. The summary is concise and captures the main idea of the text.
2. The summary is coherent and logically structured.
3. The summary is written in your own words and does not contain verbatim text from the input.
4. Include all details related to time, specific hours, names, or any crucial entities without omitting them.
5. **Explicitly specify names or clear identities:** For any person, object, event, or crucial entity mentioned in the original text, if it is referred to using a pronoun (e.g., "he", "she", "it"), then in your summary, explicitly state the full name or identity on its first mention. Only use pronouns in subsequent mentions once the identity has been clearly established.
6. Use chain-of-thought reasoning internally while summarizing but **DO NOT** include reasoning, explanations, or comments in the final output.
---------------------
-EXAMPLES-
(The examples below are for reference only and should not be used in the response)

Example 1:
Conversation history:
{speaker} said to {listener}, "Where is your Mom, son?" and {listener} replied, "She went out with her friends."

Input sentence: {speaker} said to {listener}, "Oh, so do you wanna eat something outside with me tonight at 7 PM?" and {listener} replied, "Sure, where do you want to go?"
Output: {speaker} asked {listener} to eat outside together tonight at 7 PM since {listener}'s Mom went out with her friends. {listener} agreed and asked where they should go.

Example 2:
Conversation history:
{speaker} said to {listener}, "Tomorrow is Sunday!" and {listener} replied, "What are your plans for the day?"

Input sentence: {speaker} said to {listener}, "I'm going to the park to read a book at 9 AM." and {listener} replied, "That sounds relaxing."
Output: {speaker} is going to the park to read a book at 9 AM on Sunday. {listener} finds it relaxing.

Example 3:
Conversation history:
{speaker} said to {listener}, "Whoops, moving these tables is harder than I thought." and {listener} replied, "Yeah, they're pretty heavy."

Input sentence: {speaker} said to {listener}, "I'm feeling tired now after working for 3 hours." and {listener} replied, "You should rest and take care of yourself."
Output: {speaker} is feeling tired after moving tables for 3 hours. {listener} suggested resting and self-care.

Example 4:
Conversation history: 
{speaker} said to {listener}, "I'm going out with Aunt Mary tomorrow." and {listener} replied, "That sounds fun. What are you planning to do?"

Input sentence: {speaker} said to {listener}, "Oh, we're planning to visit the new museum downtown at 2 PM." and {listener} replied, "That's a great idea."
Output: {speaker} is going out with Aunt Mary to visit the new museum downtown at 2 PM. {listener} thinks it's a great idea.

Example 5 (Demonstrating explicit naming):
Conversation history:
{speaker} said to {listener}, "I'm going to prepare a gift for your grand father." and {listener} replied, "Then, what will you give him?"

Input sentence: {speaker} said to {listener}, "A new 40 Inch TV. He's always wanted one for a long time." and {listener} replied, "He'll appreciate that a lot, Dad! When will you give it to him?"
Output: {speaker} plans to give a new 40 Inch TV to {listener}'s grand father, who has long desired one. {listener} believes {listener}'s grand father will love it and asks when it will be given.
---------------------
##### REAL DATA #####
---------------------
Summarize the following text based on the given conversation history context (if provided) or the context provided by the words preceding the quoted sentence:
Conversation history: 
{conversation_history}

Input sentence: {input_sentence}
Output: 
"""
