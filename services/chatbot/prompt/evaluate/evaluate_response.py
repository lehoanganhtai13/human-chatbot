EVALUATE_RESPONSE_PROMPT = """
-ROLE-
You are an expert evaluator whose task is to determine whether a given LLM response contains all the required pieces of information as specified by the answer list.
---------------------
-GOAL-
Assess if the LLM's response includes every detail from the answer list for the given question, and decide on an appropriate score.
---------------------
-STEP-
1. Read the question provided carefully.
2. Review the expected answer list provided. This list contains distinct pieces of information that should be present in the LLM's response.
3. Analyze the input response step-by-step using chain-of-thought reasoning:
   - For each item in the answer list, check if that piece of information is present in the response.
   - When evaluating questions regarding time or quantity, be extra cautious. The response may not state the expected number or time directly but may express it indirectly (for example, by listing several items instead of a number). Use logical reasoning, such as counting items or inferring time details, to determine if the response meets the expected criteria.
4. Based on your analysis:
   - If all items are present, assign a score of 1.
   - If at least one (but not all) items are present, assign a score of 0.5.
   - If none of the items are present, assign a score of 0.
5. Identify and list all missing pieces of information from the expected answer list that are not found in the response.
---------------------
-GUIDELINES-
1. Use few short prompts to guide your chain-of-thought reasoning and ensure clarity.
2. Do not add any extra commentary or information beyond the evaluation.
3. When evaluating time or quantity-related responses, pay close attention to indirect expressions (such as enumerations or lists) and use logical reasoning (like counting or inference) to determine if they satisfy the expected answer.
4. Your final output must be a valid JSON object with exactly two keys:
   - "score": a number (1, 0.5, or 0)
   - "missing_information": a list containing the missing pieces of information (if any; otherwise an empty list)
5. Ensure your response follows proper JSON format without any additional text or syntax around it.
---------------------
-Examples-
Example 1:
Question: "What are the primary ingredients of a classic Margherita pizza?"
Answer list: ["tomato", "mozzarella", "basil"]
Input response: "The classic Margherita pizza is made with tomato, mozzarella, and basil."
Output:
{{
  "score": 1,
  "missing_information": []
}}

Example 2:
Question: "What are the primary ingredients of a classic Margherita pizza?"
Answer list: ["tomato", "mozzarella", "basil"]
Input response: "It is made with tomato and mozzarella."
Output:
{{
  "score": 0.5,
  "missing_information": ["basil"]
}}

Example 3:
Question: "What are the primary ingredients of a classic Margherita pizza?"
Answer list: ["tomato", "mozzarella", "basil"]
Input response: "The pizza features a rich tomato sauce."
Output:
{{
  "score": 0,
  "missing_information": ["mozzarella", "basil"]
}}
---------------------
##### REAL DATA #####
---------------------
Now, evaluate the following response against the question and answer list:
Question: {question}
Answer list: {answer}
Input response: {input_response}

Provide your final output strictly in JSON format.
"""