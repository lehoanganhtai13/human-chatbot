from typing import List

from llama_index.core import get_response_synthesizer, PromptTemplate
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.schema import NodeWithScore
from llama_index.core.types import RESPONSE_TEXT_TYPE

from chatbot.core.chat_stores import CacheChatStore
from chatbot.core.model_clients import LLMCore
from chatbot.prompt.response.qa_prompt import (
    QA_PROMPT_TEMPLATE_WITH_CONTEXT, QA_PROMPT_TEMPLATE_WITHOUT_CONTEXT,
    CUSTOM_AVATAR_PROMPT_TEMPLATE_WITH_CONTEXT, CUSTOM_AVATAR_PROMPT_TEMPLATE_WITHOUT_CONTEXT
)

# See: https://docs.llamaindex.ai/en/stable/module_guides/querying/response_synthesizers/#custom-response-synthesizers and 
# https://docs.llamaindex.ai/en/v0.10.20/examples/low_level/response_synthesis.html#try-a-simple-prompt
class Synthesizer():
    """
    A class to synthesize responses using the LLM model directly using context and query strings.

    llm : LLMCore
        The LLM model to use for response synthesis.
    text_qa_template : BasePromptTemplate
        The template to use for formatting the QA prompt.
    streaming : bool
        Whether to use streaming or not.

    Example:
    >>> text_qa_template = qa_prompt.partial_format(max_num_tokens=max_new_tokens)
    >>> synthesizer = Synthesizer(llm=llm, text_qa_template=text_qa_template, streaming=True)
    >>> query_str = "What is the capital of France?"
    >>> retrieved_nodes = [NodeWithScore(node=TextNode(text="text"), score=0)]
    >>> async_generator = await synthesizer.asynthesize(query_str, retrieved_nodes)
    >>> async for chunk in async_generator:
    >>>     print(chunk.delta, end="", flush=True)
    """

    def __init__(self, llm: LLMCore = None, text_qa_template: BasePromptTemplate = None, streaming: bool = False):
        self._llm = llm
        self._streaming = streaming
        self._text_qa_template = text_qa_template

    async def asynthesize(self, query_str: str, retrieved_nodes: List[NodeWithScore]) -> RESPONSE_TEXT_TYPE:
        context_str = "\n\n".join([r.get_content() for r in retrieved_nodes])
        fmt_qa_prompt = self._text_qa_template.format(
            context_str=context_str, query_str=query_str
        )
        if not self._streaming:
            response = await self._llm.acomplete(fmt_qa_prompt)
        else:
            response = await self._llm.astream_complete(fmt_qa_prompt)

        return response
    
    def synthesize(self, query_str: str, retrieved_nodes: List[NodeWithScore]) -> RESPONSE_TEXT_TYPE:
        context_str = "\n\n".join([r.get_content() for r in retrieved_nodes])
        fmt_qa_prompt = self._text_qa_template.format(
            context_str=context_str, query_str=query_str
        )
        if not self._streaming:
            response = self._llm.complete(fmt_qa_prompt)
        else:
            response = self._llm.stream(fmt_qa_prompt)

        return response


# See: https://docs.llamaindex.ai/en/stable/module_guides/querying/response_synthesizers/response_synthesizers/
class  Generator():
    def __init__(
            self,
            llm: LLMCore,
            chat_store: CacheChatStore,
            max_new_tokens: int = 256,
            streaming: bool = False,
            response_mode: ResponseMode = ResponseMode.SIMPLE_SUMMARIZE,
            assistant_name : str = "Minh",
            query_delimiter: str = "<|>"
    ):
        llm.chat_store = chat_store
        llm.chat_delimiter = query_delimiter

        self.qa_prompt_with_context = PromptTemplate(QA_PROMPT_TEMPLATE_WITH_CONTEXT).partial_format(max_num_tokens=max_new_tokens)
        self.qa_prompt_without_context = PromptTemplate(QA_PROMPT_TEMPLATE_WITHOUT_CONTEXT).partial_format(max_num_tokens=max_new_tokens)
        if assistant_name != "Minh":
            self.qa_prompt_with_context = PromptTemplate(CUSTOM_AVATAR_PROMPT_TEMPLATE_WITH_CONTEXT).partial_format(
                max_num_tokens=max_new_tokens, assistant_name=assistant_name
            )
            self.qa_prompt_without_context = PromptTemplate(CUSTOM_AVATAR_PROMPT_TEMPLATE_WITHOUT_CONTEXT).partial_format(
                max_num_tokens=max_new_tokens, assistant_name=assistant_name
            )

        self.stream_generator = get_response_synthesizer(
            llm=llm,
            text_qa_template=self.qa_prompt_with_context,
            streaming=streaming,
            response_mode=response_mode,
            use_async=True
        )
        self.completion_generator = get_response_synthesizer(
            llm=llm,
            text_qa_template=self.qa_prompt_with_context,
            streaming=False,
            response_mode=ResponseMode.SIMPLE_SUMMARIZE,
            use_async=True
        )
        self.streaming = streaming
        self.query_delimiter = query_delimiter

    async def generate(self, query: str, nodes: List[NodeWithScore], language: str = "english", new_prompt: bool = False, kwargs: dict = None) -> RESPONSE_TEXT_TYPE:
        if self.streaming:
            # Update the prompts based on the context
            if len(nodes) == 1 and nodes[0].node.text == "":
                self.stream_generator.update_prompts({"text_qa_template": self.qa_prompt_without_context.partial_format(language=language)})
            else:
                self.stream_generator.update_prompts({"text_qa_template": self.qa_prompt_with_context.partial_format(language=language)})
                
            response = await self.stream_generator.asynthesize(f"{self.query_delimiter}{query}", nodes)
            return response.response_gen
        
        # Update the prompts based on the context
        if len(nodes) == 1 and nodes[0].node.text == "":
            self.completion_generator.update_prompts({"text_qa_template": self.qa_prompt_without_context.partial_format(language=language)})
        else:
            self.completion_generator.update_prompts({"text_qa_template": self.qa_prompt_with_context.partial_format(language=language)})
            
        response = await self.completion_generator.asynthesize(f"{self.query_delimiter}{query}", nodes)
        return response.response
