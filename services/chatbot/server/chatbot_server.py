import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
import time
import traceback

from llama_index.core import Document, PromptTemplate
from llama_index.core.schema import TextNode
from llama_index.core.indices.property_graph import DynamicLLMPathExtractor
from llama_index.core.schema import NodeWithScore

from chatbot.config.system_config import SETTINGS
from chatbot.data.character_story import DR_CHOI_REWRITE
from chatbot.utils.chat_store import CacheChatStore, PersistentChatStore
from chatbot.utils.generator import Generator, ResponseMode
from chatbot.utils.graph_retriever import Retriever, CustomSubRetriever
from chatbot.utils.graph_store import FalkorDBGraphStore, parse_dynamic_triplets_with_props
from chatbot.utils.models_client import EmbedderCore, LLMCore
from chatbot.utils.predefined_entities import (
    ENTITY_PROPERTIES,
    ENTITY_TYPES,
    RELATION_PROPERTIES,
    RELATION_TYPES,
)
from chatbot.utils.translator import Translator

from chatbot.prompt.graph.extraction import (
    EXTRACT_ENTITIES_PROMPT_TEMPLATE,
    EXTRACT_GRAPH_TRIPLETS_PROMPT_TEMPLATE
)
from chatbot.prompt.instruction.summary import INSTRUCTION_SUMMARY_PROMPT
from chatbot.prompt.instruction.extraction import ASSISTANT_NAME_EXTRACTION_PROMPT
from chatbot.prompt.routing.query_routing import QUERY_ROUTING_PROMPT_TEMPLATE
from chatbot.prompt.translate.translate import TRANSLATION_PROMPT


class ChatbotServer:
    def __init__(self, user_id: str, avatar_name: str = "Choi", use_default_story: bool = True, avatar_instruction_text: str = "", warm_up: bool = True):

        # Define the user ID, assistant name, and summarized user ID and assistant ID for message summarization and query transformation
        if not use_default_story:
            self.user_id = user_id
            self.assistant_id = avatar_name
            self.assistant_name = None
            self.summarized_user_id = "the user"
            self.summarized_assistant_id = "the assistant"
        else:
            self.user_id = user_id
            self.assistant_id = avatar_name
            self.assistant_name = "Choi"
            self.summarized_user_id = "David"
            self.summarized_assistant_id = "Choi"

        # Load the LLM config
        self.llm_config = {}
        with open("./chatbot/config/llm_config.json", "r") as f:
            self.llm_config = json.load(f)
            print("Loaded LLM configurations:", self.llm_config)

        print("Initializing the translator and language detector...")
        self.en_translator = Translator(
            source="auto", target="english", capitalize_sentences=True
        )
        print("Warm up the translator...")
        # Warm up the translator
        translate_time = 0.5
        max_retry = 10
        while translate_time > 0.3 and max_retry > 0:
            start_translate_time = time.time()
            self.en_translator.translate("Warm up the translator")
            translate_time = time.time() - start_translate_time

        print(f"Initializing the chat store in host {SETTINGS.CHAT_STORE_HOST}...")
        memory_summarize_llm = self.init_llm("memory_summarize_llm")
        self.cache_chat_store = CacheChatStore(
            host=SETTINGS.CHAT_STORE_HOST,
            port=int(SETTINGS.CHAT_STORE_PORT),
            db=int(SETTINGS.CHAT_STORE_DB),
            username=SETTINGS.CHAT_STORE_USERNAME,
            password=SETTINGS.CHAT_STORE_PASSWORD,
            live_time_seconds=int(SETTINGS.CHAT_STORE_TTL),
            max_messages_pairs=int(SETTINGS.CHAT_STORE_MAX_MESSAGES_PAIRS),
            llm=memory_summarize_llm,
        )

        print("Initializing the persistent chat store...")
        self.persistent_chat_store = PersistentChatStore(
            uri=f"mongodb://{SETTINGS.PERSISTENT_CHAT_STORE_USERNAME}:{SETTINGS.PERSISTENT_CHAT_STORE_PASSWORD}@{SETTINGS.PERSISTENT_CHAT_STORE_HOST}:{SETTINGS.PERSISTENT_CHAT_STORE_PORT}",
            db_name=SETTINGS.PERSISTENT_CHAT_STORE_DB,
            collection_name=SETTINGS.PERSISTENT_CHAT_STORE_COLLECTION,
            use_async=False,
        )
        
        print("Initializing the LLM instruction summarizer...")
        instruction_summarize_llm = self.init_llm("instruction_summarize_llm")

        # Build a new graph memory if the assistant is not Choi and the avatar instruction text is given
        character_stories = []
        translated_instruction = ""
        if not use_default_story:
            try:
                print("Initializing the LLM translator...")
                translate_llm = self.init_llm("translate_llm")
                translate_time = time.time()
                translated_instruction = translate_llm.complete(
                    prompt=TRANSLATION_PROMPT.format(language="English", text=avatar_instruction_text.replace("\n", " ").strip())
                ).text
                print(f"Time taken to translate the instruction text: {time.time() - translate_time:.4f} seconds")
                print(f"Translated instruction text: {translated_instruction}")
            except Exception as e:
                print(f"Failed to translate the instruction text: {e}")

            if translated_instruction != "":
                summarized_instruction = instruction_summarize_llm.complete(
                    prompt=INSTRUCTION_SUMMARY_PROMPT.format(instruction=translated_instruction)
                ).text

                split_instructions_list = summarized_instruction.split("/---------------------/")
                for instruction in split_instructions_list:
                    sentences = [sentence.strip() for sentence in instruction.strip().split(".") if sentence.strip()]

                    # Split the document into two parts if it has more than 4 sentences
                    if len(sentences) > 4:
                        mid_point = len(sentences) // 2
                        document1 = Document(text=". ".join(sentences[:mid_point]) + ".")
                        document2 = Document(text=". ".join(sentences[mid_point:]) + ".")
                        character_stories.append(document1)
                        character_stories.append(document2)
                    else:
                        character_stories.append(Document(text=instruction))
            else:
                # Use the default Choi story
                sentences = [sentence.strip() for sentence in DR_CHOI_REWRITE.strip().split(".") if sentence.strip()]
                mid_point = len(sentences) // 2
                document1 = Document(text=". ".join(sentences[:mid_point]) + ".")
                document2 = Document(text=". ".join(sentences[mid_point:]) + ".")
                character_stories = [document1, document2]
        else:
            # Use the default Choi story
            sentences = [sentence.strip() for sentence in DR_CHOI_REWRITE.strip().split(".") if sentence.strip()]
            mid_point = len(sentences) // 2
            document1 = Document(text=". ".join(sentences[:mid_point]) + ".")
            document2 = Document(text=". ".join(sentences[mid_point:]) + ".")
            character_stories = [document1, document2]

        print(f"Initializing the graph store in host {SETTINGS.GRAPH_STORE_HOST}...")

        print("Initializing the embedder...")
        embedder = EmbedderCore(
            uri=SETTINGS.EMBEDDER_SERVING_URL, model_id=SETTINGS.EMBEDDER_MODEL_ID
        )
        print("Response OpenAI model:", SETTINGS.OPENAI_RESPONSE_MODEL_ID)
        graph_extract_llm = self.init_llm("graph_extract_llm")
        kg_extractor = DynamicLLMPathExtractor(
            llm=graph_extract_llm,
            parse_fn=parse_dynamic_triplets_with_props,
            extract_prompt=PromptTemplate(EXTRACT_GRAPH_TRIPLETS_PROMPT_TEMPLATE),
            allowed_entity_types=ENTITY_TYPES,
            allowed_relation_types=RELATION_PROPERTIES,
            allowed_entity_props=RELATION_TYPES,
            allowed_relation_props=ENTITY_PROPERTIES,
        )
        self.graph_store = FalkorDBGraphStore(
            url=f"falkor://{SETTINGS.GRAPH_STORE_USERNAME}:{SETTINGS.GRAPH_STORE_PASSWORD}@{SETTINGS.GRAPH_STORE_HOST}:{SETTINGS.GRAPH_STORE_PORT}",
            database=f"{self.user_id}_{self.assistant_id}_db",
            llm=graph_extract_llm,
            embedder=embedder,
            graph_extractor=kg_extractor,
            documents=character_stories,
            build=SETTINGS.GRAPH_STORE_BUILD,
            force_build=SETTINGS.GRAPH_STORE_FORCE_BUILD,
            show_progress=True,
        )

        print("Initializing the graph retriever...")
        cypher_generate_llm = self.init_llm("cypher_generate_llm")
        query_transform_llm = self.init_llm("query_transform_llm")
        sub_retriever = CustomSubRetriever(
            graph_store=self.graph_store.index.property_graph_store,
            include_text=True,
            embed_model=embedder,
            llm=cypher_generate_llm,
            similarity_top_k=10,
            path_depth=1,
        )
        self.graph_retriever = Retriever(
            sub_retriever=sub_retriever,
            graph_store=self.graph_store.index,
            llm=query_transform_llm,
            cache_chat_store=self.cache_chat_store,
        )

        print("Initializing the LLM router...")
        self.routing_llm = self.init_llm("routing_llm")

        if self.assistant_name == None or not use_default_story:
            if translated_instruction != "":
                print("Initializing the LLM assistant name extractor...")
                llm_assistant_name_extract = LLMCore(
                    uri=SETTINGS.LLM_SERVING_URL,
                    model_id=SETTINGS.LOCAL_LLM_MODEL_ID,
                    max_new_tokens=SETTINGS.MAX_NEW_TOKENS,
                    temperature=0.0,
                    is_chat=False,
                    use_openai=False,
                )

                # Extract the assistant name from the avatar instruction text
                result = llm_assistant_name_extract.complete(
                    prompt=ASSISTANT_NAME_EXTRACTION_PROMPT.format(instruction=translated_instruction)
                ).text
                self.assistant_name = json.loads(result)["name"]
                print("Extracted assistant name:", self.assistant_name)
            else:
                raise ValueError("No assistant name is given and the avatar instruction text is empty.")

        print("Initializing the generator...")
        self.response_llm = self.init_llm("response_llm")
        self.generator = Generator(
            llm=self.response_llm,
            chat_store=self.cache_chat_store,
            max_new_tokens=SETTINGS.MAX_NEW_TOKENS,
            streaming=True,
            response_mode=ResponseMode.COMPACT,
            assistant_name=self.assistant_name,
        )
        if warm_up:
            # Warm up the chatbot
            executor = ThreadPoolExecutor()
            future = executor.submit(
                self.run_async_task_in_thread, self.warm_up_chatbot
            )
            future.result() # Wait for the warm-up to finish

    def cleanup(self):
        """Clean up the chatbot server."""

        print("Disconnecting the OpenAI WebSocket...")
        self.generator.llm.websocket_client.ws.close()

    def init_llm(self, task):
        """Initialize the LLM model for a specific task."""

        config = self.llm_config.get(task)
        use_openai = config["provider"] == "openai"
        max_new_tokens = SETTINGS.LONG_MAX_NEW_TOKENS if config["max_new_tokens"] == "long" else SETTINGS.MAX_NEW_TOKENS
        temperature = config["temperature"]

        return LLMCore(
            uri=SETTINGS.LLM_SERVING_URL,
            model_id=SETTINGS.OPENAI_MODEL_ID if use_openai else SETTINGS.LOCAL_LLM_MODEL_ID,
            OPENAI_API_KEY=SETTINGS.OPENAI_API_KEY if use_openai else "EMPTY",
            use_openai=use_openai,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            is_chat=(task == "response_llm"),
            use_websocket=(task == "response_llm"),
        )
    
    def run_async_task_in_thread(self, task, *args):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(task(*args))
        loop.close()

    async def warm_up_chatbot(self):
        """Warm up the chatbot by generating a response to a predefined query."""

        print("\n=====================================\n")
        print("Warm up the chatbot...")

        total_processing_time = 2.0
        max_time = 3

        # lan = "korean"
        lan = "english"
        msg = "안녕하세요, 당신은 누구입니까?"
        self.final_query = "Hi, who are you?"
        print(f"Warmed up query: {msg}")

        while total_processing_time > 1.6 and max_time > 0:
            start = time.time()
            mixed_query = f"|{self.user_id}|{self.assistant_id}|{msg}"

            retry_times = 3
            retrieved_nodes = [NodeWithScore(node=TextNode(text=""), score=0)]
            while retry_times > 0:
                try:
                    # ================ Retrieve nodes from the graph store ================

                    # Decide whether to retrieve nodes based on the routing result
                    routing_result = int(self.routing_llm.complete(prompt=PromptTemplate(QUERY_ROUTING_PROMPT_TEMPLATE).format(text=self.final_query)).text)
                    print(f"\nRouting result: {routing_result}")
                    if routing_result == 0:
                        break # No need to retrieve nodes
    
                    # Define the prompt for extracting entities based on the list of entity names of each entity type
                    schema_info = self.graph_store.get_schema_info_str()
                    prompt_template = EXTRACT_ENTITIES_PROMPT_TEMPLATE.format(
                        text="{text}", schema=schema_info,
                        max_extracted_entities=20
                    )

                    retrieved_nodes = await self.graph_retriever.async_retrieve(
                        query=self.final_query,
                        prompt_template_str=prompt_template,
                        user_id=self.summarized_user_id,
                        assistant_id=self.summarized_assistant_id,
                    )
                    break
                except Exception as e:
                    print(f"Failed to retrieve nodes: {e}")
                    retry_times -= 1
                    if retry_times == 0:
                        print("Failed to retrieve nodes")

            # ================ Generate a response based on the retrieved nodes ================

            streamer = await self.generator.generate(
                query=mixed_query, nodes=retrieved_nodes,
                language=lan,
            )
            
            print("Warmed up response: ", end="", flush=True)
            async for chunk in streamer:
                print(chunk, end="", flush=True)

            total_processing_time = time.time() - start
            print(f"\nTotal processing time: {total_processing_time:.4f} seconds")
            max_time -= 1

        print("\nWarmed up the chatbot done!")
        print("\n=====================================\n")

    async def async_response_generator(self, streamer):
        """Generate a response asynchronously and stream it to the client."""

        check_time = False
        try:
            # Stream the response to the client
            async for chunk in streamer:
                if not check_time:
                    check_time = True
                    print(f"Time taken to generate first token: {time.time() - self.start:.4f} seconds")
                self.response_text.write(chunk)
                yield chunk

            print("\nResponse text:", self.response_text.getvalue())

        except Exception as e:
            print(f"Error in streaming response: {e}")
            traceback.print_exc()

    async def chat(self, message: str):
        """Chat with the chatbot and generate a response."""

        self.start = time.time()

        # Detect the language of the message and translate it into English
        self.original_query = message
        self.final_query = message
        detect_time = time.time()
        lan, _ = self.en_translator.detect_language(message)
        print(f"Detected language {lan} in {time.time() - detect_time:.4f} seconds")
        if lan != "english":
            translate_time = time.time()
            self.final_query = self.en_translator.translate(text=message, force_target=True)
            print(f"Time taken to translate the query: {time.time() - translate_time:.4f} seconds")

        mixed_query = f"|{self.user_id}|{self.assistant_id}|{message}"

        retry_times = 3
        retrieved_nodes = [NodeWithScore(node=TextNode(text=""), score=0)]
        while retry_times > 0:
            try:
                # ================ Retrieve nodes from the graph store ================
 
                retrieve_time = time.time()

                # Decide whether to retrieve nodes based on the routing result
                response = self.routing_llm.complete(prompt=PromptTemplate(QUERY_ROUTING_PROMPT_TEMPLATE).format(text=self.final_query)).text
                routing_result = int(json.loads(response)["retrieval_type"])
                print(f"Routing result: {routing_result}")
                if routing_result == 0:
                    break # No need to retrieve nodes
 
                # Define the prompt for extracting entities based on the list of entity names of each entity type
                schema_info = self.graph_store.get_schema_info_str()
                prompt_template = EXTRACT_ENTITIES_PROMPT_TEMPLATE.format(
                    text="{text}", schema=schema_info,
                    max_extracted_entities=20
                )

                retrieved_nodes = await self.graph_retriever.async_retrieve(
                    query=self.final_query,
                    prompt_template_str=prompt_template,
                    user_id=self.summarized_user_id,
                    assistant_id=self.summarized_assistant_id,
                )
                print(f"Time taken to retrieve nodes: {time.time() - retrieve_time:.4f} seconds")

                break
            except Exception as e:
                print(f"Failed to retrieve nodes: {e}")
                retry_times -= 1
                if retry_times == 0:
                    print("Failed to retrieve nodes")

        print(f"Time taken before generating stream: {time.time() - self.start:.4f} seconds")

        # ================= Print the retrieved nodes =================

        print(f"Retrieved nodes:")
        for node in retrieved_nodes:
            print(f"Node text: {node.node.text}")
            print("-------------------")

        # ================ Generate a response based on the retrieved nodes ================
        
        streamer = await self.generator.generate(
            query=mixed_query, nodes=retrieved_nodes,
            language=lan,
        )
        self.response_text = StringIO()

        return self.async_response_generator(streamer)

    def extract_messsage(self, user_id):
        """Extract a specified message pair from the cache chat store periodically."""

        max_messages = self.cache_chat_store.max_messages_pairs
        chat_store_len = int(len(self.cache_chat_store.get_chat_history(user_id)) / 2)
        if not chat_store_len < int(max_messages / 2):
            starting_index = 1 - int(max_messages / 2)
            extracted_chat_index = starting_index + chat_store_len
            extracted_chat = self.cache_chat_store.get_chat_history(user_id)[
                int(extracted_chat_index - 0.5) * 2 : extracted_chat_index * 2
            ]
            return extracted_chat
        return None
    
    def post_processing(self):
        """Post-process the chatbot response to store the chat history and update the graph store."""

        try:
            print("Original response text:", self.response_text.getvalue())

            # ================ Store the chat history ================

            translated_cache_id = f"translated_{self.user_id}_{self.assistant_id}"
            original_cache_id = f"original_{self.user_id}_{self.assistant_id}"

            # Store the recent query and response in the cache chat store and persistent chat store.
            # The original query and response are stored in the cache chat store for later insertion with the prompt to response LLM.
            # The translated ones are used during retrieval process.
            original_removed_messages = self.cache_chat_store.add_message_pair(
                original_cache_id, self.original_query, self.response_text.getvalue()
            )
            translated_removed_messages = self.cache_chat_store.add_message_pair(
                translated_cache_id, self.final_query, self.en_translator.translate(self.response_text.getvalue(), force_target=True)
            )
            self.persistent_chat_store.save_chat(
                translated_cache_id, self.final_query, self.en_translator.translate(self.response_text.getvalue(), force_target=True)
            )
            print(f"Translated removed messages: {translated_removed_messages}")
            print(f"Original removed messages: {original_removed_messages}")

            # ================ Extract the chat memory ================

            # Extract the a specified message pair from the cache
            extracted_chat = self.extract_messsage(translated_cache_id)
            print(f"Extracted chat: {extracted_chat}")

            # ================ Update the graph store =================

            # Summarize the extracted chat and insert it into the graph store
            if extracted_chat:

                # Summarize the extracted chat
                memory_message = self.cache_chat_store.transform_message_pair(
                    extracted_chat, self.summarized_user_id, self.summarized_assistant_id, self.user_id # TODO: optimize to summarize all the important details (name, time, etc.)
                )
                memory_document = Document(text=memory_message)

                # Insert the summarized chat memory into the graph store
                retry_times = 3
                while retry_times > 0:
                    try:
                        inset_time = time.time()
                        self.graph_store.insert_document(document=memory_document)
                        end_insert = time.time()
                        print(
                            f"Time taken to insert document into the graph store: {end_insert - inset_time:.4f} seconds"
                        )
                        break
                    except Exception as e:
                        print(f"Failed to insert document into the graph store: {e}")
                        retry_times -= 1
                        if retry_times == 0:
                            print("Failed to insert document into the graph store")
        except Exception as e:
            print(f"Error in post-processing: {e}")
            import traceback
            traceback.print_exc()
    