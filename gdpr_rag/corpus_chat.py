import logging
import pandas as pd
from openai import OpenAI
import re
import fnmatch
import regex # fuzzy lookup of references in a section of text
import copy
from enum import Enum



# import src.data
# importlib.reload(src.data)
from regulations_rag.regulation_index import RegulationIndex, EmbeddingParameters

from regulations_rag.string_tools import match_strings_to_reference_list

from regulations_rag.regulation_reader import RegulationReader
                           
from regulations_rag.embeddings import get_ada_embedding, \
                           get_closest_nodes, \
                           num_tokens_from_string,  \
                           num_tokens_from_messages

from regulations_rag.rerank import RerankAlgos, rerank

from regulations_rag.corpus import Corpus

logger = logging.getLogger(__name__)

# Create custom log levels for the really detailed logs
DEV_LEVEL = 15
ANALYSIS_LEVEL = 25
logging.addLevelName(DEV_LEVEL, 'DEV')       
logging.addLevelName(ANALYSIS_LEVEL, 'ANALYSIS')       
###############################################
## DO this in the main file, before importing chat_bot.py
# # Set up basic configuration first
# log_file = '.....'
# logging_level = logging.INFO
# if log_file == '':
#     logging.basicConfig(level=logging_level)
# else: 
#     logging.basicConfig(filename=log_file, filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging_level)

# # Then import chat_bot.py so it inherits these settings
###############################################



class CorpusChat():
    class State(Enum):
        RAG = "rag"
        NO_DATA = "no_relevant_embeddings"
        NEEDS_DATA = "requires_additional_sections"
        STUCK = "stuck"

    class Prefix(Enum):
        ANSWER = "ANSWER:"
        SECTION = "SECTION:"
        NONE = "NONE:"

    class Errors(Enum):
        NO_DATA = "I was unable to find any relevant documentation to assist in answering the question. Can you try rephrasing the question?"
        NO_RELEVANT_DATA = "The documentation I have been provided does not help me answer the question. Please rephrase it and let's try again?"
        STUCK = "Unfortunately the system is in an unrecoverable state. Please clear the chat history and retry your query"
        UNKNOWN_STATE = "The system is in an unknown state and cannot proceed. Please clear the chat history and retry your query"
        NOT_FOLLOWING_INSTRUCTIONS = "The call to the LLM resulted in a response that did not fit parameters, even after retrying it. Please clear the chat history and retry your query."

    def __init__(self, 
                 openai_client, 
                 embedding_parameters,
                 chat_parameters,
                 corpus_index,
                 rerank_algo = RerankAlgos.NONE,   
                 user_name_for_logging = 'test_user'): 

        self.user_name = user_name_for_logging
        self.openai_client = openai_client
        self.embedding_parameters = embedding_parameters
        self.chat_parameters = chat_parameters

        self.index = corpus_index
        self.corpus = self.index.corpus

        self.rerank_algo = rerank_algo
        self.reset_conversation_history()

    def reset_conversation_history(self):
        logger.log(ANALYSIS_LEVEL, f"{self.user_name}: Reset Conversation History")        
        self.messages = []
        self.messages_without_rag = []
        self.references = {}
        self.system_state = CorpusChat.State.RAG


    def _extract_question_from_rag_data(self, decorated_question):
        return decorated_question.split("\n")[0][len("Question: "):]

    def append_content(self, role, content):
        if role not in ["user", "assistant", "system"]:
            logger.error(f"Tried to add a message for the role {role} which is not a valid role")
            return

        content_without_rag = content

        if role == "user":            
            if content.startswith("Question:"):
                content_without_rag = self._extract_question_from_rag_data(content)

        # nothing special for assistant or system messages

        if self.messages and (self.messages[-1]["role"] == role and self.messages[-1]["content"] == content): # don't duplicate messages in the list
            return

        self.messages.append({"role": role, "content": content})
        # logger.log(ANALYSIS_LEVEL, f"{role} to {self.user_name}: {content}")        
        self.messages_without_rag.append({"role": role, "content": content_without_rag})        

    def append_references(self, reformatted_response, df_references):
        self.references[reformatted_response.strip()] = df_references

    def _create_system_message(self, number_of_options = 3, review = False):
        """
        Generates a system message instructing the system on how to answer questions using specific guidelines.

        The message includes the pattern for referencing sections from the South African Exchange Control Manual and outlines three 
        options for responding to questions: answering directly, requesting additional documentation, or stating none if no other options apply.
        The optional parameter number_of_options can be set to 2 if you do not want the system to ask for more information. This can be  
        used when the system asks for more information but the information is already in the RAG prompt (which happens surprisingly often in some models)

        Parameters:
        - number_of_options (in): Can only be 3 (answer, ask for more info, none) or 2 (answer, none)


        Returns:
            str: A formatted message detailing how to respond to questions based on the manual's guidelines.
        """
        if not review:
            sys_instruction = f"You are answering questions about {self.index.corpus_description} for {self.index.user_type} based only on the reference extracts provided. You have {number_of_options} options:\n"
        else:
            sys_instruction = f"Please review your answer. You were asked to assist the user by responding to their question in 1 of {number_of_options} ways but your response does not follow the expected format. Please reformat your response so that it follows the requested format.\n" 

        sys_option_ans  = f"Answer the question. Preface an answer with the tag '{CorpusChat.Prefix.ANSWER.value}'. All referenced extracts must be quoted at the end of the answer, not in the body, by number, in a comma separated list starting after the keyword 'Reference: '. Do not include the word Extract, only provide the number(s).\n"
        sys_option_sec  = f"Request additional documentation. If, in the body of the extract(s) provided, there is a reference to another section that is directly relevant and not already provided, respond with the word '{CorpusChat.Prefix.SECTION.value}' followed by 'Extract extract_number, Reference section_reference' - for example SECTION: Extract 1, Reference Article 98.\n"
        sys_option_none = f"State '{CorpusChat.Prefix.NONE.value}' and nothing else in all other cases\n"

        if number_of_options == 2:
            sys_instruction = f"{sys_instruction}1) {sys_option_ans}2) {sys_option_none}"
        elif number_of_options == 3:
            sys_instruction = f"{sys_instruction}1) {sys_option_ans}2) {sys_option_sec}3) {sys_option_none}"
        else:
            logger.log(DEV_LEVEL, f"Forcing the number of options in the system message to be 3. They were {number_of_options}")
            number_of_options == 3
            sys_instruction = f"{sys_instruction}1) {sys_option_ans}2) {sys_option_sec}3) {sys_option_none}"

        return sys_instruction


    def _check_response(self, response, df_definitions, df_sections):
        # Return dictionaries:
        #{"success": False, "path": "SECTION:"/"ANSWER:", "llm_followup_instruction": llm_instruction} 
        #{"success": True, "path": "SECTION:", "document": 'GDPR', "section": section_reference}
        #{"success": True, "path": "ANSWER:"", "answer": llm_text, "reference": references_as_integers}
        #{"success": True, "path": "NONE:"}


        if response.startswith(CorpusChat.Prefix.ANSWER.value):
            prefix = CorpusChat.Prefix.ANSWER.value

            answer = response[len(prefix):].strip()

            # Extract and clean references from the raw response
            references = answer.split("Reference:")[-1].split(",") if "Reference:" in answer else []
            cleaned_references = [ref.strip() for ref in references if ref.strip()]
            llm_text = answer[:answer.rfind("Reference:")].strip() if "Reference:" in answer else answer

            # Did not supply any references which is in line with the Instruction
            if not cleaned_references:                
                return {"success": True, "path": prefix, "answer": llm_text, "reference": []}

            # Loop through each item in the list
            references_as_integers = []
            for item in cleaned_references:
                try:
                    # Attempt to convert the item to an integer
                    integer_value = int(item)
                    if integer_value < 1 or integer_value > len(df_definitions) + len(df_sections):
                        llm_instruction = "When answering the question, you have made reference to an extract number that was not provided. Please re-write your references and only refer to the extracts provided by their number"
                        return {"success": False, "path": prefix, "llm_followup_instruction": llm_instruction}
                    references_as_integers.append(integer_value)
                except ValueError:
                    match = re.search(r'\d', item)
                    if not match:
                        llm_instruction = "When answering the question, you have made reference to an extract but I am unable to extract the number from your reference. Please re-write your answer using integer extract number(s)"
                        return {"success": False, "path": prefix, "llm_followup_instruction": llm_instruction}
                    else:
                        integer_value = int(match.group(0))
                        if integer_value < 1 or integer_value > len(df_definitions) + len(df_sections):
                            llm_instruction = "When answering the question, you have made reference to an extract number that was not provided. Please re-write your references and only refer to the extracts provided by their number"
                            return {"success": False, "path": prefix, "llm_followup_instruction": llm_instruction}
                        references_as_integers.append(integer_value)
            
            return {"success": True, "path": prefix, "answer": llm_text, "reference": references_as_integers}

        elif response.startswith(CorpusChat.Prefix.SECTION.value):
            prefix = CorpusChat.Prefix.SECTION.value
            request = response[len(prefix):].strip()

            document_name = ""
            section_reference = ""
            pattern = r"Extract (\d+), Reference (.+)"
            match = re.match(pattern, request)
            if match:
                extract_number = int(match.group(1))
                if extract_number < 1 or extract_number > len(df_definitions) + len(df_sections):
                        llm_instruction = "When requesting an additional section, you have made reference to an extract number that was not provided. Please re-write your answer and use a valid extract number"
                        return {"success": False, "path": prefix, "llm_followup_instruction": llm_instruction} 
                if extract_number < len(df_definitions):
                    document_name = df_definitions.iloc[extract_number-1]["document"]
                else:
                    document_name = df_sections.iloc[extract_number-len(df_definitions)-1]["document"]
                doc = self.corpus.get_document(document_name)
                section_reference = match.group(2)
                # many cases will get a GDPR reference and that may start with the word 'Article'
                if section_reference.lower().startswith("article "):
                    section_reference = section_reference[8:]

                if doc.reference_checker.is_valid(section_reference):
                    return {"success": True, "path": prefix, "extract": extract_number, "document": document_name, "section": section_reference}
                elif document_name != 'GDPR': # most articles will refer to GDPR so it makes sense to check if it is a GDPR reference as well
                    gdpr_doc = self.corpus.get_document('GDPR')
                    if gdpr_doc.reference_checker.is_valid(section_reference):
                        return {"success": True, "path": prefix, "extract": extract_number, "document": 'GDPR', "section": section_reference}
                
                document_index = doc.reference_checker.text_version
                if document_name != 'GDPR':
                    gdpr_doc = self.corpus.get_document('GDPR')
                    if document_index == "":
                        document_index = gdpr_doc.reference_checker.text_version
                    else:
                        document_index += ", or " + self.corpus.get_document('GDPR').reference_checker.text_version
                llm_instruction = f'The reference {section_reference} does not appear to be a valid reference for the document. Try using the format {document_index}'
                return {"success": False, "path": prefix, "llm_followup_instruction": llm_instruction} 
            else:
                llm_instruction = 'When requesting an additional section, you did not use the format r"Extract (\d+), Reference (.+)" or you included additional text. Please re-write your response using this format'
                return {"success": False, "path": prefix, "llm_followup_instruction": llm_instruction} 

            return {"success": True, "path": prefix, "extract": extract_number, "document": document_name, "section": section_reference}

        elif response.startswith(CorpusChat.Prefix.NONE.value):
            return {"success": True, "path": CorpusChat.Prefix.NONE.value}

        llm_instruction = f"Your response, did not begin with one of the keywords, '{CorpusChat.Prefix.ANSWER.value}', '{CorpusChat.Prefix.SECTION.value}' or '{CorpusChat.Prefix.NONE.value}'. Please review the question and provide an answer in the required format. Also make sure the referenced extracts are quoted at the end of the answer, not in the body, by number, in a comma separated list starting after the keyword 'Reference: '. Do not include the word Extract, only provide the number(s).\n"
        return {"success": False, "path": "NONE", "llm_followup_instruction": llm_instruction}
        


    def _add_rag_data_to_question(self, question, df_definitions, df_search_sections):
        """
        Appends relevant definitions and sections from the Manual to the question to provide context.

        This method formats the question with additional information from the Manual, including definitions and referenced sections, to aid in answering the question more accurately.

        Parameters:
        - question (str): The original question to be answered.
        - df_definitions (DataFrame): A DataFrame containing definitions from the Manual.
        - df_search_sections (DataFrame): A DataFrame containing sections from the Manual relevant to the question.

        Returns:
        - str: The question appended with definitions and sections from the Manual.
        """

        user_content = f'Question: {question}\n\n'
        counter = 1
        if not df_definitions.empty:
            for definition in df_definitions['definition'].to_list():
                # Append the formatted string with the definition and a newline character
                user_content += f"Extract {counter}:\n{definition}\n"
                counter += 1

        if not df_search_sections.empty:
            for section in df_search_sections['regulation_text'].to_list():
                # Append the formatted string with the definition and a newline character
                user_content += f"Extract {counter}:\n{section}\n"
                counter += 1

        return user_content



    def similarity_search(self, user_content):
        """
        Finds the index values, definitions and workflows that are most similar to the provided user content. A workflow is 
        triggered if the lowest cosine similarity score for the closest workflow is lower that the lowest cosine similarity
        score for the closest definition and the closest index

        Parameters:
        - user_content (str): The content provided by the user to conduct the similarity search against.

        Returns:
        - tuple: Contains the most relevant workflow triggered (if any), a DataFrame of relevant definitions, and
                a DataFrame of the top relevant sections sorted by their relevance.
        """
        question_embedding = get_ada_embedding(self.openai_client, user_content, self.embedding_parameters.model, self.embedding_parameters.dimensions)        
        logger.log(DEV_LEVEL, "#################   Similarity Search       #################")
        if len(self.index.workflow) > 0:
            relevant_workflows = self.index.get_relevant_workflow(user_content = user_content, user_content_embedding = question_embedding, threshold = self.embedding_parameters.threshold)
            if not relevant_workflows.empty > 0:
                # relevant_workflows is sorted in the get_closest_nodes method
                most_relevant_workflow_score = relevant_workflows.iloc[0]['cosine_distance']
                workflow_triggered = relevant_workflows.iloc[0]['workflow']
                logger.info(f"Found a potentially relevant workflow: {workflow_triggered}")
            else:
                logger.log(DEV_LEVEL, "No relevant workflow found")
                most_relevant_workflow_score = 1.0
                workflow_triggered = "none"
        else:
            most_relevant_workflow_score = 1.0
            workflow_triggered = "none"

        relevant_definitions = self.index.get_relevant_definitions(user_content = user_content, user_content_embedding = question_embedding, threshold = self.embedding_parameters.threshold)
        if not relevant_definitions.empty:

            most_relevant_definition_score = relevant_definitions.iloc[0]['cosine_distance']

            if most_relevant_definition_score < most_relevant_workflow_score: # there is something more relevant than a workflow
                logger.log(DEV_LEVEL, f"Found a definition that was more relevant than the workflow: {workflow_triggered}")
                workflow_triggered = "none"        


        relevant_sections = self.index.get_relevant_sections(user_content = user_content, 
                                                             user_content_embedding = question_embedding, 
                                                             threshold = self.embedding_parameters.threshold, 
                                                             rerank_algo = self.rerank_algo)
        if not relevant_sections.empty:    
        
            if not relevant_sections.empty:    
                most_relevant_section_score = relevant_sections.iloc[0]['cosine_distance']

                if most_relevant_section_score < most_relevant_workflow_score and workflow_triggered != "none": # there is something more relevant than a workflow
                    logger.log(DEV_LEVEL, f"Found a section that was more relevant than the workflow: {workflow_triggered}")
                    workflow_triggered = "none"

        return workflow_triggered, relevant_definitions, relevant_sections

    def _get_api_response(self, messages, testing=False, manual_responses_for_testing=[], response_index=0):
        """
        Fetches a response from the OpenAI API or uses a canned response based on the testing flag.

        Parameters:
        - messages (list): The list of messages to send as context to the OpenAI API.
        - testing (bool): Flag to determine whether to use the OpenAI API or canned responses.
        - manual_responses_for_testing (list of str): A list of canned responses to use if testing is True.
        - response_index (int): The index of the response to use from manual_responses_for_testing.

        Returns:
        - str: The response from the OpenAI API or the selected canned response.

        NOTE: In some tests I want a manual response to the first call so that I can test the rest of the code. To do that
              I set testing = True but only pass one message in manual_responses_for_testing 
        """
        if testing and response_index < len(manual_responses_for_testing):
            # NOTE: In some tests I force the first response but want to test the API call on the second attempt 
            #       so I set testing = True but only pass one message in manual_responses_for_testing 
            response_text = manual_responses_for_testing[response_index]
        else:
            model_to_use = self.chat_parameters.model
            total_tokens = num_tokens_from_messages(messages, model_to_use)
            
            if total_tokens > 15000:
                return "The is too much information in the prompt so we are unable to answer this question. Please try again or word the question differently"

            # Adjust model based on token count, similar to your original logic
            if (model_to_use in ["gpt-3.5-turbo", "gpt-4"]) and total_tokens > 3500:
                logger.warning("Switching to the gpt-3.5-turbo-16k model due to long prompt.")                
                model_to_use = "gpt-3.5-turbo-16k"
            
            response = self.openai_client.chat.completions.create(
                            model=model_to_use,
                            temperature=self.chat_parameters.temperature,
                            max_tokens=self.chat_parameters.max_tokens,
                            messages=messages
                        )
            response_text = response.choices[0].message.content
        return response_text

    def _truncate_message_list(self, system_message, message_list, token_limit=2000):
        """
        Truncates the message list to fit within a specified token limit, ensuring the inclusion of the system message 
        and the most recent messages from the message list. The function guarantees that the returned list always contains 
        the system message and at least the last message from the message list, even if their combined token count exceeds 
        the token limit.

        Parameters:
        - system_message (list): A list containing a single dictionary with the system message.
        - message_list (list): A list of dictionaries representing the user and assistant messages.
        - token_limit (int, optional): The maximum allowed number of tokens. Defaults to 2000.

        Returns:
        - list: A list of messages truncated to meet the token limit, including the system message and the last messages.
        """
        if not message_list:
            return system_message

        # Initialize the token count with the system message and the last message in the list
        token_count = sum(num_tokens_from_string(msg["content"]) for msg in system_message + [message_list[-1]])
        number_of_messages = 1

        # Add messages from the end of the list until the token limit is reached or all messages are included
        while number_of_messages < len(message_list) and token_count < token_limit:
            next_message = message_list[-(number_of_messages + 1)]
            next_message_token_count = num_tokens_from_string(next_message["content"])

            # Check if adding the next message would exceed the token limit
            if token_count + next_message_token_count > token_limit:
                break
            # else keep going
            token_count += next_message_token_count
            number_of_messages += 1

        number_of_messages_excluding_system = max(1, number_of_messages - 1)
        # Compile the truncated list of messages, always including the system message and the most recent messages
        truncated_messages = [system_message[0]] + message_list[-number_of_messages_excluding_system:]

        return truncated_messages


    def resource_augmented_query(self, user_question, df_definitions, df_search_sections, number_of_options = 3,
                                 testing = False, manual_responses_for_testing = []):
        # This function's job is to make sure the response follows the rules or it fails. To this end, it will try a second call to the LLM 
        # if the fist call does not follow the rules. The second call will include a description of what was wrong with the fist answer. If,
        # after the second call, the LLM still does not return a response that follows the rules, this will return a False value for success.

        # NOTE: This function does not alter the user question nor the input RAG data so the RAG version of the user question is 
        #        self._add_rag_data_to_question(user_question, df_definitions, df_search_sections)

        # Returns
        # Unsuccessful path
        #   {"success": False, "path": ANSWER" / SECTION: / NONE:, "assistant_response": content} 
        # i.e replace the "llm_followup_instruction" key from the the dictionary created in self._check_response() with the key "assistant_message"

        # successful returns are the same as the dictionary created in self._check_response() 
        #   {"success": True, "path": "SECTION:", "extract", extract_num_as_int "document": document_name, "section": section_reference} NB the document may not be the same as the document in extract_num_as_int
        #   {"success": True, "path": "ANSWER:"", "answer": llm_text, "reference": references_as_integers}
        #   {"success": True, "path": "NONE:"}

        # TODO: Ensure that the user_provides_input method takes responsibility for setting self.system_state = CorpusChat.State.STUCK
        #       if the result is unsuccessful

# Here are the dictionary items returned from self._check_response()
#{"success": False, "path": "SECTION:"/"ANSWER:", "llm_followup_instruction": llm_instruction} 
#{"success": True, "path": "SECTION:", "document": 'GDPR', "section": section_reference}
#{"success": True, "path": "ANSWER:"", "answer": llm_text, "reference": references_as_integers}
#{"success": True, "path": "NONE:"}

        if self.system_state != CorpusChat.State.RAG:
            logger.error("resource_augmented_query method called but the the system is not in rag state")
            return {"success": False, "path": "NONE", "assistant_response": CorpusChat.State.STUCK.value} 
        

        if len(self.messages) > 1 or len(df_definitions) + len(df_search_sections) > 0: # should always be the case as we check this in the control loop
            logger.log(DEV_LEVEL, "#################   RAG Prompts   #################")

            system_content = self._create_system_message(number_of_options, review=False)
            logger.log(DEV_LEVEL, "System Prompt:\n" + system_content)

            # Replace the user question with the RAG version of it
            user_question = self._add_rag_data_to_question(user_question, df_definitions, df_search_sections)
            logger.log(DEV_LEVEL, "User Prompt with RAG:\n" + user_question) # this will be output with ANALYSIS_LEVEL

            chat_messages = copy.deepcopy(self.messages)
            chat_messages.append({"role": "user", "content": user_question})

            # Create a temporary message list. We will only add the messages to the chat history if we get well formatted answers
            system_message = [{"role": "system", "content": system_content}]
            truncated_chat = self._truncate_message_list(system_message, chat_messages, 2000)

            response = self._get_api_response(messages = truncated_chat, testing=testing, manual_responses_for_testing=manual_responses_for_testing, response_index = 0)
            check_result = self._check_response(response, df_definitions=df_definitions, df_sections=df_search_sections)
            if check_result["success"]:
                check_result["question"] = user_question
                return check_result

            # The model did not perform as instructed so we not ask it to check its work
            logger.info(f"Initial chat API response did not follow instructions. New instruction: {check_result['llm_followup_instruction']}")

            # despondent_user_content = f"Please check your answer and make sure you preface your response using only one of the three permissible words, {CorpusChat.Prefix.ANSWER.value}, {CorpusChat.Prefix.SECTION.value} or {CorpusChat.Prefix.NONE.value}"
            # system_content = self._create_system_message(number_of_options, review=True)

            despondent_user_messages = truncated_chat + [
                                        {"role": "assistant", "content": response},
                                        {"role": "user", "content": check_result["llm_followup_instruction"]}]
                                        
            response = self._get_api_response(messages = despondent_user_messages, testing=testing, manual_responses_for_testing=manual_responses_for_testing, response_index = 1)
            check_result = self._check_response(response, df_definitions=df_definitions, df_sections=df_search_sections)
            if check_result["success"]:
                return check_result
            else: 
                msg = f"Even after trying a second time, the LLM was not following instructions. New instruction: {check_result['llm_followup_instruction']}"
                logger.error(msg)
                return {"success": False, "path": check_result["path"], "assistant_response": CorpusChat.Errors.NOT_FOLLOWING_INSTRUCTIONS.value} 


        msg = "A call to resource_augmented_query was made with insufficient information"
        logger.error(msg)
        return {"success": False, "path": "NONE", "assistant_response": CorpusChat.Errors.NOT_FOLLOWING_INSTRUCTIONS.value} 
        



#     def chat_completion(self, user_content, testing = False, manual_responses_for_testing = []):
#         self.user_provides_input(user_content, testing, manual_responses_for_testing)                        
#         return self.messages[-1]["content"]

    # Note: To test the workflow I need some way to control the openai API responses. I have chosen to do this with the two parameters
    #       testing: a flag. If false the function will run calling the openai api for responses. If false the function will 
    #                        select the response from the list of responses manual_responses_for_testing
    #       manual_responses_for_testing: A list of text. If testing == True, these values will be used as if they were the 
    #                                     the response from the API. This function can make multiple calls to the API so the i-th
    #                                     row in the list corresponds to the i-th call of the API
    #  
    def user_provides_input(self, user_content, testing = False, manual_responses_for_testing = []):
        
        if user_content is None:
            logger.error(f"{self.user_name}: user_provides_input() function received an empty input. This should not have happened and is an indication there is a bug in the frontend. The system will be placed into a 'stuck' status")
            self.append_content("assistant", CorpusChat.Errors.UNKNOWN_STATE.value)
            self.system_state = CorpusChat.State.STUCK
            return

        logger.log(ANALYSIS_LEVEL, f"{self.user_name} question: {user_content}")        
        workflow_triggered, df_definitions, df_search_sections = self.similarity_search(user_content) # df_search_sections MUST not have "document"
        if workflow_triggered == "documentation":
            raise NotImplementedError()
            # user_content = self.enrich_user_request_for_documentation(user_content, self.messages_without_rag)
            # workflow_triggered, df_definitions, df_search_sections = self.similarity_search(user_content)


        if self.system_state == CorpusChat.State.STUCK:
            self.append_content("user", user_content)
            self.append_content("assistant", CorpusChat.Errors.STUCK.value)
            return 

        elif self.system_state == CorpusChat.State.RAG:            
            if len(self.messages) < 2 and (len(df_definitions) + len(df_search_sections) == 0):
                logger.log(DEV_LEVEL, "Note: Unable to find any definitions or text related to this query")
                self.system_state = CorpusChat.State.RAG         
                self.append_content("user", user_content)       
                self.append_content("assistant", CorpusChat.Errors.NO_DATA.value)
                return
            else:
                result = self.resource_augmented_query(user_question = user_content, df_definitions = df_definitions, df_search_sections = df_search_sections, number_of_options=3,
                                                       testing = testing, manual_responses_for_testing = manual_responses_for_testing)
                if not result["success"]:
                    logger.error("Note: RAG returned an unexpected response")
                    self.append_content("user", self._add_rag_data_to_question(user_content, df_definitions, df_search_sections))
                    self.append_content("assistant", CorpusChat.Errors.NOT_FOLLOWING_INSTRUCTIONS.value)
                    self.system_state = CorpusChat.State.STUCK # We are at a dead end.
                    return


                if result["path"] == CorpusChat.Prefix.ANSWER.value:
                    #   result = {"success": True, "path": "ANSWER:"", "answer": llm_text, "reference": references_as_integers}
                    self.append_content("user", self._add_rag_data_to_question(user_content, df_definitions, df_search_sections))
                    reformatted_response, df_definitions, df_search_sections = self.reformat_assistant_answer(result, df_definitions = df_definitions, df_search_sections = df_search_sections)
                    # collect references
                    df_references = self.collect_references(df_definitions, df_search_sections)
                    self.append_content("assistant", reformatted_response)
                    self.append_references(reformatted_response, df_references)
                    self.system_state = CorpusChat.State.RAG 
                    return 
                elif result["path"] == CorpusChat.Prefix.SECTION.value:
                    #   result = {"success": True, "path": "SECTION:", "extract", extract_num_as_int "document": document_name, "section": section_reference} NB the document may not be the same as the document in extract_num_as_int
                    logger.log(DEV_LEVEL, f"System requested for more info: Extract {result['extract']} requested section {result['section']}")
                    # Asking for an invalid section or a section that is already in the RAG
                    force = False
                    document = self.corpus.get_document(result['document'])
                    if document.reference_checker.is_reference_or_parents_in_list(result['section'], df_search_sections["section_reference"].tolist()):
                        logger.log(DEV_LEVEL, f"But {result['section']} is already in the RAG data, so now forcing the system to answer or opt out")
                        force = True

                    if force:
                        result = self.resource_augmented_query(user_question = user_content, df_definitions = df_definitions, df_search_sections = df_search_sections, number_of_options=2,
                                                               testing = testing, manual_responses_for_testing = manual_responses_for_testing[1:])

                    else:
                        df_search_sections = self.add_section_to_resource(result, df_definitions, df_search_sections)
                        if self.system_state == CorpusChat.State.STUCK: # failed to add the sections
                            # TODO: Do you want to ask the user for help?
                            logger.info("Note: Request to add resources failed")
                            self.append_content("assistant", CorpusChat.Errors.STUCK.value)
                            return

                        # ... and try again with new resources
                        result = self.resource_augmented_query(user_question = user_content, df_definitions = df_definitions, df_search_sections = df_search_sections, number_of_options=3,
                                                               testing = testing, manual_responses_for_testing = manual_responses_for_testing[1:])
                    
                    if result["path"] == CorpusChat.Prefix.ANSWER.value:
                        logger.info("Note: Question answered with the additional information")
                        self.append_content("user", self._add_rag_data_to_question(user_content, df_definitions, df_search_sections))
                        reformatted_response, df_definitions, df_search_sections = self.reformat_assistant_answer(result, df_definitions = df_definitions, df_search_sections = df_search_sections)
                        # collect references
                        df_references = self.collect_references(df_definitions, df_search_sections)                        
                        self.append_content("assistant", reformatted_response)
                        self.append_references(reformatted_response, df_references)
                        self.system_state = CorpusChat.State.RAG 
                        return
                    
                    else: 
                        logger.info("Note: Even with the additional information, they system was unable to answer the question. Placing the system in 'stuck' mode")
                        logger.info(f"The response from the query with additional resources was: \n{response}")
                        msg = "A call for additional sections did not result in sufficient information to answer the question. The system is now stuck. Please clear the chat history and retry your query"
                        self.append_content("assistant", msg)
                        self.system_state = CorpusChat.State.STUCK
                        return

                elif result["path"] == CorpusChat.Prefix.NONE.value:
                    #   result {"success": True, "path": "NONE:"}
                    logger.info("Note: The LLM was not able to find anything relevant in the supplied sections")
                    self.append_content("user", self._add_rag_data_to_question(user_content, df_definitions, df_search_sections))
                    self.append_content("assistant", CorpusChat.Errors.NO_RELEVANT_DATA.value)
                    self.system_state = CorpusChat.State.RAG
                    return

                else:
                    logger.error("Note: RAG returned an unexpected response")
                    self.append_content("assistant", CorpusChat.Errors.NOT_FOLLOWING_INSTRUCTIONS.value)
                    self.system_state = CorpusChat.State.STUCK # We are at a dead end.
                    return
        else:
            logger.error("The system is in an unknown state")
            self.append_content("assistant", CorpusChat.Errors.UNKNOWN_STATE.value)
            return

    def collect_references(self, df_definitions, df_search_sections):
        references = []
        for index, row in df_definitions.iterrows():
            references.append([row['document'], row['section_reference'], row['definition']])
        for index, row in df_search_sections.iterrows():
            references.append([row['document'], row['section_reference'], row['regulation_text']])

        df_references = pd.DataFrame(references, columns = ["document", "section_reference", "text"])
        return df_references

    def add_section_to_resource(self, result, df_definitions, df_search_sections):
        '''
        Returns a DataFrame that should be be used in RAG that is a replacement for df_search_sections.

        If there is an issue, this will return df_search_sections otherwise it will create a DataFrame with the the entry from 
        result["document"], result["section"] and df_search_sections.iloc[result["extract"] - len(df_definitions) - 1]

        '''
        # result = {"success": True, "path": "SECTION:", "extract": extract_num_as_int, "document": document_name, "section": section_reference} NB the document may not be the same as the document in extract_num_as_int
        # Step 1) confirm it is requesting something that passes validation
        doc = self.corpus.get_document(result["document"])
        modified_section_to_add = doc.reference_checker.extract_valid_reference(result["section"])
        
        if modified_section_to_add is None:
            logger.info(f"Tried to add {section_to_add} the Valid_Index object could not extract a valid reference from this")
            self.system_state = CorpusChat.State.STUCK 
            return df_search_sections
        
        text_to_add = ""
        try: # passes index verification but there is an error retrieving the section
            text_to_add = doc.get_text(modified_section_to_add)

        except Exception as e:
            logger.log(DEV_LEVEL, f"Tried to add {modified_section_to_add} but a call to get this regulation resulted in an exception {e}")
            self.system_state = CorpusChat.State.STUCK 
            return df_search_sections

        if text_to_add == "":
            logger.log(DEV_LEVEL, f"The section {modified_section_to_add} does not have any text. Is it a valid reference")
            return df_search_sections
          
        # prepare the new data
        new_row = [result["document"], result["section"], text_to_add]
        new_sections = pd.DataFrame([new_row], columns = ["document", "section_reference", "regulation_text"])


        if result["extract"] > len(df_definitions): # Delete the other sections, keep the referring section and the new data
            row_to_keep = result["extract"] - len(df_definitions) - 1
            sections_to_keep = df_search_sections.iloc[[row_to_keep]]
            new_sections = pd.concat([sections_to_keep, new_sections], ignore_index = True)

        return new_sections




    def reformat_assistant_answer(self, result, df_definitions, df_search_sections):
        """
        Reformats the "Reference" section from the LLM's response to ensure consistency with the correctly formatted
        sections from the data used to populate the RAG content. It identifies each reference in the LLM's response,
        replaces it with the closest match from a list of provided references, and removes duplicates.

        Parameters:
        - result = {"success": True, "path": "ANSWER:"", "answer": llm_text, "reference": references_as_integers}
        - df_definitions
        - sections_in_rag (list): A list of correctly formatted reference sections.

        Returns:
        - str: The reformatted response with consistent reference formatting and no duplicate references.
        If however there were problems extracting the references, this will return the original inputs
        - DataFrame: ALL the input df_definitions 
        - DataFrame: a subset of the input df_search_sections - the ones referenced in the LLM answer
        """


        # Extract and clean references from the raw response
        cleaned_references = result["reference"]

        # Early return if no references found. Keep the definitions but empty the search sections
        if not cleaned_references:
            headings = df_search_sections.columns.to_list()
            empty_results = pd.DataFrame([], columns = headings)
            return result["answer"], df_definitions, empty_results

        integer_references = cleaned_references

        used_definitions = []
        used_sections = []
        reference_string = ""
        number_of_definitions = len(df_definitions)
        for reference in integer_references:
            if reference <= number_of_definitions:
                row_number = reference - 1
                document_name = self.corpus.get_document(df_definitions.iloc[row_number]["document"]).name
                section_reference = df_definitions.iloc[row_number]["section_reference"]
                if section_reference == "":
                    reference_string += f"The definitions in {document_name}  \n"
                else:
                    reference_string += f"Definition {section_reference} from {document_name}  \n"
                used_definitions.append(row_number)
            else:
                row_number = reference - number_of_definitions - 1
                document_name = self.corpus.get_document(df_search_sections.iloc[row_number]["document"]).name
                section_reference = df_search_sections.iloc[row_number]["section_reference"]
                if section_reference == "":
                    reference_string += f"The document {document_name}  \n"
                else:
                    reference_string += f"Section {section_reference} from {document_name}  \n"
                used_sections.append(row_number)

        # Now update df_definitions, df_search_sections so that they only contained the sections referenced by the answer
        # df_definitions = df_definitions.iloc[used_definitions] I think I want to keep all the definitions
        df_search_sections = df_search_sections.iloc[used_sections]
        df_definitions = df_definitions.iloc[used_definitions]

        # Reconstruct the answer with reformatted references
        formatted_references = f"  \nReference:  \n{reference_string}"
        return result['answer'] + formatted_references, df_definitions, df_search_sections


#     def enrich_user_request_for_documentation(self, user_content, messages_without_rag, model_to_use="gpt-3.5-turbo"):
#         """
#         Enhances a user's request for documentation based on the conversation history. It constructs a standalone request
#         for documentation, utilizing the most recent conversation history to formulate a question that specifies what documentation
#         is required.

#         Parameters:
#         - user_content (str): The latest user content to be used for generating documentation requests.
#         - messages_without_rag (list): A list of message dictionaries that exclude RAG content, to be used as conversation history.
#         - model_to_use (str, optional): Specifies the AI model to use for generating the documentation request. Defaults to "gpt-3.5-turbo".

#         Returns:
#         - str: The enhanced documentation request generated by the model.
#         """
#         logger.info("Enriching user request for documentation based on conversation history.")

#         # Preparing the initial system message to guide the model in request generation
#         system_content = "You are assisting a user to construct a stand alone request for documentation from a conversation. \
# At the end of the conversation they have asked a question about the documentation they require. Your job is to review the conversation history and to respond with this question \
# 'What documentation is required as evidence for ...' where you need to replace the ellipses with a short description of the most recent conversation history. Try to keep the question short and general."
        
#         # Create a complete list of messages excluding the system message
#         messages_copy = copy.deepcopy(messages_without_rag)
#         messages_copy.append({'role': 'user', 'content': user_content})
        
#         # Truncate messages list to meet a specific token limit and ensure there is space for the system message
#         system_message={'role': 'system', 'content': system_content}
#         truncated_messages = self._truncate_message_list([system_message], messages_copy, token_limit=3500)
#         # NOTE, the truncated_messages will now contain the system message

#         # Generate the enhanced documentation request using the specified AI model
#         response = self.openai_client.chat.completions.create(
#             model=model_to_use,
#             temperature=1.0,
#             max_tokens=200,
#             messages=truncated_messages
#         )

#         # Extract the initial response and log information for review
#         initial_response = response.choices[0].message.content
#         logger.info(f"{self.user_name} original question: {user_content}")
#         logger.info(f"System enhanced question: {initial_response}")

#         # Check if the response starts as expected; log a warning if not
#         if not initial_response.startswith('What documentation is required as evidence for'):
#             logger.warning("The function did not enrich the user request for documentation as expected, which may create problems.")

#         return initial_response