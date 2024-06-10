import logging

from abc import ABC, abstractmethod

from regulations_rag.corpus_index import CorpusIndex
from regulations_rag.rerank import RerankAlgos, rerank

# Create a logger for this module
logger = logging.getLogger(__name__)
DEV_LEVEL = 15
logging.addLevelName(DEV_LEVEL, 'DEV')       




import os
from regulations_rag.embeddings import get_closest_nodes
from regulations_rag.standard_regulation_index import required_columns_workflow, load_parquet_data
from regulations_rag.embeddings import num_tokens_from_string

from gdpr_rag.gdpr_corpus import GDPRCorpus
import pandas as pd

class GDPRCorpusIndex(CorpusIndex):
    def __init__(self, key):
        #key = os.getenv('encryption_key_gdpr')


        corpus = GDPRCorpus("./gdpr_rag/documents/")
        index_folder = "./inputs/index/"
        index_df = pd.DataFrame()
        for filename in os.listdir(index_folder):
            if filename.endswith(".parquet"):  
                filepath = os.path.join(index_folder, filename)
                df = load_parquet_data(filepath, key)
                index_df = pd.concat([index_df, df], ignore_index = True)

        user_type = "a Controller"
        corpus_description = "the General Data Protection Regulation (GDPR)"


        super().__init__(user_type, corpus_description, corpus)

        self.definitions = index_df[index_df['source'] == 'definitions'].copy(deep=True)
        self.index = index_df[index_df['source'] != 'definitions'].copy(deep=True)
        self.workflow = pd.DataFrame([], columns = required_columns_workflow)

    def get_relevant_definitions(self, user_content, user_content_embedding, threshold):
        """
        Retrieves definitions close to the given user content embedding.

        Parameters:
        -----------
        user_content : str
            The users question
        user_content_embedding : ndarray
            The embedding vector of the user's content.
        threshold : float
            The similarity threshold for relevant definitions.

        Returns:
        --------
        DataFrame
            A DataFrame with with a column 'document', "section_reference" and 'definition' containing the text of the close definitions.
        """
        relevant_definitions = get_closest_nodes(self.definitions, embedding_column_name = "embedding", content_embedding = user_content_embedding, threshold = threshold)

        if not relevant_definitions.empty:
            logger.log(DEV_LEVEL, "--   Relevant Definitions")
            relevant_definitions["definition"] = relevant_definitions.apply(lambda row: self.corpus.get_text(row["document"], row["section_reference"]), axis=1)
            for index, row in relevant_definitions.iterrows():
                logger.log(DEV_LEVEL, f'{row["cosine_distance"]:.4f}: {row["text"]}')
        else:
            logger.log(DEV_LEVEL, "--   No relevant definitions found")

        return relevant_definitions

    def cap_rag_section_token_length(self, relevant_sections, capped_number_of_tokens):
        #relevant_sections["regulation_text"] = corpus.get_text(relevant_sections["document"], relevant_sections["section_reference"])
        relevant_sections["regulation_text"] = ""
        relevant_sections["token_count"] = 0
        for index, row in relevant_sections.iterrows():
            text = self.corpus.get_text(row["document"], row["section_reference"])
            relevant_sections.loc[index, "regulation_text"] = text
            relevant_sections.loc[index, "token_count"] = num_tokens_from_string(text)

        # Initialize the cumulative sum and the counter 'n'
        cumulative_sum = 0
        counter = 0
        n = 0

        # Correct loop to apply the specific logic
        for index, row in relevant_sections.iterrows():
            next_cumulative_sum = cumulative_sum + row["token_count"]
            # Condition to check before exceeding the cap
            if next_cumulative_sum > capped_number_of_tokens:
                n = counter  # Correct 'n' for 1-based index as per user specification
                break
            else:
                cumulative_sum = next_cumulative_sum
            counter += 1

        # Apply boundary conditions
        if n == 0:  # If 'n' has not been updated, check the boundary conditions
            if relevant_sections["token_count"].iloc[0] > capped_number_of_tokens:
                n = 1
            else:
                n = len(relevant_sections)  # Set 'n' to the total length if cap never exceeded

        if n != len(relevant_sections):
            logger.log(DEV_LEVEL, f"--   Token capping reduced the number of reference sections from {len(relevant_sections)} to {n}")

        final_row = min(n, 5)
        top_subset_df = relevant_sections.nsmallest(final_row, 'cosine_distance').reset_index(drop=True)

        return top_subset_df



    def get_relevant_sections(self, user_content, user_content_embedding, threshold, rerank_algo = RerankAlgos.NONE):
        """
        Retrieves sections close to the given user content embedding.

        Parameters:
        -----------
        user_content_embedding : ndarray
            The embedding vector of the user's content.
        threshold : float
            The similarity threshold for relevant sections.

        Returns:
        --------
        DataFrame
            A DataFrame with sections close to the user content embedding. This method also adds the content of the manual
            to the DataFrame in the column "document", "section_reference", "regulation_text"
        """
        relevant_sections = get_closest_nodes(self.index, embedding_column_name = "embedding", content_embedding = user_content_embedding, threshold = threshold)         
        n = rerank_algo.params["initial_section_number_cap"]
        relevant_sections = relevant_sections.nsmallest(n, 'cosine_distance')      
        logger.log(DEV_LEVEL, f"Selecting the top {n} items based on cosine-similarity score")
        for index, row in relevant_sections.iterrows():
            logger.log(DEV_LEVEL, f'{row["cosine_distance"]:.4f}: {row["document"]:>20}: {row["section_reference"]:>20}: {row["source"]:>15}: {row["text"]}')

        if not relevant_sections.empty:
            logger.log(DEV_LEVEL, "--   Relevant sections found")
            rerank_algo.params["user_question"] = user_content
            reranked_sections = rerank(relevant_sections=relevant_sections, rerank_algo=rerank_algo).copy(deep=True)        
            capped_sections = self.cap_rag_section_token_length(reranked_sections, rerank_algo.params["final_token_cap"])
            relevant_sections = capped_sections
            relevant_sections["regulation_text"] = relevant_sections.apply(lambda row: self.corpus.get_text(row["document"], row["section_reference"]), axis=1)
        else:
            logger.log(DEV_LEVEL, "--   No relevant sections found")
            columns = self.index.columns.to_list()
            columns.append("regulation_text")
            relevant_sections = pd.DataFrame([], columns = columns)
            

        return relevant_sections


    def get_relevant_workflow(self, user_content_embedding, threshold):
        """
        Retrieves workflow steps close to the given user content embedding if available.

        Parameters:
        -----------
        user_content_embedding : ndarray
            The embedding vector of the user's content.
        threshold : float
            The similarity threshold for relevant workflow steps.

        Returns:
        --------
        DataFrame
            A DataFrame with workflow steps close to the user content embedding.
            Returns an empty DataFrame if no workflow information is available.
        """
        if len(self.workflow) > 0:
            return get_closest_nodes(self.workflow, embedding_column_name = "embedding", content_embedding = user_content_embedding, threshold = threshold)
        else:
            return pd.DataFrame([], columns = required_columns_workflow)
