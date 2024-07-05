import logging
import os
import pandas as pd

from regulations_rag.file_tools import load_parquet_data
from regulations_rag.corpus_index import DataFrameCorpusIndex
from gdpr_rag.gdpr_corpus import GDPRCorpus

# Create a logger for this module
logger = logging.getLogger(__name__)
DEV_LEVEL = 15
logging.addLevelName(DEV_LEVEL, 'DEV')       

required_columns_workflow = ["workflow", "text", "embedding"]

class GDPRCorpusIndex(DataFrameCorpusIndex):
    def __init__(self, key):
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

        definitions = index_df[index_df['source'] == 'definitions'].copy(deep=True)
        # I now need to add the actual definitions in using the get_test method
        doc = corpus.get_document("GDPR")
        definitions['definition'] = definitions['section_reference'].apply(lambda x: doc.get_text(section_reference = x, add_markdown_decorators = False, add_headings = False))
        # ... except the get_text method adds some stuff before the defn so I strip it out
        definitions['definition'] = definitions['definition'].str.replace(r'^\s*\d+\.\s*', '', regex=True)

        index = index_df[index_df['source'] != 'definitions'].copy(deep=True)
        workflow = pd.DataFrame([], columns = required_columns_workflow)

        super().__init__(user_type, corpus_description, corpus, definitions, index, workflow)

#     def get_relevant_definitions(self, user_content, user_content_embedding, threshold):
#     def cap_rag_section_token_length(self, relevant_sections, capped_number_of_tokens):
#     def get_relevant_sections(self, user_content, user_content_embedding, threshold, rerank_algo = RerankAlgos.NONE):
#     def get_relevant_workflow(self, user_content_embedding, threshold):

