import re
import pandas as pd
from regulations_rag.regulation_reader import  load_csv_data
from regulations_rag.document import Document
from regulations_rag.reference_checker import ReferenceChecker
from regulations_rag.reference_checker import MultiReferenceChecker
from regulations_rag.regulation_table_of_content import StandardTableOfContent


class Article_47_BCR(Document):
    def __init__(self, path_to_manual_as_csv_file = "./inputs/documents/article_47_bcr.csv"):

        main = self.MainSection()
        alt = self.AltSection()
        analysis = self.AnalysisSection()
        reference_checker = MultiReferenceChecker([main, alt, analysis])

        self.document_as_df = load_csv_data(path_to_file = path_to_manual_as_csv_file)

        document_name = "Recommendations 1/2022 on the Application for Approval and on the elements and principles to be found in Controller Binding Corporate Rules"
        super().__init__(document_name, reference_checker=reference_checker)
        if not self.check_columns():
            raise AttributeError(f"The input csv file for the Article_47_BCR class does not have the correct column headings")


    def check_columns(self):
        expected_columns = ["section", "point", "section_reference", "heading", "text"] 

        actual_columns = self.document_as_df.columns.to_list()
        for column in expected_columns:
            if column not in actual_columns:
                print(f"{column} not in the DataFrame version of the manual")
                return False
        return True

    def get_text(self, section_reference, add_markdown_decorators = True, add_headings = True, section_only = True):
        text, footnotes = super().get_text_and_footnotes(section_reference, add_markdown_decorators, add_headings, section_only)
        return super()._format_text_and_footnotes(text, footnotes)

    def get_heading(self, section_reference, add_markdown_decorators = False):
        return super().get_heading(section_reference, add_markdown_decorators)

    def get_toc(self):
        return StandardTableOfContent(root_node_name = self.name, index_checker = self.reference_checker, regulation_df = self.document_as_df)



    class MainSection(ReferenceChecker):
        def __init__(self):
            exclusion_list = [] 
            index_patterns = [
                r'^\d+',   
                r'^\.\d+', 
                r'^\.\d+', 
                r'^\.\d+', 
            ]    
            text_pattern = r'(\d+(\.\d+)?(\.\d+)?(\.\d+)?)'

            super().__init__(regex_list_of_indices = index_patterns, text_version = text_pattern, exclusion_list=exclusion_list)

    class AltSection(ReferenceChecker):
        def __init__(self):
            exclusion_list = [] #
            index_patterns = [
                r'\bApplication\b',
                r'\.\s(Part|Annex)\s\d+', # "". Part" or ".Annex"
                r'\.\d+',
            ]
            text_pattern = r'Application. (Part/Annex\s\d+)?(\.(\d+))?'

            super().__init__(regex_list_of_indices = index_patterns, text_version = text_pattern, exclusion_list=exclusion_list)

    class AnalysisSection(ReferenceChecker):
        def __init__(self):
            exclusion_list = [] #
            index_patterns = [
                r'\bAnalysis\b',
                r'\s\d+', 
            ]
            text_pattern = r'Analysis (\d+)?'

            super().__init__(regex_list_of_indices = index_patterns, text_version = text_pattern, exclusion_list=exclusion_list)

