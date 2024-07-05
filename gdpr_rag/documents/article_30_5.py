import pandas as pd
from regulations_rag.document import Document
from regulations_rag.regulation_table_of_content import StandardTableOfContent


from regulations_rag.reference_checker import EmptyReferenceChecker


class Article_30_5(Document):
    def __init__(self, path_to_manual_as_csv_file = "./inputs/documents/article_30_5.csv"):
        reference_checker = EmptyReferenceChecker()

        self.document_as_df = pd.read_csv(path_to_manual_as_csv_file, sep="|", encoding="utf-8", na_filter=False)  
        # Check for NaN values in the DataFrame
        if self.document_as_df.isna().any().any():
            msg = f'Encountered NaN values while loading the GDPR manual. This will cause ugly issues with the get_regulation_detail method'
            logger.error(msg)
            raise ValueError(msg)

        document_name = "WORKING PARTY 29 POSITION PAPER on the derogations from the obligation to maintain records of processing activities pursuant to Article 30(5) GDPR"
        super().__init__(document_name, reference_checker=reference_checker)
        if not self.check_columns():
            raise AttributeError(f"The input csv file for the Article_30_5 class does not have the correct column headings")

    def check_columns(self):
        expected_columns = ["section_reference", "heading", "text"] 

        actual_columns = self.document_as_df.columns.to_list()
        for column in expected_columns:
            if column not in actual_columns:
                print(f"{column} not in the DataFrame version of the manual")
                return False
        return True

    def get_text(self, section_reference, add_markdown_decorators = True, add_headings = True, section_only = True):
        if section_reference == "" or section_reference == "all":
            return self.document_as_df.iloc[0]['text']
        else:
            return ""

    def get_heading(self, section_reference, add_markdown_decorators = False):
        if section_reference == "" or section_reference == "all":
            return "Entire document"
        else:
            return ""

    def get_toc(self):
        return StandardTableOfContent(root_node_name = self.name, reference_checker = self.reference_checker, regulation_df = self.document_as_df)

