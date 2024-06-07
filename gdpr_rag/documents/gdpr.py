import pandas as pd
import re
from regulations_rag.document import Document
from regulations_rag.regulation_reader import  load_csv_data
from regulations_rag.reference_checker import ReferenceChecker
from regulations_rag.regulation_table_of_content import StandardTableOfContent



class GDPR(Document):
    def __init__(self, path_to_manual_as_csv_file = "./inputs/documents/gdpr.csv"):
        reference_checker = self.GDPRReferenceChecker()


        self.document_as_df = load_csv_data(path_to_file = path_to_manual_as_csv_file)

        document_name = "General Data Protection Regulation"
        super().__init__(document_name, reference_checker=reference_checker)
        if not self.check_columns():
            raise AttributeError(f"The input csv file for the GDPR class does not have the correct column headings")

        self.toc_reference_checker = self.GDPRToCReferenceChecker()
        self.toc = self.get_toc()


    def check_columns(self):
        expected_columns = ["chapter_number", "chapter_heading", "section_number", "section_heading", "article_number", "article_heading", "major_reference", "minor_reference", "content", "section_reference"] # this is a minimum - it could contain more

        actual_columns = self.document_as_df.columns.to_list()
        for column in expected_columns:
            if column not in actual_columns:
                print(f"{column} not in the DataFrame version of the manual")
                return False
        return True


    def get_text(self, section_reference, add_markdown_decorators = True, add_headings = True, section_only = True):
        
        ## NOTE add_markdown_decorators = True, add_headings = True, section_only = False not implemented 

        footnote_pattern = ''
        if self.toc_reference_checker.is_valid(section_reference):
            node = self.toc.get_node(section_reference)
            if not node.children:
                gdpr_reference = node.full_node_name.split('.', 1)[1]
                return self.get_text(gdpr_reference, add_markdown_decorators, footnote_pattern)
            else:
                all_article_text = ""
                for child in node.children:
                    child_node_gdpr_reference = child.full_node_name.split('.', 1)[1]
                    all_article_text = all_article_text + self.get_text(child_node_gdpr_reference, add_markdown_decorators, footnote_pattern) + "\n\n"
                return all_article_text

        if not self.reference_checker.is_valid(section_reference):
            return ""

        if section_reference == "":
            subframe = self.document_as_df
        else:
            pattern = r'^\d{1,2}$' # there is a special case here were if I ask for "2" for example, the "startswith()" query will also include 21, 22 etc
            if re.match(pattern, section_reference):
                tmp1 = self.document_as_df[self.document_as_df["section_reference"].str.startswith(section_reference + "(")] # use startswith to get children as well
                tmp2 = self.document_as_df[self.document_as_df["section_reference"] == section_reference] 
                subframe = pd.concat([tmp1, tmp2]).sort_index()
            else:
                subframe = self.document_as_df[self.document_as_df["section_reference"].str.startswith(section_reference)] # use startswith to get children as well
            parent_reference = self.reference_checker.get_parent_reference(section_reference)
            while parent_reference:
                parent_df = self.document_as_df[self.document_as_df["section_reference"] == (parent_reference)] # use equality to get only the lines for the parent
                subframe = pd.concat([subframe, parent_df]).sort_index()
                parent_reference = self.reference_checker.get_parent_reference(parent_reference)

            if len(subframe) == 0:
                return ""

        space = " "
        line_end = "\n"
        heading = ''
        formatted_regulation = ""
        if add_markdown_decorators:
            space = '&nbsp;'
            line_end = "\n\n"
            formatted_regulation = "# "

        formatted_regulation = formatted_regulation + f"{subframe.iloc[0]['article_number']} {subframe.iloc[0]['article_heading']}{line_end}"  
        for index, row in subframe.iterrows():

            line = row["content"] + line_end
            if row["minor_reference"]:
                line = 2 * 4 * space + heading + f"({row['minor_reference']}) " + line
            elif row["major_reference"]:
                line = 1 * 4 * space + heading + f"{row['major_reference']}. " + line
            else:
                line = line


            formatted_regulation = formatted_regulation + line

        return formatted_regulation


    # Note: This method will not work correctly if empty values in the dataframe are NaN as is the case when loading
    #       a dataframe form a file without the 'na_filter=False' option. You should ensure that the dataframe does 
    #       not have any NaN value for the text fields. Try running self.document_as_df.isna().any().any() as a test before you get here
    def get_heading(self, section_reference, add_markdown_decorators = False):
        ## NOTE add_markdown_decorators not implemented 
        if not self.reference_checker.is_valid(section_reference):
            return ""

        pattern = r'^\d{1,2}$' # there is a special case here were if I ask for "2" for example, the "startswith()" query will also include 21, 22 etc
        if re.match(pattern, section_reference):
            tmp1 = self.document_as_df[self.document_as_df["section_reference"].str.startswith(section_reference + "(")] # use startswith to get children as well
            tmp2 = self.document_as_df[self.document_as_df["section_reference"] == section_reference] 
            subframe = pd.concat([tmp1, tmp2], ignore_index=True).sort_index()
        else:
            subframe = self.document_as_df[self.document_as_df["section_reference"].str.startswith(section_reference)] # use startswith to get children as well
        parent_reference = self.reference_checker.get_parent_reference(section_reference)
        while parent_reference:
            parent_df = self.document_as_df[self.document_as_df["section_reference"] == (parent_reference)] # use equality to get only the lines for the parent
            subframe = pd.concat([subframe, parent_df], ignore_index=True).sort_index()
            parent_reference = self.reference_checker.get_parent_reference(parent_reference)

        if len(subframe) == 0:
            return ""

        formatted_regulation = ""
        formatted_regulation = f"Chapter {subframe.iloc[0]['chapter_number']} {subframe.iloc[0]['chapter_heading']}."  
        if subframe.iloc[0]['section_number']:
            formatted_regulation = formatted_regulation + f" Section {subframe.iloc[0]['section_number']} {subframe.iloc[0]['section_heading']}."  

        formatted_regulation = formatted_regulation + f" Article {subframe.iloc[0]['article_number']} {subframe.iloc[0]['article_heading']}."  

        return formatted_regulation

    def get_toc(self):
        # create the dataframe using chapters and articles
        gdpr_data_for_tree = []
        chapter_number = ""
        article_number = 0

        for index, row in self.document_as_df.iterrows():
            if row["chapter_number"] != chapter_number:
                chapter_number = row["chapter_number"]
                section_number = ""
                gdpr_data_for_tree.append([row["chapter_number"], True, row["chapter_heading"]])

            if row["article_number"] != article_number:
                article_number = row["article_number"]
                gdpr_data_for_tree.append([f'{chapter_number}.{row["article_number"]}', True, row["article_heading"]])

        gdpr_df_for_tree = pd.DataFrame(gdpr_data_for_tree, columns = ["section_reference", "heading", "text"])

        return StandardTableOfContent(root_node_name = self.name, index_checker = self.GDPRToCReferenceChecker(), regulation_df = gdpr_df_for_tree)

    class GDPRReferenceChecker(ReferenceChecker):
        def __init__(self):
            exclusion_list = []

            gdpr_index_patterns = [
                r'^\d{1,2}',   # Matches numbers, excluding leading zeros. - Article Number
                r'^\((?:[1-9]|[1-9][0-9])\)',   # Matches numbers within parentheses, excluding leading zeros. 
                r'^\([a-z]\)',                  # Matches single lowercase letters within parentheses.
            ]
            
            # ^Article : Matches the beginning of the string, followed by "Article ". - mandatory
            # (\d{1,2}): Captures a one or two digit number immediately following "Article ". - mandatory
            # (?:\((\d{1,2})\))?: An optional non-capturing group that contains a capturing group for a one or two digit number enclosed in parentheses. The entire group is made optional by ?, so it matches 0 or 1 times.
            # (?:\(([a-z])\))?: Another optional non-capturing group that contains a capturing group for a single lowercase letter enclosed in parentheses. This part is also optional.
            text_pattern = r'(\d{1,2})(?:\((\d{1,2})\))?(?:\(([a-z])\))?'

            super().__init__(regex_list_of_indices = gdpr_index_patterns, text_version = text_pattern, exclusion_list=exclusion_list)


    # Reference checker for TOC only
    class GDPRToCReferenceChecker(ReferenceChecker):
        def __init__(self):
            exclusion_list = []

            gdpr_index_patterns = [
                r'^\b(I|II|III|IV|V|VI|VII|VIII|IX|X|XI)\b',
                r'^\.\d{1,2}',   # Matches numbers, excluding leading zeros. - Article Number
            ]
            
            # ^Article : Matches the beginning of the string, followed by "Article ". - mandatory
            # (\d{1,2}): Captures a one or two digit number immediately following "Article ". - mandatory
            # (?:\((\d{1,2})\))?: An optional non-capturing group that contains a capturing group for a one or two digit number enclosed in parentheses. The entire group is made optional by ?, so it matches 0 or 1 times.
            # (?:\(([a-z])\))?: Another optional non-capturing group that contains a capturing group for a single lowercase letter enclosed in parentheses. This part is also optional.
            text_pattern = r'((I|II|III|IV|V|VI|VII|VIII|IX|X|XI))(?:\((\d{1,2})\))?'

            super().__init__(regex_list_of_indices = gdpr_index_patterns, text_version = text_pattern, exclusion_list=exclusion_list)

