import os
import importlib.util
import inspect


import streamlit as st

import streamlit_antd_components as sac
from anytree import Node, PreOrderIter
from regulations_rag.regulation_table_of_content import StandardTableOfContent

from gdpr_rag.documents.gdpr import GDPR
from gdpr_rag.documents.article_47_bcr import Article_47_BCR
from gdpr_rag.documents.dpia import DPIA

# If there is page reload, switch to a page where init_session was called.
if 'chat' not in st.session_state:
    st.switch_page('question_answering.py')

def load_class_from_file(filepath):
    spec = importlib.util.spec_from_file_location("module.name", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Find the class that is not abstract
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ == module.__name__ and not inspect.isabstract(obj):
            return obj
    raise ValueError(f"No suitable class found in {filepath}")


# Function to add a property to every node in the tree
def add_property_to_all_nodes(root, property_name, property_value):
    for node in PreOrderIter(root):
        setattr(node, property_name, property_value)


def anytree_to_treeitem(node):
    if not hasattr(node, 'full_node_name') or node.full_node_name == '':
        return sac.TreeItem(
        label= f'{node.name}',
        children=[anytree_to_treeitem(child) for child in node.children] if node.children else None
        )
    else: 
        return sac.TreeItem(
            label= f'{node.full_node_name} {node.heading_text}',
            children=[anytree_to_treeitem(child) for child in node.children] if node.children else None
        )




def load_tree_data():
    date_ordered_list_of_documents = ['gdpr.py', 'article_30_5.py', 'article_47_bcr.py', 'decision_making.py', 'dpia.py', 'dpo.py', 'article_49_intl_transfer.py',
                                    'lead_sa.py', 'data_breach.py', 'data_portability.py', 'transparency.py', 'codes.py', 'online_services.py', 'territorial_scope.py',
                                    'video.py', 'covid_health.py', 'covid_location.py', 'consent.py', 'forgotten.py', 'protection.py']


    input_folder = './gdpr_rag/documents/'

    combined_toc = Node("Corpus")

    for filename in date_ordered_list_of_documents:
        filepath = os.path.join(input_folder, filename)
        Class = load_class_from_file(filepath)
        instance = Class()
        toc = instance.get_toc()
        class_name = Class.__name__
        add_property_to_all_nodes(toc.root, 'document', class_name)
        toc.root.parent = combined_toc

    sac_tree = anytree_to_treeitem(combined_toc.root)
    # Display the tree using sac.tree
    return combined_toc, sac_tree


def find_nth_item(tree_item, n):
    def traverse_tree(item, count):
        if count[0] == n:
            return item
        count[0] += 1
        for child in item.children:
            result = traverse_tree(child, count)
            if result:
                return result
        return None
    return traverse_tree(tree_item, [0])

def get_text_for_node(node_number):
    combined_toc = st.session_state['tree_data']
    anytree_node = find_nth_item(combined_toc, node_number)
    
    if hasattr(find_nth_item(combined_toc, node_number), "full_node_name"):
        node = find_nth_item(combined_toc, node_number)
        return st.session_state['chat'].corpus.get_document(node.document).get_text(node.full_node_name, add_markdown_decorators = True, add_headings = True, section_only = False)
    else:
        return "No selection to display yet"



if 'tree' not in st.session_state:    
    anytree_toc, sac_tree_data = load_tree_data()
    st.session_state['tree'] = sac_tree_data
    st.session_state['tree_data'] = anytree_toc

#selected = sac.tree(items=[st.session_state['tree']], label='Included Documents', index=0, size='md', return_index=True)
selected = sac.tree(items=[st.session_state['tree']], label='Included Documents', size='md', return_index=True)

st.write(get_text_for_node(selected), unsafe_allow_html=True)

# for doc in included_docs:

#     reference_checker = doc.reference_checker
#     df = doc.document_as_df
#     toc = StandardTableOfContent(root_node_name = "root", index_checker = reference_checker, regulation_df = df)


#     sac.tree(items=[
#         sac.TreeItem(doc.document_name, children=[
#             sac.TreeItem(node.get_name()),
#             sac.TreeItem(node.get_name(), children=[
#                 sac.TreeItem(node.get_name()),
#                 sac.TreeItem(node.get_name()),
#                 sac.TreeItem(node.get_name()),
#             ]),
#         ]),
#         sac.TreeItem('disabled', disabled=True),
#         sac.TreeItem('item3', children=[
#             sac.TreeItem('item3-1'),
#             sac.TreeItem('item3-2'),
#         ]),
#     ], label='GDPR Documents', index=0, format_func='title', size='md', icon='table', checkbox=True, checkbox_strict=True)