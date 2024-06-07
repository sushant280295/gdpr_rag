import shutil
import os



def deploy(base_folder):
    '''
    base_folder = ".." when running from "working" folder

    ''' 
    destination = "e:/code/chat/gdpr_rag"

    items_to_copy = [base_folder + "/gdpr_rag/",
                     base_folder + "/question_answering.py",
                     base_folder + "/inputs/",
                     base_folder + "/.gitignore",
                     base_folder + "/pages/"
                     ]



    for item_path in items_to_copy:
        # Ensure the item_path is normalized
        item_path = os.path.normpath(item_path)
        
        # Create new destination path
        relative_path = os.path.relpath(item_path, base_folder)
        new_destination = os.path.join(destination, relative_path)
        
        if os.path.isfile(item_path):
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(new_destination), exist_ok=True)
            
            # Copy the file
            shutil.copy(item_path, new_destination)
        elif os.path.isdir(item_path):
            # Ensure the destination directory does not already exist
            if os.path.exists(new_destination):
                shutil.rmtree(new_destination)

            # Copy the entire directory and its contents
            shutil.copytree(item_path, new_destination)
