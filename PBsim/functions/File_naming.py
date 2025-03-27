import os


def get_unique_filename(output_folder, output_type, filename, file_type ):
    """Ensure the file doesn't overwrite an existing one by appending a number if needed."""
    counter = 1
    output_folder = str(output_folder)
    
    
    if output_type == 'fig':
        os.makedirs(os.path.join(output_folder, "Figures"), exist_ok=True)
        os.makedirs(os.path.join(output_folder, "Figures\\", f"trial_{counter}"), exist_ok=True)
        file_path = os.path.join(output_folder, "Figures\\", f"trial_{counter}")

        while os.path.exists(file_path):
            file_path = os.path.join(output_folder, "Figures\\", f"trial_{counter}")
            os.makedirs(file_path, exist_ok=True)
            file_path = os.path.join(file_path, f"{filename}{file_type}")
            if not os.path.exists(file_path):
                return file_path
            counter += 1

    elif output_type == 'gif':
        
        os.makedirs(os.path.join(output_folder, "Gifs\\"), exist_ok=True)
        os.makedirs(os.path.join(output_folder, "Gifs\\", f"trial_{counter}"), exist_ok=True)
        file_path = os.path.join(output_folder, "Gifs\\", f"trial_{counter}")

        while os.path.exists(file_path):
            file_path = os.path.join(output_folder, "Gifs\\", f"trial_{counter}")
            os.makedirs(file_path, exist_ok=True)
            file_path = os.path.join(file_path, f"{filename}{file_type}")
            if not os.path.exists(file_path):
                return file_path
            counter += 1


    return file_path

    