import openai
import os
import sys

import ast

def extract_names(ast_node):
    """Extracts all class, function, and variable names from a parsed AST node."""
    names = set()

    # Visit all the child nodes of the current node.
    for child in ast.iter_child_nodes(ast_node):

        # If the child node defines a class or function, add its name to the set.
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(child.name)

        # If the child node defines a variable, add its name to the set.
        elif isinstance(child, ast.Assign):
            for target in child.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)

        # Recursively visit the child node's children.
        names.update(extract_names(child))

    return names


def get_language_from_file_name(file_name):
    ext = file_name.split(".")[-1]
    language_map = {
        "js": "JavaScript",
        "ts": "TypeScript",
        "c": "C",
        "cpp": "C++",
        "cs": "C#",
        "swift": "Swift",
        "py": "Python",
        "rs": "Rust",
        "sql": "SQL",
        "css": "CSS",
        "php": "PHP",
        "rb": "Ruby",
        "kt": "Kotlin",
        "go": "Go",
        "r": "R",
        "java": "Java",
        "h": "C",
        "hpp": "C++",
        "hxx": "C++",
    }
    if ext in language_map:
        return language_map[ext]
    else:
        return ""

def find_code_start(code):
    # Split the code into lines
    lines = code.split("\n")

    # Skip empty lines at the beginning
    i = 0
    while i < len(lines) and lines[i].strip() == "":
        i += 1

    # Get the first line of code
    first_line = lines[i].strip()

    # If the first line is just "```", return 3
    if first_line == "```":
        return 3

    # If the first line starts with "```" followed by a single word, return the length of the word plus 3
    if first_line.startswith("```"):
        word = first_line[3:].strip()
        if len(word) > 0 and " " not in word:
            return len(word) + 3

    # If the first line doesn't match either of the above patterns, return -1
    return -1

test = """
```python
def abs(n):
    # Check if integer is negative
    if n < 0:
        # Return the opposite sign of n (i.e., multiply n by -1)
        return -n
    else:
        # Return n (which is already a positive integer or zero)
        return n
```
"""

#print(find_code_start(test))
#import sys
#sys.exit(0)

def process(filename, code, language=None):
    """
    This function takes in a string of code and an optional language parameter. If language is specified,
    the function translates each docstring and comment in the code to the specified language and includes the 
    translated text in the output. If language is not specified, the function does not include any translations
    in the output. The output text includes the original code, high-level explanatory comments, and any 
    translated text (if language is specified). 

    Args:
    code (str): A string of code.
    language (str, optional): A language code to specify the output language of docstrings and comments. 
                              Defaults to None.

    Returns:
    str: A string of the processed code.
    """
    if language:
        translate_text = f"Write each docstring and comment first in English, then add a newline and '---', and add the translation to {language}."
    else:
        translate_text = ""

    programming_language = get_language_from_file_name(filename)+ " "
    
    content = f"Rewrite the following {programming_language}code by adding high-level explanatory comments and docstrings, if they are not already present. Try to infer what each function does, using the names and computations as hints. If there are existing comments, augment them rather than replacing them. If the existing comments are inconsistent with the code, correct them. Provide input-output examples in comments. {translate_text} {code}"

    # print(content)
    
    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo", 
      messages=[
          {"role": "system",
           "content" : "You are a {programming_language}programming assistant who ONLY responds with blocks of code. You never respond with text. Just code, starting with ``` and ending with ```."},
          {"role": "user",
           "content": content}
      ]
    )

    c = completion
    
    text = c['choices'][0]['message']['content']

    first_index = find_code_start(text) # text.find("```")
    second_index = text.find("```", first_index + 1)
    if first_index == -1 or second_index == -1:
        # Assume that a code block was emitted that wasn't surrounded by ```.
        code_block = text
    else:
        code_block = text[first_index:second_index]
        
    if get_language_from_file_name(filename) == "Python":
        try:
            result_ast = ast.parse(code_block)
        except:
            print("WUT")
            result_ast = None

        if result_ast:
            orig_ast = ast.parse(code)
            if extract_names(orig_ast) != extract_names(result_ast):
                code_block = None
        else:
            code_block = None
            
    return code_block


def api_key():
    key = ""
    try:
        key = os.environ['OPENAI_API_KEY']
    except:
        pass
    return key
    
