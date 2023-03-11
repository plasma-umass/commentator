import ast
import openai_async
import asyncio
import logging
import openai
import os
import sys
import tqdm
from typing import cast, Optional, List, Set, Tuple, Union
logname = 'commentator.log'
logging.basicConfig(filename=logname, filemode='a', format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S', level=logging.INFO)
logging.info('Running commentator.')

successful_comments = 0

async def get_comments(programming_language: str, func_name: str, translate_text: str, the_code: str, pbar) -> Optional[openai.api_resources.Completion]:
    """
    Rewrite the following `programming_language` code by adding high-level explanatory comments, PEP 257 docstrings, 
    and PEP 484 style type annotations. Infer what each function does, using the names and computations as hints. 
    If there are existing comments or types, augment them rather than replacing them. If the existing comments are 
    inconsistent with the code, correct them. Every function argument and return value should be typed if possible. 
    Do not change any other code. 
    
    :param programming_language: a string representing the programming language associated with the code to be 
                                  commented
    :param translate_text: a string representing the text to be translated
    :param the_code: a string representing the code to be commented
    :return: an optional completion object
    """
    import httpx
    content = f'Rewrite the following {programming_language}code by adding high-level explanatory comments, PEP 257 docstrings, and PEP 484 style type annotations. Infer what each function does, using the names and computations as hints. If there are existing comments or types, augment them rather than replacing them. If the existing comments are inconsistent with the code, correct them. Every function argument and return value should be typed if possible. Do not change any other code. {translate_text} {the_code}'
    try:
        max_trials = 3
        for trial in range(max_trials):
            completion = await openai_async.chat_complete(openai.api_key, timeout=30, payload={'model': 'gpt-3.5-turbo', 'messages': [{'role': 'system', 'content': 'You are a {programming_language}programming assistant who ONLY responds with blocks of code. You never respond with text. Just code, starting with ``` and ending with ```.', 'role': 'user', 'content': content}]})
            code_block = extract_code_block(completion.json())
            if validated(the_code, code_block):
                global successful_comments
                successful_comments += 1
                # Splice in the types and the docstring from the generated function into the original function.
                the_code_ast = ast.parse(the_code)
                code_block_ast = ast.parse(code_block)
                for node in ast.walk(the_code_ast):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        tca = node
                        break
                for node in ast.walk(code_block_ast):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        cba = node
                        break
                replace_function_annotations(tca, cba)
                code_block = ast.unparse(tca)
                break
            else:
                logging.info(f'Failed to validate:\n-----\n{code_block}\n-----')
                code_block = ''
    except (openai.error.AuthenticationError, httpx.LocalProtocolError):
        print()
        print('You need an OpenAI key to use commentator. You can get a key here: https://openai.com/api/')
        print('Invoke commentator with the api-key argument or set the environment variable OPENAI_API_KEY.')
        import sys
        sys.exit(1)
    except Exception as e:
        pbar.update(1)
        return ''
    pbar.update(1)
    return code_block

def replace_function_annotations(target, source):
    """
    Replaces the docstrings, argument type annotations, and return type annotations of a function with those of another function.

    Args:
        target (ast node): The function node to be updated.
        source (ast node): The function node to use as the source for the annotations.

    Returns:
        None
    """
    # Replace argument and return type annotations
    for i, (target_arg, source_arg) in enumerate(zip(target.args.args, source.args.args)):
        if source_arg.annotation is not None:
            target_arg.annotation = source_arg.annotation
        elif i < len(source.args.defaults):
            target_arg.annotation = type(source.args.defaults[i]).__name__

    if source.returns is not None:
        target.returns = source.returns
    else:
        target.returns = ast.parse('None').body[0].value

    # Replace docstring
    if ast.get_docstring(source):
        if ast.get_docstring(target):
            target.body[0].value.s = ast.get_docstring(source)
        else:
            docstring_node = ast.Expr(ast.Str(ast.get_docstring(source)))
            target.body.insert(0, docstring_node)
            

def update_args(old_function_ast: Union[ast.FunctionDef, ast.AsyncFunctionDef], new_function_ast: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Union[ast.FunctionDef, ast.AsyncFunctionDef]:
    """
    Updates the arguments of a function defined by `old_function_ast` with the arguments of `new_function_ast`.

    Args:
        old_function_ast: The AST node of the function to be updated.
        new_function_ast: The AST node of the function whose arguments to update with.

    Returns:
        The updated AST node of the old function.
    """
    arg_names = [arg.arg for arg in old_function_ast.args.args]
    new_args = []
    for arg in new_function_ast.args.args:
        if arg.arg in arg_names:
            old_arg = old_function_ast.args.args[arg_names.index(arg.arg)]
            new_arg = ast.arg(arg=arg.arg, annotation=arg.annotation)
            new_args.append(new_arg)
        else:
            new_args.append(arg)
    old_function_ast.args.args = new_args
    return old_function_ast
test = '\ndef abs(n):\n    """ WUT """\n    # Check if integer is negative\n    if n < 0:\n        # Return the opposite sign of n (i.e., multiply n by -1)\n        return -n\n    else:\n        # Return n (which is already a positive integer or zero)\n        return n\n'
test2 = '\ndef abs(n):\n    if n < 0:\n        return -n\n    else:\n        return n\n'
import ast

def remove_code_before_function(code: str) -> str:
    """
    Remove any code above a function definition in the provided code string.

    Args:
        code: The code string to process.

    Returns:
        The code string with all code above the first function definition removed.
    """
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start_index = node.lineno - 1
            break
    else:
        return code
    lines = code.splitlines()
    return '\n'.join(lines[start_index:])

def remove_annotations(node: ast.AST) -> None:
    """
    Removes type annotations from an Abstract Syntax Tree node if they exist, both for function and variable annotations.

    Args:
        node: The AST node to remove annotations from.
    """
    if isinstance(node, ast.AnnAssign):
        del node.annotation
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for arg in node.args.args:
            arg.annotation = None
        node.returns = None

def remove_comments(node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module, ast.Expr]) -> None:
    """
    Removes comments in a Python code node.
    :param node: The code node to remove comments from.
    """
    if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef) or isinstance(node, ast.ClassDef) or isinstance(node, ast.Module):
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
            node.body[0].value.s = ''
        node.body = [n for n in node.body if not isinstance(n, ast.Expr) or not isinstance(n.value, ast.Str)]
    elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Str):
        node.value.s = ''

def compare_python_code(code1: str, code2: str) -> bool:
    """
    Compares two blocks of Python code by parsing them into ASTs and then
    removing comments and annotations from each AST before comparing their
    string representations. Returns True if the code blocks are identical and
    False otherwise.

    Args:
        code1: A string containing the first block of code to compare.
        code2: A string containing the second block of code to compare.

    Returns:
        A boolean indicating whether the two code blocks are identical.
    """
    tree1 = ast.parse(code1)
    tree2 = ast.parse(code2)
    for node in ast.walk(tree1):
        remove_comments(node)
        remove_annotations(node)
    for node in ast.walk(tree2):
        remove_comments(node)
        remove_annotations(node)
    try:
        diff = ast.unparse(tree1) == ast.unparse(tree2)
        return diff
    except:
        return False

def has_types(func_code: str) -> bool:
    """
    Check if a given function has type annotations for all its arguments and return value.

    Args:
        func_code: The code of the function to check.

    Returns:
        True if the function has type annotations for all its arguments and its return value.
        False otherwise.

    """
    tree = ast.parse(func_code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            all_typed = all([arg.annotation is not None for arg in node.args.args]) and node.returns is not None
            return all_typed
    return False

def has_docstring(func_code: str) -> bool:
    """
    Determine if a given function has a docstring.

    Args:
        func_code (str): Function code in string form.

    Returns:
        bool: True if the function has a docstring, False otherwise.
    """
    tree = ast.parse(func_code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
                return len(node.body[0].value.s) > 0
    return False

def now_has_types(code1, code2):
    tree1 = ast.parse(code1)
    tree2 = ast.parse(code2)
    for node in ast.walk(tree2):
        remove_annotations(node)
    return ast.unparse(tree1) != ast.unparse(tree2)

def extract_function_ast(program_str: str, function_name: str) -> Union[ast.FunctionDef, ast.AsyncFunctionDef]:
    """
    Extract the abstract syntax tree (AST) for a function with a given name from a given program string.

    Args:
        program_str (str): A string representing the program code.
        function_name (str): A string representing the name of the function to extract the AST for.

    Returns:
        ast.FunctionDef: The AST node representing the function definition.

    Raises:
        ValueError: If no function with the given name is found in the AST.
    """
    program_ast = ast.parse(program_str)
    function_node = next((n for n in program_ast.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == function_name), None)
    if function_node is None:
        raise ValueError(f"No function named '{function_name}' was found")
    return function_node

def extract_function_source(program_str, function_name):
    return ast.unparse(extract_function_ast(program_str, function_name))
    program_ast = ast.parse(program_str)
    function_node = next((n for n in program_ast.body if isinstance(n, ast.FunctionDef) and n.name == function_name), None)
    if function_node is None:
        raise ValueError(f"No function named '{function_name}' was found")
    return ast.unparse(function_node)

def enumerate_functions(program_str: str) -> List[str]:
    """
    Returns a list of names of functions and async functions defined in a given Python program string.

    Args:
        program_str: A Python program in string format.

    Returns:
        A list of names of functions and async functions defined in the program.
    """
    program_ast = ast.parse(program_str)
    names = [n.name for n in program_ast.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    return names

def replace_function(program_str: str, function_name: str, new_function_str: str) -> str:
    """Replace a function within a Python program with a new function.

    Args:
        program_str: A string representing a Python program.
        function_name: The name of the function to be replaced.
        new_function_str: A string representing the new function to replace the old one.

    Returns:
        A string representing the modified Python program.
    """
    program_ast = ast.parse(program_str)
    function_node = next((n for n in program_ast.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == function_name), None)
    if function_node is None:
        raise ValueError(f"No function named '{function_name}' was found")
    new_function_ast = extract_function_ast(new_function_str, function_name)
    function_node.body = new_function_ast.body
    function_node.returns = new_function_ast.returns
    update_args(function_node, new_function_ast)
    return ast.unparse(program_ast)

def extract_names(ast_node: ast.AST) -> Set[str]:
    """
    Extracts all class, function, and variable names from a parsed AST node.

    Args:
        ast_node: A parsed Abstract Syntax Tree (AST) node.

    Returns:
        A set of all the found class, function, and variable names.
    """
    names = set()
    for child in ast.iter_child_nodes(ast_node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(child.name)
        elif isinstance(child, ast.Assign):
            for target in child.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        names.update(extract_names(child))
    return names

def get_language_from_file_name(file_name: str) -> str:
    """Given a file name, extracts the extension and maps it to a programming language.

    Args:
      file_name: A string representing the name of the file.

    Returns:
      A string representing a programming language, or an empty string if the extension is not recognized.
    """
    ext = file_name.split('.')[-1]
    language_map = {'js': 'JavaScript', 'ts': 'TypeScript', 'c': 'C', 'cpp': 'C++', 'cs': 'C#', 'swift': 'Swift', 'py': 'Python', 'rs': 'Rust', 'sql': 'SQL', 'css': 'CSS', 'php': 'PHP', 'rb': 'Ruby', 'kt': 'Kotlin', 'go': 'Go', 'r': 'R', 'java': 'Java', 'h': 'C', 'hpp': 'C++', 'hxx': 'C++'}
    if ext in language_map:
        return language_map[ext]
    else:
        return ''

def find_code_start(code: str) -> int:
    """
    Finds the starting location of a code block in a string.

    Args:
        code: A string containing code.

    Returns:
        An integer representing the starting position of the code block.

    """
    lines = code.split('\n')
    i = 0
    while i < len(lines) and lines[i].strip() == '':
        i += 1
    first_line = lines[i].strip()
    if first_line == '```':
        return 3
    if first_line.startswith('```'):
        word = first_line[3:].strip()
        if len(word) > 0 and ' ' not in word:
            return len(word) + 3
    return -1
    '\n    Returns -1 if code block is not found.\n    '
test = '\n```python\ndef abs(n):\n    # Check if integer is negative\n    if n < 0:\n        # Return the opposite sign of n (i.e., multiply n by -1)\n        return -n\n    else:\n        # Return n (which is already a positive integer or zero)\n        return n\n```\n'

def extract_code_block(completion: dict) -> str:
    """
    Extracts code block from the given completion dictionary.

    Args:
        completion (dict): Completion dictionary containing text and other data.

    Returns:
        str: Extracted code block from the completion dictionary.
    """
    c = completion
    text = c['choices'][0]['message']['content']
    first_index = find_code_start(text)
    second_index = text.find('```', first_index + 1)
    if first_index == -1 or second_index == -1:
        code_block = text
    else:
        code_block = text[first_index:second_index]
    return code_block

def validated(the_code: str, code_block: str) -> bool:
    """Check if code block is valid using AST parsing and code comparison.

    Args:
        the_code: A string representing the original code.
        code_block: A string representing the code block to validate.

    Returns:
        A boolean indicating whether the code block is valid or not.
    """
    try:
        result_ast = ast.parse(code_block)
    except:
        return False
    #if result_ast:
    #    if not compare_python_code(remove_code_before_function(the_code), remove_code_before_function(code_block)):
    #        return False
    if result_ast and has_types(code_block):
        return True
    return False

async def commentate(filename: str, code: str, language: Optional[str]=None) -> Tuple[str, int]:
    """
    This function takes in a string of code and an optional language parameter. If language is specified,
    the function translates each docstring and comment in the code to the specified language and includes the 
    translated text in the output. If language is not specified, the function does not include any translations
    in the output. The output text includes the original code, high-level explanatory comments, and any 
    translated text (if language is specified). 

    Args:
        filename (str): A string containing the file name.
        code (str): A string of code.
        language (str, optional): A language code to specify the output language of docstrings and comments. 
                                Defaults to None.

    Returns:
        str, int: A string of the processed code and the number of successfully commented functions.
    """
    if language:
        translate_text = f"Write each docstring and comment first in English, then add a newline and '---', and add the translation to {language}."
    else:
        translate_text = ''
    programming_language = get_language_from_file_name(filename) + ' '
    the_funcs = []
    for func_name in enumerate_functions(code):
        the_code = extract_function_source(code, func_name)
        if not (has_docstring(the_code) and has_types(the_code)):
            the_funcs.append(func_name)
    if len(the_funcs) == 0:
        print('All functions already commented and contain type annotations.')
    else:
        from tqdm import tqdm
        num_items = len(the_funcs)
        pbar = tqdm(total=num_items, desc='Processing functions')
        tasks = [get_comments(programming_language, f, translate_text, extract_function_source(code, f), pbar) for f in the_funcs]
        results = await asyncio.gather(*tasks)
        code_blocks = results
        for func_name, code_block in zip(the_funcs, code_blocks):
            if not code_block:
                continue
            code = replace_function(code, func_name, code_block)
    global successful_comments
    return (code, successful_comments)

def api_key() -> str:
    """
    Get the API key from the environment variable 'OPENAI_API_KEY'.
    
    :return: The value of the environment variable 'OPENAI_API_KEY'.
    :rtype: str
    """
    key = ''
    try:
        key = os.environ['OPENAI_API_KEY']
    except:
        pass
    return key
