import asyncio
import contextlib
import logging
import os
import re
import subprocess
import sys
import tempfile
import traceback
import typing
from collections import deque
from typing import Any
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

import ast_comments as ast # type: ignore
import httpx
import litellm # type: ignore
from rich.progress import Progress

from . import collect_types
from . import strip_comments
from . import strip_imports
from . import strip_types


litellm.set_verbose = False
logname = "commentator.log"
logging.basicConfig(
    filename=logname,
    filemode="w",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logging.info("Running Commentator.")
# logging.getLogger().setLevel(logging.INFO)
logging.getLogger().setLevel(logging.WARNING)

total_cost = 0.0

def extract_python_code(input_string: str) -> Optional[str]:
    """
    This function takes in a string of text with possibly multiple Python
    functions and extracts the first function it can find. It assumes that functions
    start with the keyword 'def' at the beginning of a line and that indentation is
    used to define the body of the function. It collects all the lines that belong to
    the same function and returns it as a string.

    Args:
    input_string (str): A string that may potentially contain Python functions.

    Returns:
    Optional[str]: A string of the first Python function found, or None if no function is found.
    """
    # Split the input string into lines
    lines = input_string.split('\n')
    # Initialize an empty list to store functions
    functions = []
    # Initialize an empty list to store the current function being extracted
    current_func = []
    # Initialize variable to track if we're inside a function definition
    in_func = False
    # Initialize variable to track the indent level of the function
    indent_level = 0
    # Loop over all lines of code
    for line in lines:
        # If the line starts with 'def ' and we're not already in a function
        if line.strip().startswith('def ') and (not in_func):
            # We've encountered the start of a new function
            in_func = True
            current_func = [line]  # Start collecting lines for this function
            # Set the indent level to the current line's indentation
            indent_level = len(line) - len(line.lstrip())
        # If we're already in a function
        elif in_func:
            # Get the current line's indentation
            current_indent = len(line) - len(line.lstrip())
            # If the line is not blank, and its indented level is deeper than 
            # or equal to the function declaration
            if line.strip() == '' or current_indent > indent_level:
                # This line is part of the current function
                current_func.append(line)
            else:
                # This line marks the end of the current function
                # Add the current function to our list of functions
                functions.append('\n'.join(current_func))
                # Reset for the next function
                in_func = False
                indent_level = 0
                # Check if this line is a new function
                if line.strip().startswith('def '):
                    in_func = True
                    current_func = [line]
                    indent_level = len(line) - len(line.lstrip())
    # If the file ends without dedenting, add the last function to the list
    if in_func:
        functions.append('\n'.join(current_func))
    # If we found at least one function, return the first one
    # Otherwise, return None
    if len(functions) >= 1:
        return functions[0]  # For now, only return first function
    else:
        return None


def prev_extract_python_code(input_str: str) -> str:
    # Pattern to match code blocks enclosed in triple backquotes
    code_block_pattern = r"```python\n(.*?)\n```"
    # Pattern to match a Python function directly (somewhat simplified)
    direct_code_pattern = r"(def\s+\w+\(.*?\):(?:\n\s+.+)+)"

    # First, try to find a code block enclosed in triple backquotes
    match = re.search(code_block_pattern, input_str, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If not found, try to match a direct Python function definition
    match = re.search(direct_code_pattern, input_str, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If no code is found, return an empty string.
    return ""


def print_key_info():
    import os

    print("You need a key (or keys) from an AI service to use CWhy.")
    print()
    print("OpenAI:")
    print("  You can get a key here: https://platform.openai.com/api-keys")
    print("  Set the environment variable OPENAI_API_KEY to your key value:")
    print("    export OPENAI_API_KEY=<your key>")
    print()
    print("Bedrock:")
    print("  To use Bedrock, you need an AWS account.")
    print("  Set the following environment variables:")
    your_key_id = "<your key id>"
    with contextlib.suppress(KeyError):
        if os.environ["AWS_ACCESS_KEY_ID"]:
            your_key_id += " (already defined)"
    print(f"    export AWS_ACCESS_KEY_ID={your_key_id}")
    your_secret_key = "<your secret key>"
    with contextlib.suppress(KeyError):
        if os.environ["AWS_SECRET_ACCESS_KEY"]:
            your_secret_key += " (already defined)"
    print(f"    export AWS_SECRET_ACCESS_KEY={your_secret_key}")
    print("    export AWS_REGION_NAME=us-west-2")
    print("  You also need to request access to Claude:")
    print(
        "   https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html#manage-model-access"
    )


successful_comments = 0

# If keys are defined in the environment, we use the appropriate service.
service = None
_DEFAULT_FALLBACK_MODELS = []

if { "USE_OLLAMA" } <= os.environ.keys():
    service = "ollama"
    _DEFAULT_FALLBACK_MODELS = ["ollama/deepseek-coder:33b-instruct", "ollama/codellama:70b-python", "ollama/llama2"]
elif { "OPENAI_API_KEY" } <= os.environ.keys():
    service = "OpenAI"
    _DEFAULT_FALLBACK_MODELS = ["openai/gpt-4", "openai/gpt-3.5-turbo"]
elif {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION_NAME",
    } <= os.environ.keys():
        service = "Bedrock"
        _DEFAULT_FALLBACK_MODELS = ["bedrock/anthropic.claude-v2:1"]
else:
    print_key_info()
    sys.exit(1)


print(f"Using {service}")


def generate_import(node):
    """
    Generate an import statement that imports from typing any types from
    typing that were declared as annotations in the given AST node.
    """
    # Find all type annotations in the node
    typing_classes = [item for item in dir(typing) if not item.startswith("_")]
    # ["List", "Tuple", "Set", "FrozenSet", "Callable", "Dict", "DefaultDict", "Deque", "Any", "TextIO", "Union", "Optional", "cast"]
    types_used = collect_types.collect_types(node)
    typing_imports = set()
    for t in types_used:
        for c in typing_classes:
            if t.startswith(c):
                typing_imports.add(c)

    # Generate the import statement
    if typing_imports:
        return "from typing import " + ", ".join(sorted(list(typing_imports)))
    else:
        return ""


def equivalent_code(the_code, code_block):

    def strip_all(code):
        the_ast = ast.parse(code)
        stripped = strip_comments.strip_comments(the_ast)
        stripped = strip_types.strip_types(ast.parse(stripped))
        stripped = strip_imports.strip_imports(ast.parse(stripped))
        return stripped
        
    stripped_the_code = strip_all(the_code)
    stripped_code_block = strip_all(code_block)

    logging.info(f"Stripped the code = \n{stripped_the_code}")
    logging.info(f"Stripped code block = \n{stripped_code_block}")
    return stripped_the_code, stripped_code_block


async def run_mypy_on_code(file_name: str, code: str) -> Tuple[list[str], int]:
    """
    Run mypy on the given code string from the given file and return the stderr output containing mypy error messages and the number of errors.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as tmp:
        tmp_file_name = tmp.name
        tmp.write(code.encode("utf-8"))
        tmp.flush()
        result = await asyncio.create_subprocess_exec(
            "mypy",
            "--strict",
            tmp_file_name,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = await result.communicate()
        os.unlink(tmp_file_name)  # Delete the temporary file
        stdout_decoded = stdout.decode("utf-8")
        error_lines = [
            line.replace(tmp_file_name, file_name)
            for line in stdout_decoded.split("\n")
            if "error:" in line
        ]
        return error_lines, len(error_lines)


def generate_prompt(
    programming_language: str,
    func_name: str,
    check: bool,
    translate_text: str,
    the_code: str,
) -> str:
    """
    Generate the request prompt based on the programming language and specifications.
    """
    if "Python" in programming_language:
        if check:
            return f"Report ALL comments in the given {programming_language} code that are inconsistent with what the code actually does. Put that report inline as a new comment to the code with an explanation prefaced by `# INCONSISTENT COMMENT`. ONLY RETURN THE UPDATED FUNCTION AND COMMENTS: {the_code}"
        else:
            return f"Add comments to all code. Add both low-level and high-level explanatory comments as per-line comments starting with #, PEP 257 docstrings, and PEP 484 style type annotations. Make sure to add comments before all loops, branches, and complicated lines of code. Infer what each function does, using the names, comments, and computations as hints. If there are existing comments or types, augment them rather than replacing them. DO NOT DELETE EXISTING COMMENTS. If existing comments are inconsistent with the code, correct them. Every function argument and return value should be typed. {translate_text}ONLY RETURN THE UPDATED FUNCTION. The code:\n{the_code}"
    elif programming_language in ["C", "C++"]:
        return f"Add comments to the following {programming_language} code. Add both low-level and high-level explanatory comments as per-line comments. Use Google's comment style. Make sure to add comments before all loops, branches, and complicated lines of code. Infer what each function does, using the names, comments, and computations as hints. If there are existing comments, augment them rather than replacing them. If existing comments are inconsistent with the code, correct them. Use swear words judiciously. {translate_text}ONLY RETURN THE UPDATED FUNCTION: {the_code}"
    else:
        prompt = f"Add comments to the following {programming_language} code. Add both low-level and high-level explanatory comments as per-line comments. Make sure to add comments before all loops, branches, and complicated lines of code. Infer what each function does, using the names, comments, and computations as hints. If there are existing comments, augment them rather than replacing them. If existing comments are inconsistent with the code, correct them. {translate_text}ONLY RETURN THE UPDATED FUNCTION: {the_code}"
        logging.info(prompt)
        return prompt


async def get_comments(
    programming_language: str,
    func_name: str,
    check: bool,
    translate_text: str,
    the_code: str,
    pbar,
    progress,
) -> Any:  # Optional[openai.api_resources.Completion]:
    import httpx

    prompt = generate_prompt(
        programming_language, func_name, check, translate_text, the_code
    )

    last_good_code_block = ""
    
    try:
        max_trials = 3
        timeout_value = 30
        for trial in range(max_trials):
            # Append mypy errors to the code as comments if there are any errors
            mypy_errors, error_count = await run_mypy_on_code("prog.py", the_code)
            error_comments = ""
            if error_count > 0:
                error_comments = "\nFix these Mypy errors:\n" + "\n".join(
                    [f"# {error}" for error in mypy_errors]
                )
            completion = await litellm.acompletion(
                model=_DEFAULT_FALLBACK_MODELS[0],
                # TO DO: retry models in order on failures
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert {programming_language}programming assistant who ONLY responds with blocks of commented and typed code. You never respond with text. Just code, starting with ``` and ending with ```." },
                    {
                        "role": "user",
                        "content": prompt + error_comments,
                    }
                ],
            )

            logging.info(completion)

            global total_cost
            try:
                total_cost += float(litellm.completion_cost(completion_response=completion))
            except litellm.exceptions.NotFoundError:
                # Log the error exactly once.
                no_cost_model_reported: bool
                try:
                    if no_cost_model_reported:
                        pass
                except NameError:
                    logging.info(f"No cost model found for the current model: {_DEFAULT_FALLBACK_MODELS[0]}")
                    no_cost_model_reported = True

            code_block = completion["choices"][0]["message"]["content"]

            logging.info(code_block)

            prev_code_block = code_block
            code_block = extract_python_code(code_block)

            if not code_block:
                logging.warning(
                    "Failed to extract code from this block:\n" + prev_code_block
                )
                continue
            else:
                logging.info("AFTER extraction: " + code_block)

            if check:
                if "INCONSISTENT" in code_block:
                    print(f"inconsistency found:\n{code_block}")
                break

            logging.info(f"PROCESSING {code_block}")

            if validated(the_code, code_block):
                logging.info(f"Validated code block:\n-----\n{code_block}\n-----")
                global successful_comments
                successful_comments += 1
                # If the commented version is equivalent to the uncommented version, use it.
                try:
                    stripped_the_code, stripped_code_block = equivalent_code(
                        the_code, code_block
                    )
                    if stripped_the_code == stripped_code_block:
                        logging.info(
                            f"CODE EQUIVALENT\n=====\n{stripped_the_code}\n=====\n{stripped_code_block}"
                        )
                        last_good_code_block = code_block
                    else:
                        logging.info(f"CODE DIFFERENT {len(stripped_the_code)} {len(stripped_code_block)}\n")
                        code_block = ""
                        continue
                except IndentationError:
                    logging.info("Indentation error.")
                    code_block = ""
                    continue
            else:
                code_block = ""

    except httpx.LocalProtocolError:
        print_key_info()
        sys.exit(1)
    except httpx.ReadTimeout:
        # exponential backoff
        timeout_value *= 2
    except litellm.exceptions.PermissionDeniedError:
        global service
        print("Permission denied error.")
        if service == "Bedrock":
            print("You may need to request access to Claude:")
            print(
                "https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html#manage-model-access"
            )
        sys.exit(1)
    except Exception as e:
        print(f"Commentator exception: {e}")
        print(f"Please post as an issue to https://github.com/plasma-umass/commentator")
        traceback.print_exc()
        return ""
    progress.update(pbar, advance=1)
    return last_good_code_block


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
    for i, (target_arg, source_arg) in enumerate(
        zip(target.args.args, source.args.args)
    ):
        if source_arg.annotation is not None:
            target_arg.annotation = source_arg.annotation
        elif i < len(source.args.defaults):
            target_arg.annotation = type(source.args.defaults[i]).__name__

    if source.returns is not None:
        target.returns = source.returns
    else:
        target.returns = ast.parse("None").body[0].value

    # Replace docstring
    if ast.get_docstring(source):
        if ast.get_docstring(target):
            target.body[0].value.s = ast.get_docstring(source)
        else:
            docstring_node = ast.Expr(ast.Str(ast.get_docstring(source)))
            target.body.insert(0, docstring_node)


def update_args(
    old_function_ast: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    new_function_ast: Union[ast.FunctionDef, ast.AsyncFunctionDef],
) -> Union[ast.FunctionDef, ast.AsyncFunctionDef]:
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
            new_arg = ast.arg(arg=arg.arg, annotation=arg.annotation)
            new_args.append(new_arg)
        else:
            new_args.append(arg)
    old_function_ast.args.args = new_args
    return old_function_ast


test = '\ndef abs(n):\n    """ WUT """\n    # Check if integer is negative\n    if n < 0:\n        # Return the opposite sign of n (i.e., multiply n by -1)\n        return -n\n    else:\n        # Return n (which is already a positive integer or zero)\n        return n\n'
test2 = "\ndef abs(n):\n    if n < 0:\n        return -n\n    else:\n        return n\n"


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
    return "\n".join(lines[start_index:])


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


def remove_comments(
    node: Union[
        ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module, ast.Expr
    ]
) -> None:
    """
    Removes comments in a Python code node.
    :param node: The code node to remove comments from.
    """
    if (
        isinstance(node, ast.FunctionDef)
        or isinstance(node, ast.AsyncFunctionDef)
        or isinstance(node, ast.ClassDef)
        or isinstance(node, ast.Module)
    ):
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Str)
        ):
            node.body[0].value.s = ""
        node.body = [
            n
            for n in node.body
            if not isinstance(n, ast.Expr) or not isinstance(n.value, ast.Str)
        ]
    elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Str):
        node.value.s = ""


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
            all_typed = (
                all([arg.annotation is not None for arg in node.args.args])
                and node.returns is not None
            )
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
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Str)
            ):
                return len(node.body[0].value.s) > 0
    return False


def now_has_types(code1, code2):
    tree1 = ast.parse(code1)
    tree2 = ast.parse(code2)
    for node in ast.walk(tree2):
        remove_annotations(node)
    return ast.unparse(tree1) != ast.unparse(tree2)


class FunctionExtractor(ast.NodeVisitor):
    def __init__(self, target: str):
        self.target = target.split(".")
        self.current : List[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.current.append(node.name)
        if self.current == self.target:
            self.result = node
            return
        self.generic_visit(node)
        self.current.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.current.append(node.name)
        if self.current == self.target:
            self.result = node
            return
        self.generic_visit(node)
        self.current.pop()

    def visit_ClassDef(self, node: ast.ClassDef):
        self.current.append(node.name)
        self.generic_visit(node)
        self.current.pop()


def extract_function_ast(
    program_str: str, function_name: str
) -> Union[ast.FunctionDef, ast.AsyncFunctionDef]:
    module = ast.parse(program_str)
    function_extractor = FunctionExtractor(function_name)
    function_extractor.visit(module)
    return function_extractor.result


def extract_function_source(program_str, function_name):
    return ast.unparse(extract_function_ast(program_str, function_name))


class EnumerateFunctions(ast.NodeVisitor):
    def __init__(self):
        self.names = []
        self.current_class = deque()
        self.current_function = deque()

    def visit_ClassDef(self, node):
        self.current_class.append(node.name)
        self.generic_visit(node)
        self.current_class.pop()

    def visit_FunctionDef(self, node):
        self.current_function.append(node.name)
        self.process_function(node)
        self.current_function.pop()

    def visit_AsyncFunctionDef(self, node):
        self.current_function.append(node.name)
        self.process_function(node)
        self.current_function.pop()

    def process_function(self, node: ast.AST) -> None:
        if len(self.current_class) > 0:
            name = ".".join(self.current_class)  # + '.' + node.name
        elif len(self.current_function) > 1:
            name = ".".join(self.current_function)  # + '.' + node.name
        else:
            name = node.name
        self.names.append(name)
        self.generic_visit(node)


class FunctionEnumerator(ast.NodeVisitor):
    def __init__(self):
        self.function_names = []
        self.namespace = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.namespace.append(node.name)
        self.function_names.append(".".join(self.namespace))
        self.generic_visit(node)
        self.namespace.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.namespace.append(node.name)
        self.function_names.append(".".join(self.namespace))
        self.generic_visit(node)
        self.namespace.pop()

    def visit_ClassDef(self, node: ast.ClassDef):
        self.namespace.append(node.name)
        self.generic_visit(node)
        self.namespace.pop()


def enumerate_functions(program_str: str) -> List[str]:
    """
    Returns a list of names of functions and async functions defined in a given Python program string.

    Args:
        program_str: A Python program in string format.

    Returns:
        A list of names of functions and async functions defined in the program.
    """
    try:
        module = ast.parse(program_str)
        function_enumerator = FunctionEnumerator()
        function_enumerator.visit(module)
        return function_enumerator.function_names
    except SyntaxError:
        # Failed parse
        return []


class FunctionReplacer(ast.NodeTransformer):
    def __init__(self, target: str, new_node: ast.AST):
        self.target = target.split(".")
        self.current : List[str] = []
        self.new_node = new_node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.current.append(node.name)
        if self.current == self.target:
            return self.new_node
        result = self.generic_visit(node)
        self.current.pop()
        return result

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.current.append(node.name)
        if self.current == self.target:
            return self.new_node
        result = self.generic_visit(node)
        self.current.pop()
        return result

    def visit_ClassDef(self, node: ast.ClassDef):
        self.current.append(node.name)
        result = self.generic_visit(node)
        self.current.pop()
        return result


def replace_function(
    program_str: str, function_name: str, new_function_str: str
) -> str:
    module = ast.parse(program_str)
    new_function_node = ast.parse(new_function_str).body[0]
    function_replacer = FunctionReplacer(function_name, new_function_node)
    new_module = function_replacer.visit(module)
    return ast.unparse(new_module)


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


def find_code_start(code: str) -> int:
    """
    Finds the starting location of a code block in a string.

    Args:
        code: A string containing code.

    Returns:
        An integer representing the starting position of the code block.

    """
    lines = code.split("\n")
    i = 0
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    start_line = i
    while i < len(lines) and not lines[i].strip().startswith("```"):
        i += 1
    if i >= len(lines):
        i = start_line
    first_line = lines[i]
    offset = 3
    if first_line == "```":
        return offset
    matched = re.match(r"^```(?P<language>[a-z]*)", first_line)
    if matched:
        offset += len(matched.group("language")) + 1
        word = first_line[offset:].strip()
        if len(word) >= 0 and " " not in word:
            return len(word) + offset
    return -1


def extract_code_block(completion: dict) -> str:
    """
    Extracts code block from the given completion dictionary.

    Args:
        completion (dict): Completion dictionary containing text and other data.

    Returns:
        str: Extracted code block from the completion dictionary.
    """
    c = completion
    text = c["choices"][0]["message"]["content"]
    first_index = find_code_start(text)
    second_index = text.find("```", first_index + 1)
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
        result_ast_code_block = ast.parse(code_block)
        result_ast_the_code = ast.parse(the_code)
    except:
        return False
    if result_ast_code_block and has_types(code_block):
        # TODO: extend with type checking
        return True
    return False


async def commentate(
    filename: str,
    check: bool,
    code: str,
    pbar,
    progress,
    language: Optional[str] = None,
) -> Tuple[str, int]:
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
    assert service
    if language:
        translate_text = f"Write all comments in {language}."
    else:
        translate_text = ""
    programming_language = get_language_from_file_name(filename) + " "
    the_funcs = []
    for func_name in enumerate_functions(code):
        the_code = extract_function_source(code, func_name)
        # Only try to process code without docstrings or type annotations (unless checking).
        if check or not (has_docstring(the_code) and has_types(the_code)):
            the_funcs.append(func_name)
    if len(the_funcs) == 0:
        return (code, 0)
    else:
        # from tqdm import tqdm
        # num_items = len(the_funcs)
        # pbar.total = num_items
        # pbar = tqdm(total=num_items, desc=)
        tasks = [
            get_comments(
                programming_language,
                f,
                check,
                translate_text,
                extract_function_source(code, f),
                pbar,
                progress,
            )
            for f in the_funcs
        ]
        results = await asyncio.gather(*tasks)
        code_blocks = results
        for func_name, code_block in zip(the_funcs, code_blocks):
            if not code_block:
                continue
            if not check:
                prev_code = code
                try:
                    code = replace_function(code, func_name, code_block)
                except SyntaxError:
                    code = prev_code
    import_stmt = generate_import(ast.parse(code))
    if import_stmt:
        code = import_stmt + "\n" + code
    global successful_comments
    return (code, successful_comments)


# print(generate_import(ast.parse('x = 12\n\ndef whatever(n: float) -> Dict[str]:\n    """Creates a dictionary with a pre-defined key-value pair where key is \'X\' and value is 12.\nIf the input argument n is equal to 0.1234, then a new key-value pair is added to the dictionary \nwith key \'COOL\' and value 1.\n\n:param n: A float input value.\n:return: A dictionary containing key-value pairs."""\n    d = {\'X\': 12}\n    if n == 0.1234:\n        d[\'COOL\'] = 1\n    return d\n\ndef absolutely(n: int) -> Union[int, bool]:\n    """Return the absolute value of the input integer.\n\nArgs:\n    n (int): The input integer.\n\nReturns:\n    int: The absolute value of the input integer."""\n    if n < 0:\n        return -n\n    else:\n        return n\nprint(\'WOOT\')\n')))


def api_key() -> str:
    """
    Get the API key from the environment variable 'OPENAI_API_KEY'.

    :return: The value of the environment variable 'OPENAI_API_KEY'.
    :rtype: str
    """
    key = ""
    try:
        key = os.environ["OPENAI_API_KEY"]
    except:
        pass
    return key
