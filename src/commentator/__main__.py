import ast_comments as ast
import asyncio
import openai
import click
import glob
import os
import tqdm

# import commentator
from . import commentator
from . import strip_comments
from . import strip_types

async def commentate_one_file(index, file, language):
    code = file.read()
    try:
        ast.parse(code)
    except SyntaxError:
        # Failed to parse.
        return
        
    function_count = 0
    for func_name in commentator.enumerate_functions(code):
        the_code = commentator.extract_function_source(code, func_name)
        if not (commentator.has_docstring(the_code) and commentator.has_types(the_code)):
            function_count += 1
            
    if function_count == 0:
        return

    from tqdm import tqdm
    pbar = tqdm(total=function_count, desc=file.name, leave=False, unit='function') # position=index,
    
    (result, successes) = await commentator.commentate(file.name, code, pbar, language)
    if result:
        save_path = "backup"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path, file.name), 'w') as f:
            # Generate import statement first.
            import_stmt = commentator.generate_import(ast.parse(code))
            if import_stmt:
                code = import_stmt + "\n" + code
            f.write(code)
        with open(file.name, 'w') as f:
            f.write(result)
        #if successes > 1:
        #    print(f"Successfully commentated {successes} functions.")
        #elif successes == 1:
        #    print(f"Successfully commentated {successes} function.")
        #else:
        #    print("Unable to commentate any functions. See 'commentator.log'.")
    else:
        pass
        #print(f"Failed to process {file.name}.")
    
async def do_it(api_key, language, *files):
    openai.api_key = api_key
    file_list = list(*files)
    tasks = [commentate_one_file(index, file, language) for (index, file) in enumerate(file_list)]
    await asyncio.gather(*tasks)

def print_version(ctx, param, value):
    import importlib.metadata
    if not value or ctx.resilient_parsing:
        return
    click.echo(f"commentator version {importlib.metadata.metadata('python-commentator')['Version']}")
    ctx.exit(0)

async def func_one_file(index, file, func):
    with open(file.name, 'r') as f:
        code = f.read()
    try:
        the_ast = ast.parse(code)
    except SyntaxError:
        # Failed to parse.
        return
        
    function_count = len(commentator.enumerate_functions(code))
    if function_count == 0:
        return

    from tqdm import tqdm
    pbar = tqdm(total=function_count, desc=file.name, leave=False, unit='function') # position=index,

    result = func(the_ast)
    if result:
        save_path = "backup"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path, file.name), 'w') as f:
            f.write(code)
        with open(file.name, 'w') as f:
            f.write(result)
    else:
        pass

async def strip_types_one_file(index, file):
    await func_one_file(index, file, strip_types.strip_types)

async def strip_comments_one_file(index, file):
    await func_one_file(index, file, strip_comments.strip_comments)
    
async def strip_types_helper(*files):
    file_list = list(*files)
    tasks = [strip_types_one_file(index, file) for (index, file) in enumerate(file_list)]
    await asyncio.gather(*tasks)

def do_strip_types(files):
    asyncio.run(strip_types_helper(files))

async def strip_comments_helper(*files):
    file_list = list(*files)
    tasks = [strip_comments_one_file(index, file) for (index, file) in enumerate(file_list)]
    await asyncio.gather(*tasks)

def do_strip_comments(files):
    asyncio.run(strip_comments_helper(files))
    
@click.command()
@click.argument('file', nargs=-1, type=click.File('r'))
@click.option('--api-key', help="OpenAI key.", default=commentator.api_key(), required=False)
@click.option('--language', help="Write all comments in the (human) language of your choice (default=English).", required=False, default=None)
@click.option('--version', is_flag=True, callback=print_version,
              expose_value=False, is_eager=True, help="Print the current version number and exit.")
@click.option('--strip-types/--no-strip-types', default=False, help="Just strip existing types and exit.")
@click.option('--strip-comments/--no-strip-comments', default=False, help="Just strip existing comments and exit.")
def main(file, api_key, language, strip_types, strip_comments):
    """Automatically adds comments to your code.

    See https://github.com/emeryberger/commentator for more information.
    """
    if strip_types:
        do_strip_types(file)
    if strip_comments:
        do_strip_comments(file)
    if strip_types or strip_comments:
        return
    asyncio.run(do_it(api_key, language, file))

main()
