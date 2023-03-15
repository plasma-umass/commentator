import ast_comments as ast
import asyncio
import openai
import click
import glob
import os
import tqdm

# import commentator
from . import commentator

async def do_one_file(index, file, language):
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
    tasks = [do_one_file(index, file, language) for (index, file) in enumerate(file_list)]
    await asyncio.gather(*tasks)

def print_version(ctx, param, value):
    import importlib.metadata
    if not value or ctx.resilient_parsing:
        return
    click.echo(f"commentator version {importlib.metadata.metadata('python-commentator')['Version']}")
    ctx.exit(0)

@click.command()
@click.argument('file', nargs=-1, type=click.File('r'))
@click.option('--api-key', help="OpenAI key.", default=commentator.api_key(), required=False)
@click.option('--language', help="Optionally adds translations in the (human) language of your choice.", required=False, default=None)
@click.option('--version', is_flag=True, callback=print_version,
              expose_value=False, is_eager=True)
def main(file, api_key, language, version):
    """Automatically adds comments to your code.

    See https://github.com/emeryberger/commentator for more information.
    """
    asyncio.run(do_it(api_key, language, file))

main()
