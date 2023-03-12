import asyncio
import openai
import click
import glob
import os
import ast
import tqdm

# import commentator
from . import commentator

async def do_one_file(index, file, language):
    code = file.read()
    function_count = 0
    for func_name in commentator.enumerate_functions(code):
        the_code = commentator.extract_function_source(code, func_name)
        if not (commentator.has_docstring(the_code) and commentator.has_types(the_code)):
            function_count += 1
            
    from tqdm import tqdm
    pbar = tqdm(total=1, dynamic_ncols=True, desc=file.name, leave=False, unit='function') # position=index, 
    
    if function_count == 0:
        pbar.total = 1
        pbar.update(1)
        pbar.set_description(f'{file.name}: no functions need commentating.')
        # pbar.update(1)
        return

    pbar.total=function_count
    (result, successes) = await commentator.commentate(file.name, code, pbar, language)
    if result:
        save_path = "backup"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path, file.name), 'w') as f:
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
    
@click.command()
@click.argument('file', nargs=-1, type=click.File('r'))
@click.option('--api-key', help="OpenAI key.", default=commentator.api_key(), required=False)
@click.option('--language', help="Optionally adds translations in the (human) language of your choice.", required=False, default=None)
def main(file, api_key, language):
    """Automatically adds comments to your code.

    See https://github.com/emeryberger/commentator for more information.
    """
    asyncio.run(do_it(api_key, language, file))
    print("Commentator complete.")

main()
