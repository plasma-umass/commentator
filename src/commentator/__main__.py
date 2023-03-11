import asyncio
import openai
import click
import glob
import os
import ast

# import commentator
from . import commentator

async def do_it(api_key, language, *files):
    openai.api_key = api_key
    file_list = list(*files)

    for file in file_list:
        code = file.read()
        (result, successes) = await commentator.commentate(file.name, code, language)
        if result:
            save_path = "backup"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            with open(os.path.join(save_path, file.name), 'w') as f:
                f.write(code)
            with open(file.name, 'w') as f:
                f.write(result)
            if successes > 1:
                print(f"Successfully commentated {successes} functions.")
            elif successes == 1:
                print(f"Successfully commentated {successes} function.")
            #else:
            #    print("Unable to commentate any functions. See 'commentator.log'.")
        else:
            print(f"Failed to process {file.name}.")
    
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
