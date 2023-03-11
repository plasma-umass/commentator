import asyncio
import openai
import click
import glob
import os
import ast

# import commentator
from . import commentator

async def do_it(file, api_key, language):
    openai.api_key = api_key
    files = [file]

    for file in files:
        print(f"Commentating {file}:")
        with open(file, 'r') as f:
            code = f.read()
            (result, successes) = await commentator.commentate(file, code, language)
        if result:
            save_path = "backup"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            with open(os.path.join(save_path, file), 'w') as f:
                f.write(code)
            with open(file, 'w') as f:
                f.write(result)
            if successes > 0:
                print(f"Successfully commentated {successes} functions.")
            else:
                print("Unable to commentate any functions. See 'commentator.log'.")
        else:
            print(f"Failed to process {file}.")
    
@click.command()
@click.argument('file', type=click.Path(exists=True))
@click.argument('api-key', default=commentator.api_key())
@click.option('--language', help="Optionally adds translations in the (human) language of your choice.", required=False, default=None)
def main(file, api_key, language):
    """Automatically adds comments to your code.

    See https://github.com/emeryberger/commentator for more information.
    """
    asyncio.run(do_it(file, api_key, language))
    print("Commentator complete.")

main()
