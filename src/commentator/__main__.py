import openai
import click
import glob
import os
import ast

# import commentator
from . import commentator

@click.command()
@click.argument('file', type=click.Path(exists=True))
@click.argument('api-key', default=commentator.api_key())
@click.option('--language', help="Optionally adds translations in the (human) language of your choice.", required=False, default=None)
def main(file, api_key, language):
    """Automatically adds comments to your code.

    See https://github.com/emeryberger/commentator for more information.
    """
    openai.api_key = api_key
    files = [file]

    for file in files:
        print(f"Commentating {file}:")
        with open(file, 'r') as f:
            code = f.read()
            for i in range(10):
                result = commentator.commentate(file, code, language)
                if result:
                    break
                print("Failed to parse. Trying again.")
        if result:
            save_path = "backup"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            with open(os.path.join(save_path, file), 'w') as f:
                f.write(code)
            with open(file, 'w') as f:
                f.write(result)
        else:
            print(f"Failed to process {file}.")
    print("Commentator complete.")

main()
