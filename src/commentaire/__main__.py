import openai
import click
import glob

from . import commentaire

@click.command()
@click.argument('file', type=click.Path(exists=True))
@click.argument('api-key', default=commentaire.api_key())
@click.option('--language', required=False, default=None)
def main(file, api_key, language):
    openai.api_key = api_key
    files = [file]

    for file in files:
        print(f"Processing {file}.")
        with open(file, 'r') as f:
            code = f.read()
            result = commentaire.process(code, language)
        if result:
            with open(file, 'w') as f:
                f.write(result)
        else:
            print(f"Failed to process {file}.")
    print("Commentaire complete.")

main()
