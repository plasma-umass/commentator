import click
import glob
import openai
import os
import sys

def process(code, language=None):
    """
    This function takes in a string of Python code and an optional language parameter. If language is specified,
    the function translates each docstring and comment in the code to the specified language and includes the 
    translated text in the output. If language is not specified, the function does not include any translations
    in the output. The output text includes the original code, high-level explanatory comments, and any 
    translated text (if language is specified). 

    Args:
    code (str): A string of Python code.
    language (str, optional): A language code to specify the output language of docstrings and comments. 
                              Defaults to None.

    Returns:
    str: A string of the processed code.
    """
    if language:
        translate_text = f"For each docstring and comment, add a newline and '---', and then include the translation in {language}."
    else:
        translate_text = ""

    content = f"Rewrite the following Python code by adding high-level explanatory comments and docstrings, if they are not already present. Try to infer what each function does, using the names and computations as hints. {translate_text} {code}"

    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo", 
      messages=[
          {"role": "system",
           "content" : "You are a Python programming assistant who ONLY responds with blocks of code."},
          {"role": "user",
           "content": content}
      ]
    )

    c = completion
    text = c['choices'][0]['message']['content']
    first_index = text.find("```")
    second_index = text.find("```", first_index + 1)
    if first_index == -1 or second_index == -1:
        return None
    return text[first_index + 3:second_index]


def api_key():
    key = ""
    try:
        key = os.environ['OPENAI_API_KEY']
    except:
        pass
    return key
    
@click.command()
@click.argument('file', type=click.Path(exists=True))
@click.argument('api-key', default=api_key())
@click.option('--language', required=False, default=None)
def commentaire(file, api_key, language):
    openai.api_key = api_key
    files = [file]

    for file in files:
        print(f"Processing {file}.")
        with open(file, 'r') as f:
            code = f.read()
            result = process(code, language)
        with open(file, 'w') as f:
            f.write(result)
    print("Commentaire complete.")


if __name__ == '__main__':
    commentaire()
