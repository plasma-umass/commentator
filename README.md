# Commentaire

Commentaire is a Python program that generates comments and optional
translations for Python code. It uses OpenAI's GPT-3 language model to
add high-level explanatory comments and docstrings to Python code.

## Usage

To use Commentaire, you must first set up an OpenAI API key. If you
already have an API key, you can set it as an environment variable
called `OPENAI_API_KEY`. Otherwise, you can pass your API key as an
argument to the `commentaire` command. (If you do not have one yet,
you can get a key here: https://openai.com/api/.)

```
$ export OPENAI_API_KEY=<your-api-key>
```

or

```
$ commentaire <file> <api-key>
```

Commentaire takes a path to a Python file and an optional language
parameter. If language is specified, Commentaire translates each
docstring and comment in the code to the specified language and
includes the translated text in the output. If language is not
specified, Commentaire does not include any translations in the
output.


## Installation

To install Commentaire, you can use pip:

```
$ pip install commentaire
```


## Functionality

Commentaire has a single function, `process`, which takes in a string
of Python code and an optional language parameter. If language is
specified, the function translates each docstring and comment in the
code to the specified language and includes the translated text in the
output. If language is not specified, the function does not include
any translations in the output. The output text includes the original
code, high-level explanatory comments, and any translated text (if
language is specified).

The `commentaire` command uses `process` to add comments and
translations to Python code in a file. It processes each file in a
list of files passed as arguments to the `commentaire` command.

## Example

Suppose you have a file called `example.py` with the following code:

```
def foo(x):
    y = x + 1
    return y
```

You can run Commentaire on this file to add comments (and optionally, translations to another language):

```
$ commentaire example.py --language Spanish
```

The resulting code will be:

```
"""
This function takes in a value x and returns its incremented value.

Esta función toma un valor x y devuelve su valor incrementado.
"""
def foo(x):
    y = x + 1
    return y
```

Note that Commentaire has added high-level explanatory comments and
translated the existing docstring and comment to Spanish.




