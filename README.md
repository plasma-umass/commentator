# Commentator

Commentator is a Python program that generates comments and optional
translations for Python code. It uses OpenAI's GPT-3 language model to
add high-level explanatory comments and docstrings to Python code.

[![PyPI Latest Release](https://img.shields.io/pypi/v/python-commentator.svg)](https://pypi.org/project/python-commentator/)[![Downloads](https://pepy.tech/badge/python-commentator)](https://pepy.tech/project/python-commentator) [![Downloads](https://pepy.tech/badge/python-commentator/month)](https://pepy.tech/project/python-commentator) ![Python versions](https://img.shields.io/pypi/pyversions/python-commentator.svg?style=flat-square)

## Usage

To use Commentator, you must first set up an OpenAI API key. If you
already have an API key, you can set it as an environment variable
called `OPENAI_API_KEY`. Otherwise, you can pass your API key as an
argument to the `commentator` command. (If you do not have one yet,
you can get a key here: https://openai.com/api/.)

```
$ export OPENAI_API_KEY=<your-api-key>
```

or

```
$ commentator --api-key <your-api-key> FILE1 ...
```

Commentator takes a path to a Python file and an optional language
parameter. If language is specified, Commentator translates each
docstring and comment in the code to the specified language. The
default language is English.


## Installation

To install Commentator, you can use pip:

```
$ pip install python-commentator
```

## Example

Suppose you have a file called `example.py` with the following code:

```
def absolutely(n):
    if n < 0:
        return -n
    else:
        return n
```

Run Commentator on this file to add comments and type annotations:

```
$ commentator example.py
```

The resulting code might be:

```
def absolutely(n: int) -> int:
    """
    Return the absolute value of a number.
    
    Args:
    - n (int): the number whose absolute value we want to find
    
    Returns:
    - int: the absolute value of n
    """
    if n < 0:
        # If n is negative
        # Return the negative version of n (i.e. its absolute value)
        return -n
    else:
        # Otherwise (if n is non-negative)
        # Return n as-is (it already is its absolute value)
        return n
```

Note that Commentator has added a docstring and type annotations.




