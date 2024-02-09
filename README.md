# Commentator

by [Emery Berger](https://emeryberger.com)

Commentator uses OpenAI's large language model (GPT) to add high-level
explanatory comments, docstrings, *and types* to Python code.

[![PyPI](https://img.shields.io/pypi/v/python-commentator.svg)](https://pypi.org/project/python-commentator/)
[![downloads](https://static.pepy.tech/badge/python-commentator)](https://pepy.tech/project/python-commentator)
[![downloads/month](https://static.pepy.tech/badge/python-commentator/month)](https://pepy.tech/project/python-commentator)
![Python versions](https://img.shields.io/pypi/pyversions/python-commentator.svg?style=flat-square)

## Usage

 >  **Note**
 >
 >  Commentator needs to be connected to an [OpenAI account](https://openai.com/api/) or an Amazon Web Services account.
 >
 >  _OpenAI_
 >
 >  _Your account will need to have a positive balance for this to work_
 >  ([check your OpenAI balance](https://platform.openai.com/usage)).
 >  [Get an OpenAI key here](https://platform.openai.com/api-keys).
 > 
 >  Commentator currently defaults to GPT-4, and falls back to GPT-3.5-turbo if a request error occurs. For the newest and best
 >  model (GPT-4) to work, you need to have purchased  at least $1 in credits (if your API account was created before
 >  August 13, 2023) or $0.50 (if you have a newer API account).
 > 
 >  Once you have an API key, set it as an environment variable called `OPENAI_API_KEY`.
 > 
 >  ```bash
 >  # On Linux/MacOS:
 >  export OPENAI_API_KEY=<your-api-key>
 >  
 >  # On Windows:
 >  $env:OPENAI_API_KEY=<your-api-key>
 >  ```
 >
 >  _Amazon Bedrock_
 >
 >  **New**: Commentator now has alpha support for Amazon Bedrock, using the Claude model.
 >  To use Bedrock, you need to set three environment variables.
 >
 >  ```bash
 >  # On Linux/MacOS:
 >  export AWS_ACCESS_KEY_ID=<your-access-key>
 >  export AWS_SECRET_ACCESS_KEY=<your-secret-key>
 >  export AWS_REGION_NAME=<your-region>
 >  ```
 >
 >  If you do not already have access keys, you should be able create them by
 >  modifying this link with your own username and region:
 >
 >     https://us-east-1.console.aws.amazon.com/iam/home?region=us-east-1#/users/details/YOUR_USER_NAME?section=security_credentials
 >
 >  You also need to request access to Claude (change region as needed):
 >     https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/modelaccess
 >
 >  Commentator will automatically select which AI service to use (OpenAI or AWS Bedrock) when it detects that the appropriate
 >  environment variables have been set.

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




