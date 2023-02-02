'''
Decorator that determines whether to return a list of strings or a single 
string depend on a function/method's arguments
'''

from inspect import signature, Parameter
import re
from typing import *
from functools import reduce


def flatten_list(x: List[List[Any]]) -> List[Any]: 
    '''
    Utility function to flatten lists of lists to a single list
    '''
    def flatten(x, y):
        x.extend(y)
        return x
    return reduce(flatten, x)


def detokenizer_wrapper(f: Callable):

    def return_correct_format(*args, **kwargs):
        if 'detokenize' not in kwargs:
            function_signature = signature(f)
            if 'detokenize' in function_signature.parameters:
                if function_signature.parameters['detokenize'].default is not Parameter.empty:
                    kwargs['detokenize'] = function_signature.parameters['detokenize'].default
                else:
                   raise ValueError(
                        "'detokenize' not found in kwargs or default kwargs. " +
                        "Please provide a keyword argument or rewrite the function " + 
                        "with 'detokenize' as a default argument."
                    )
            else:    
                raise ValueError(
                        f"'detokenize' not found in function arguments. Rewrite the " + 
                        f"function with 'detokenize' as an argument."
                    )
        if isinstance(kwargs['detokenize'], bool):
            if kwargs['detokenize']:
                from nltk.tokenize.treebank import TreebankWordDetokenizer
                detokenizer = TreebankWordDetokenizer()
                detokenize = lambda x: detokenizer.detokenize(x, True)
            else:
                detokenize = lambda x: x
            kwargs['detokenize'] = detokenize
            return f(*args, **kwargs)
        elif callable(kwargs['detokenize']):
            return f(*args, **kwargs)
        else:
            raise TypeError(
                f"Unexpected type for 'detokenize': {kwargs['detokenize']} " +
                f"'detokenize' must be a boolean (or callable if using custom detokenizer)."
            )
        
    return return_correct_format

if __name__ == '__main__':
    pass