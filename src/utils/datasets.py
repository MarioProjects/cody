from datasets import load_dataset

from src.utils.data import (
    is_valid_python,
    insert_instruction_as_docstring,
    break_into_definition_and_solution
)


def get_code_alpaca_20k():
    ''' Preprocess the Code Alpaca 20k dataset to be used in HumanEval trainings
    '''
    # Data Load
    code_alpaca_dataset = load_dataset("sahil2801/CodeAlpaca-20k")

    # Filtering
    ## Valid Python Code: 
    ### Filter the dataset to only include `output` that is valid python code
    ### We might find some code snippets that are not valid python code like:
    ### 'Height of triangle = opposite side length * sin (angle) / side length'
    code_alpaca_dataset = code_alpaca_dataset.filter(lambda x: is_valid_python(x['output']))
    
    ## Only Functions:
    ### Finally get only the examples that start with 'def', as at HumanEval we are only interested in functions
    ### and we need to learn to use the function signature with the parameters to generate the function body
    code_alpaca_dataset = code_alpaca_dataset.filter(lambda x: x['output'].startswith('def'))

    # HumanEval Specific Preprocessing
    def _preprare_prompt_completion(example):
        '''
        We need to transform Code Alpaca's data into the format tha HumanEval uses.
        To do so we use some utility functions. We can summarize the steps as going from:

        1. Intial Code Alpaca's data format:
        -----------------------------------------------------
        output = ```
        def foo():
            return 1
        ```
        instructions = "Write a function that returns 1"
        -----------------------------------------------------
        

        2. To HumanEval's data format:
        -----------------------------------------------------
        prompt = ```
        def foo():
            """ Write a function that returns 1
            """
        ```
        completion = ```
            return 1
        ```
        -----------------------------------------------------
        '''
        instruction = example["instruction"]
        code_def = example["output"]
        full = insert_instruction_as_docstring(instruction, code_def)

        # Some text might contain new lines, out of the code function definition
        # We just need the code function definition
        code = ""
        function_started = False
        for line in full.split("\n"):
            if line.startswith("def ") and not function_started:
                function_started = True
            # if line just with spaces, remove it
            elif not line.strip():
                continue
            # if the line is not empty, and not indented, and already in function break
            elif line and not line.startswith("    ") and function_started:
                break
            
            if function_started:
                code += line + "\n"

        try:
            inputs, outputs = break_into_definition_and_solution(code)
        except ValueError: # If the code is not valid, return None
            return {"code": "", "prompt": "", "completion": ""}
        
        # If not outputs, return None (excludes this example from the dataset)
        # this happens when the indentation is not correct
        if not outputs:
            return {"code": "", "prompt": "", "completion": ""}

        example["code"] = code
        example["prompt"] = inputs
        example["completion"] = outputs
        return example
    
    dataset = code_alpaca_dataset.map(_preprare_prompt_completion)

    # Remove examples that have no code
    dataset = dataset.filter(lambda x: x["code"])

    # Finally return the preprocessed dataset
    return dataset
