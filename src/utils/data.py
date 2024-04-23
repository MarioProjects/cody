"""
Data utility functions for the Cody project.
"""

def is_valid_python(code_to_check: str) -> bool:
    ''' Check if the given code is a valid python code or not.
    Args:
        code_to_check (str): The code to check.

    Returns:
        bool: True if the code is valid, False otherwise.
    '''
    try:
        compile(code_to_check, "<string>", "exec")
        return True
    except SyntaxError:
        return False
    

def break_into_lines(s: str, max_line_length: int = 80, indentation: int = 4) -> str:
    ''' Given a string, break it into lines of length at most `max_line_length`
    add a given indentation to each line except the first one

    Args:
        s (str): The string to break into lines
        max_line_length (int): The maximum length of each line
        indentation (int): The number of spaces to add at the beginning of each line

    Returns:
        str: The string broken into lines
    '''
    lines = []
    while len(s) > max_line_length:
        idx = s.rfind(" ", 0, max_line_length)
        if idx == -1:
            idx = max_line_length
        lines.append(s[:idx])
        s = s[idx+1:]
    lines.append(s)
    
    result = ""
    for i, line in enumerate(lines):
        if i == 0:
            result += line
        else:
            result += " " * indentation + line
        if i < len(lines) - 1:
            result += "\n"
    return result


def insert_instruction_as_docstring(instruction: str, code_def: str) -> str:
    ''' Insert the instruction as docstring in the function

    Example:

    instruction:
    Write a function to find the number of distinct states in a given matrix.

    code_def:
    def find_num_distinct_states(matrix):
        states = set()
        for row in matrix:
            state = "".join([str(x) for x in row])
            states.add(state)
        return len(states)

    output:
    def find_num_distinct_states(matrix):
        """ Write a function to find the number of distinct states in a given matrix.
        """
        states = set()
        for row in matrix:
            state = "".join([str(x) for x in row])
            states.add(state)
        return len(states)
    '''
    lines = code_def.split("\n")
    lines.insert(1, f'    """ {break_into_lines(instruction)}\n    """')
    return "\n".join(lines)


def break_into_definition_and_solution(full_function: str) -> tuple:
    ''' Break a full function definition and solution into defintion and solution

    Example:

    input:
    def find_num_distinct_states(matrix):
        """ Write a function to find the number of distinct states in a given matrix.
        This can continue...
        """
        states = set()
        for row in matrix:
            state = "".join([str(x) for x in row])
            states.add(state)
        return len(states)

    output:
    definition:
    def find_num_distinct_states(matrix):
        """ Write a function to find the number of distinct states in a given matrix.
        This can continue...
        """

    solution:
        states = set()
        for row in matrix:
            state = "".join([str(x) for x in row])
            states.add(state)
        return len(states)
    '''
    # Try to find both types of docstring starts
    triple_double_quotes_start = full_function.find('"""')
    triple_single_quotes_start = full_function.find("'''")
    
    # Determine the actual start depending on which comes first (and if they exist)
    if triple_double_quotes_start == -1 and triple_single_quotes_start == -1:
        raise ValueError("No docstring found.")
    elif triple_double_quotes_start == -1:
        docstring_start = triple_single_quotes_start
        docstring_delimiter = "'''"
    elif triple_single_quotes_start == -1 or triple_double_quotes_start < triple_single_quotes_start:
        docstring_start = triple_double_quotes_start
        docstring_delimiter = '"""'
    else:
        docstring_start = triple_single_quotes_start
        docstring_delimiter = "'''"
    
    # Find the end of the docstring
    docstring_end = full_function.find(docstring_delimiter, docstring_start + 3)
    if docstring_end == -1:
        raise ValueError("Docstring not properly closed.")
    
    # Split the definition and the solution
    definition = full_function[:docstring_end+3]
    solution = full_function[docstring_end+4:]
    
    return definition, solution
