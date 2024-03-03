# Add the necessary imports here
from math import factorial
import pandas as pd
import torch
from utils import *
import numpy as np


def solve_cv_easy(test_case: tuple) -> list:
    shredded_image, shred_width = test_case
    shredded_image = np.array(shredded_image)
    """
    This function takes a tuple as input and returns a list as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A numpy array representing a shredded image.
        - An integer representing the shred width in pixels.

    Returns:
    list: A list of integers representing the order of shreds. When combined in this order, it builds the whole image.
    """
    return []


def solve_cv_medium(input: tuple) -> list:
    combined_image_array , patch_image_array = test_case
    combined_image = np.array(combined_image_array,dtype=np.uint8)
    patch_image = np.array(patch_image_array,dtype=np.uint8)
    """
    This function takes a tuple as input and returns a list as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A numpy array representing the RGB base image.
        - A numpy array representing the RGB patch image.

    Returns:
    list: A list representing the real image.
    """
    return []


def solve_cv_hard(input: tuple) -> int:
    extracted_question, image = test_case
    image = np.array(image)
    """
    This function takes a tuple as input and returns an integer as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A string representing a question about an image.
        - An RGB image object loaded using the Pillow library.

    Returns:
    int: An integer representing the answer to the question about the image.
    """
    return 0


def solve_ml_easy(input: pd.DataFrame) -> list:
    data = pd.DataFrame(data)

    """
    This function takes a pandas DataFrame as input and returns a list as output.

    Parameters:
    input (pd.DataFrame): A pandas DataFrame representing the input data.

    Returns:
    list: A list of floats representing the output of the function.
    """
    return []


def solve_ml_medium(input: list) -> int:
    """
    This function takes a list as input and returns an integer as output.

    Parameters:
    input (list): A list of signed floats representing the input data.

    Returns:
    int: An integer representing the output of the function.
    """
    return 0



def solve_sec_medium(input: torch.Tensor) -> str:
    img = torch.tensor(img)
    """
    This function takes a torch.Tensor as input and returns a string as output.

    Parameters:
    input (torch.Tensor): A torch.Tensor representing the image that has the encoded message.

    Returns:
    str: A string representing the decoded message from the image.
    """
    return ''

def solve_sec_hard(input:tuple)->str:
    """
    This function takes a tuple as input and returns a list a string.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A key 
        - A Plain text.

    Returns:
    list:A string of ciphered text
    """
    
    return ''

def solve_problem_solving_easy(input: tuple) -> list:
    l = input[0]
    x = input[1]
    dic = {}

    for ele in l:
        if ele in dic:
            dic[ele] += 1
        else:
            dic[ele] = 1
    sorted_dict = dict(sorted(dic.items(), key=lambda x: (-x[1], x[0])))
    first_x_elements = {key: sorted_dict[key] for key in list(sorted_dict)[:x]}
    result = list(first_x_elements.keys())
    """
    This function takes a tuple as input and returns a list as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A list of strings representing a question.
        - An integer representing a key.

    Returns:
    list: A list of strings representing the solution to the problem.
    """
    return result


def solve_problem_solving_medium(input: str) -> str:
    """
    This function takes a string as input and returns a string as output.

    Parameters:
    input (str): A string representing the input data.

    Returns:
    str: A string representing the solution to the problem.
    """
    input_string = input
    stack = [] 
    num = 0 
    current_string = ''
    for i in input_string:
      if i.isdigit():
        num = 10 * num + int(i)
      elif i == "[":
        stack.append((current_string, num))
        current_string = ''
        num = 0
      elif i == "]":
        previous_char , num_of_rept = stack.pop()
        current_string = previous_char + num_of_rept * current_string
      else:
        current_string += i
    return current_string


def solve_problem_solving_hard(input: tuple) -> int:
    m = input[0]
    n = input[1]
    """
    This function takes a tuple as input and returns an integer as output.

    Parameters:
    input (tuple): A tuple containing two integers representing m and n.

    Returns:
    int: An integer representing the solution to the problem.
    """
    return factorial(m+n-2) // factorial(m-1) // factorial(n-1)


riddle_solvers = {
    'cv_easy': solve_cv_easy,
    'cv_medium': solve_cv_medium,
    'cv_hard': solve_cv_hard,
    'ml_easy': solve_ml_easy,
    'ml_medium': solve_ml_medium,
    'sec_medium_stegano': solve_sec_medium,
    'sec_hard':solve_sec_hard,
    'problem_solving_easy': solve_problem_solving_easy,
    'problem_solving_medium': solve_problem_solving_medium,
    'problem_solving_hard': solve_problem_solving_hard
}

if __name__=="__main__":
    # res = solve_problem_solving_easy((["pharaoh","sphinx","pharaoh","pharaoh","nile","sphinx","pyramid","pharaoh","sphinx","sphinx"],3))
    # res = solve_problem_solving_hard((3 , 2))
    res = solve_problem_solving_medium("3[d1[e2[l]]]")
    print(res)