from utils import *
import random
import copy
import math

def mutate_condition(program, img_dim, idx):
    """
    Mutate the condition of the program at the specified index.

    Args:
        program (Program): The program to mutate.
        img_dim (int): An integer representing the image dimension.
        idx (int): The index of the condition to mutate in the program.
    """
    setattr(program, 'cond_{}'.format(idx), generate_random_condition(img_dim))


def mutate_real_value(program, img_dim, idx):
    """
    Mutate the real value of the condition at the specified index.

    Args:
        program (Program): The program to mutate.
        img_dim (int): An integer representing the image dimension.
        idx (int): The index of the condition whose real value to mutate in the program.
    """
    cond = getattr(program, 'cond_{}'.format(idx))
    cond[2] = generate_real_value(cond[0], img_dim)
    setattr(program, 'cond_{}'.format(idx), cond)


def mutate_condition_type(program, img_dim, idx):
    """
    Mutate the condition type of the condition at the specified index.

    Args:
        program (Program): The program to mutate.
        img_dim (int): An integer representing the image dimension.
        idx (int): The index of the condition whose type to mutate in the program.
    """
    cond = getattr(program, 'cond_{}'.format(idx))
    old_cond_type = cond[0]
    cond_type = random.choice(["MIN", "MAX", "MEAN", "SCORE_DIFF", "CENTER"])
    cond[0] = cond_type
    if cond_type not in ["MIN", "MAX", "MEAN"] or old_cond_type not in ["MIN", "MAX", "MEAN"]:
        cond[2] = generate_real_value(cond_type, img_dim)
    setattr(program, 'cond_{}'.format(idx), cond)


def run_MH(init_program, init_queries, model, dataloader, img_dim, center_matrix, max_iter, queue_proc, max_g, g,\
           max_queries, lmh_dict, mean_norm, std_norm, device):
    """
    Perform Metropolis-Hastings algorithm for program optimization.

    This function implements the Metropolis-Hastings algorithm to optimize the input program
    based on a model, dataloader, and other specified parameters. The optimization is
    performed over a given number of iterations (max_iter).

    Args:
        init_program (Program): The initial program to be optimized.
        init_queries (float): The average number of queries for the init_program
        model (torch.nn.Module): The neural network model used for evaluation.
        dataloader (torch.utils.data.DataLoader): The data loader for input data.
        img_dim (int): An integer representing the image dimension.
        center_matrix (torch.Tensor): A matrix representing the distance of each pixel to the image center.
        max_iter (int): The maximum number of iterations for the Metropolis-Hastings algorithm.
        queue_proc (multiprocessing.Queue): A queue for inter-process communication, used to store results.
        max_g (int): The maximum number of pixels to perturb with finer granularity.
        g (int): The level of granularity.
        max_queries (int) : The maximal number of possible queries per image.
        lmh_dict (dict): A dictionary containing the 'min_values', 'mid_values', and 'max_values' for the perturbations.
        mean_norm (list[float]):  The mean values for each channel used in image normalization.
        std_norm (list[float]):  The standard deviation values for each channel used in image normalization.
        device (torch.device): The device where the model and tensors will be processed, e.g., 'cpu' or 'cuda'.

    Returns:
        None: The function puts the (best_program, best_queries) tuple into the queue_proc.
    """
    from synthesize import run_program
    model = model.to(device)
    model.eval()
    beta = 0.4
    best_program = copy.deepcopy(init_program)
    best_queries = init_queries
    best_score = math.e ** (-beta * (best_queries))
    mutation_functions = [
        (range(1, 5), 0, mutate_condition),
        (range(5, 9), 4, mutate_real_value),
        (range(9, 13), 8, mutate_condition_type),
    ]

    for iter_idx in range(max_iter):
        program = copy.deepcopy(best_program)
        idx_mutate = random.randrange(13)

        if idx_mutate == 0:
            for idx in range(1, 5):
                setattr(program, 'cond_{}'.format(idx), generate_random_condition(img_dim))
        else:
            for idx_range, scale, mut_func in mutation_functions:
                if idx_mutate in idx_range:
                    mut_func(program, img_dim, idx_mutate - scale)
                    break

        queries = run_program(program, model, dataloader, img_dim, center_matrix, max_g, g, max_queries, lmh_dict,
                              mean_norm, std_norm, device)
        score = math.e ** (-beta * (queries))
        if score == 0 or best_score == 0:
            alpha = 0
        else:
            alpha = min(1, score / best_score)
        if (random.uniform(0, 1) <= alpha) or (best_queries > queries):
            best_score = score
            best_program = copy.deepcopy(program)
            best_queries = queries

    queue_proc.put((best_program, best_queries))
