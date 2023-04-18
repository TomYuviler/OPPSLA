import pickle
import program_
import argparse
import torch.multiprocessing as tmp
from synthesize import run_program
from utils_ import *


def attack(args):
    """
    Perform the adversarial attack using the provided settings and programs.

    Args:
        args (argparse.Namespace): An object containing the parsed command-line arguments.
            - args.model (str): The name of the pre-trained model to use for classification.
            - args.program_path (str): The path to the pickled file containing the synthesized programs.
            - args.results_path (str): Path to the directory where the results CSV file should be saved.
            - args.classes_list (List[int]): The list of classes to attack.
            - args.max_g (int): The maximum number of pixels to perturb with finer granularity.
            - args.g (int): The level of granularity.
    """
    # Clear GPU cache
    torch.cuda.empty_cache()

    # Set up device(s)
    devices = setup_devices()

    # Load test data
    test_data, img_dim = get_data_set(args, is_train=False)
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=1)

    # Load model
    model = load_model(args.model)
    model = model.to('cpu')

    # Load pre-synthesized programs
    program_dict = pickle.load(open(args.program_path, 'rb'))

    # Generate center matrix
    center_matrix = generate_center_matrix(img_dim)

    # Create results directory if it doesn't exist
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    # Prepare for multiprocessing
    num_classes = len(args.classes_list)
    num_gpus = torch.cuda.device_count()
    available_devices = tmp.Queue()
    for i in range(num_gpus):
        available_devices.put(f"cuda:{i}")
    task_list = [(i, program_dict[args.classes_list[i]], model, test_loader, img_dim, center_matrix, args.max_g, \
         args.g, devices[i % num_gpus], True, args.classes_list[i], args.results_path) for i in range(num_classes)]

    # Perform attack
    with tmp.Pool(processes=num_gpus) as pool, tqdm(total=num_classes, desc="Attacking") as pbar:
        for idx, *args_ in task_list:
            device = available_devices.get()
            pool.apply_async(run_program, args=(args_), callback=lambda _: (available_devices.put(device), pbar.update()))

        pool.close()
        pool.join()

if __name__ == '__main__':
    tmp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description='OPPSLA attack')
    parser.add_argument('--model', default='vgg16', type=str, help='model')
    parser.add_argument('--data_set', default='cifar10', type=str, help='data set - must be CIFAR-10 or ImageNet')
    parser.add_argument('--classes_list', metavar='N', type=int, nargs='+', help='classes for the synthesis process')
    parser.add_argument('--imagenet_dir', type=str, help='directory for images of ImageNet dataset')
    parser.add_argument('--program_path', type=str, help='path of the program as a pkl file')
    parser.add_argument('--results_path', default="./results_OPPSLA", type=str, help='path of the saved results')
    parser.add_argument('--g', default=0, type=int, help='level of granularity')
    parser.add_argument('--max_g', default=0, type=int, help='number of pixels with finer granularity')

    args = parser.parse_args()
    attack(args)
