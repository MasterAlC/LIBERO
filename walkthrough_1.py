import os
os.environ["MUJOCO_GL"] = "glfw"

print("MUJOCO_GL =", os.environ.get("MUJOCO_GL"))


from IPython.display import display
from PIL import Image
from termcolor import colored

import torch
import torchvision

from libero.libero import benchmark, get_libero_path, set_libero_default_path
from libero.libero.envs import OffScreenRenderEnv

import h5py
from libero.libero.utils.dataset_utils import get_dataset_info
from IPython.display import HTML
import imageio
import cv2


# - Load default file paths -
benchmark_root_path = get_libero_path("benchmark_root")
init_states_default_path = get_libero_path("init_states")
datasets_default_path = get_libero_path("datasets")
bddl_files_default_path = get_libero_path("bddl_files")

print("Default benchmark root path: ", benchmark_root_path)
print("Default dataset root path: ", datasets_default_path)
print("Default bddl files root path: ", bddl_files_default_path)

# - Viewing available benchmarks -
benchmark_dict = benchmark.get_benchmark_dict()
print(benchmark_dict)

# - Check integrity of benchmarks -
# Initialise a benchmark
benchmark_instance = benchmark_dict["libero_10"]()
num_tasks = benchmark_instance.get_num_tasks()

print(f"{num_tasks} tasks in the benchmark: {benchmark_instance.name}")

# Check all tasks and their bddl file names
task_names = benchmark_instance.get_task_names()
print("The benchmark contains the following tasks:")
for i in range(num_tasks):
    task_name = task_names[i]
    task = benchmark_instance.get_task(i)
    bddl_file = os.path.join(
        bddl_files_default_path, task.problem_folder, task.bddl_file
    )
    print(f"\t {task_name}, detailed definition stored in {bddl_file}")
    if not os.path.exists(bddl_file):
        print(
            colored(
                f"[error] bddl file {bddl_file} could not be found. Check your paths",
                "red",
            )
        )

# Check integrity of init files
task_names = benchmark_instance.get_task_names()
for i in range(num_tasks):
    task_name = task_names[i]
    task = benchmark_instance.get_task(i)
    init_states_path = os.path.join(
        init_states_default_path, task.problem_folder, task.init_states_file
    )
    if not os.path.exists(init_states_path):
        print(
            colored(
                f"[error] the init states {init_states_path} could not be found. Check your paths",
                "red",
            )
        )
    print(f"An example of an init file is named like this: {task.init_states_file}")

    # Load torch init files
    init_states = benchmark_instance.get_task_init_states(0)
    print(init_states.shape)

# - Visualise all init states of a task -
task_id = 1
task = benchmark_instance.get_task(task_id)

env_args = {
    "bddl_file_name": os.path.join(
        bddl_files_default_path, task.problem_folder, task.bddl_file
    ),
    "camera_heights": 128,
    "camera_widths": 128,
}

env = OffScreenRenderEnv(**env_args)

init_states = benchmark_instance.get_task_init_states(task_id)

env.seed(0)


def make_grid(images, nrow=8, padding=2, normalize=False, pad_value=0):
    """Make a grid of images. Make sure images is a 4D tensor in the shape of (B x C x H x W)) or a list of torch tensors."""
    grid_image = torchvision.utils.make_grid(
        images, nrow=nrow, padding=padding, normalize=normalize, pad_value=pad_value
    ).permute(1, 2, 0)
    return grid_image


images = []
env.reset()
for eval_index in range(len(init_states)):
    env.set_init_state(init_states[eval_index])

    for _ in range(5):
        obs, _, _, _ = env.step([0.0] * 7)
    images.append(torch.from_numpy(obs["agentview_image"]).permute(2, 0, 1))

grid_image = make_grid(images, nrow=10, padding=2, pad_value=0)
img = Image.fromarray(grid_image.numpy()[::-1])
img.save("init_states_grid.png")
print("Image saved to init_states_grid.png")
img.show()
env.close()

# - View demonstration file info and replay a trajectory -
# Load demo files
demo_files = [os.path.join(datasets_default_path, benchmark_instance.get_task_demonstration(i)) for i in range(num_tasks)]
for demo_file in demo_files:
    if not os.path.exists(demo_file):
        print(colored(f"[error] demo file {demo_file} cannot be found. Check your paths", "red"))
print("All demo files exist.")

# Print dataset info
example_demo_file = demo_files[task_id]
print("Example Demo File Path", example_demo_file)

get_dataset_info(example_demo_file)

with h5py.File(example_demo_file, "r") as f:
    images = f["data/demo_0/obs/agentview_rgb"][()]
    
video_writer = imageio.get_writer("output.mp4", fps=60)
for image in images:
    video_writer.append_data(image[::-1])
    
video_writer.close()