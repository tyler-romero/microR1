import re
from dataclasses import dataclass
from functools import partial
from random import choice, randint

system_prompt = (
    "Respond in the following format: <think> ... </think> <answer> ... </answer>"
)
system_prompt_qwen = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
)

@dataclass(frozen=True)
class TaskPreset:
    message_style: str = 'default'
    linear_max_coefficient: int = 50
    linear_max_constant: int = 50
    nim_max_pile_size: int = 50
    josephus_min_people: int = 5
    josephus_max_people: int = 20
    josephus_min_k: int = 2
    josephus_max_k: int = 7
    verbose_prompts: bool = False


task_presets = {
    'default': TaskPreset(),
    'qwen_4090': TaskPreset(
        message_style='qwen',
        linear_max_coefficient=25,
        linear_max_constant=25,
        nim_max_pile_size=25,
        josephus_min_people=4,
        josephus_max_people=20,
        josephus_min_k=2,
        josephus_max_k=5,
        verbose_prompts=True,
    ),
}


def make_messages(prompt, message_style: str = 'default'):
    if message_style == 'qwen':
        return [{'role': 'system', 'content': system_prompt_qwen}, {'role': 'user', 'content': prompt + "\n" + system_prompt}]
    if message_style == 'default':
        return [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}]
    raise ValueError(f'Unknown message style: {message_style}')

# ------------------------------------------------------------------------------------------------
# Countdown Task
def make_countdown_example(
    min_operands: int = 3,
    max_operands: int = 3,
    max_target: int = 1000,
    max_number: int = 100,
    operations: list[str] = ['+', '-', '*', '/'],
    message_style: str = 'default',
):
    """Generate example for countdown task.

    The goal of this task is to combine the provided operands using basic arithmetic operations to create
    an equation that equals the target. For example, given numbers [2,3,4] and target 14, a valid solution
    would be "2 * 3 + 4".

    Returns:
        Tuple containing (messages, target, numbers) where:
            messages (list): List of conversation messages
            eval_fn (Callable): Function that checks a solution for correctness
    """
    while True:
        num_operands = randint(min_operands, max_operands)
        numbers = [randint(1, max_number) for _ in range(num_operands)]
        target = numbers[0]

        for num in numbers[1:]:
            op = choice(operations)
            if op == '/' and num != 0 and target % num == 0:
                target //= num
            else:
                # use the operation directly, or fallback to multiplication for invalid division
                target = eval(f"{target}{'+' if op == '+' else '-' if op == '-' else '*'}{num}")

        if 1 <= target <= max_target:
            prompt = (
                f"Create an equation using only the numbers {numbers} that equals {target}. "
                "Use each number once with +, -, *, or /. Do not include an equals sign in the answer."
            )
            messages = make_messages(prompt, message_style)
            # create a partial function so we don't need to pass around task details
            eval_fn = partial(evaluate_countdown_solution, target=target, numbers=numbers)
            return messages, eval_fn

def evaluate_countdown_solution(solution: str, target: int, numbers: list[int]) -> bool:
    try:
        solution_numbers = [int(num) for num in re.findall(r'\d+', solution)]
    except ValueError:
        return False
    if sorted(solution_numbers) != sorted(numbers):
        return False
    try:
        return eval(solution) == target
    except Exception:
        return False

# ------------------------------------------------------------------------------------------------
# Linear Equation Task
def make_linear_equation_example(
    max_coefficient: int = 50,
    max_constant: int = 50,
    message_style: str = 'default',
    verbose_prompt: bool = False,
):
    """Generate example for linear equation solving task.

    Creates an equation like "2x + 3 = 15" or "3x - 7 = 20" and asks for the value of x.
    The equations are generated to always have integer solutions.
    """
    while True:
        # Start with the solution we want (to ensure integer solutions)
        x_value = randint(-10, 10)
        coefficient = randint(1, max_coefficient)
        constant = randint(-max_constant, max_constant)
        if constant == 0:
            continue  # avoid unnatural equations like "5x + 0 = 10"

        # Calculate the right side of the equation
        right_side = coefficient * x_value + constant

        # Create the equation string
        left_side = f"{coefficient}x{' + ' if constant >= 0 else ' - '}{abs(constant)}"
        equation = f"{left_side} = {right_side}"

        if verbose_prompt:
            prompt = (
                f"Find the value of x that makes this equation true: {equation}\n"
                "Provide only the numerical value for x without any other text or symbols."
            )
        else:
            prompt = f"Solve for x in the equation: {equation}\nAnswer with just the number (the value of x)."

        messages = make_messages(prompt, message_style)
        eval_fn = partial(evaluate_linear_equation_solution, x_value=x_value)
        return messages, eval_fn

def evaluate_linear_equation_solution(solution: str, x_value: int) -> bool:
    try:
        # Convert the solution to a number and compare with expected value
        return float(solution.strip()) == x_value
    except (ValueError, TypeError):
        return False

# ------------------------------------------------------------------------------------------------
# Single-pile Nim
def make_single_pile_nim_example(max_pile_size: int = 50, max_pickup: int = 3, message_style: str = 'default'):
    pile = randint(max_pickup + 2, max_pile_size)
    pickup = randint(2, max_pickup)

    # For single pile Nim with max pickup limit, optimal strategy is to leave
    # (pickup + 1) stones or its multiples to force opponent into bad position
    optimal_move = min(pickup, pile % (pickup + 1) or pickup)
    if optimal_move == 0:
        optimal_move = min(pickup, pile)  # Take what we can if no optimal move

    prompt = f"In Nim with {pile} stones, taking between 1 and {pickup} per turn, what is the optimal number of stones you should take when moving first? Answer with a number."
    messages = make_messages(prompt, message_style)
    eval_fn = partial(evaluate_nim_solution, optimal_move=optimal_move)
    return messages, eval_fn

def evaluate_nim_solution(solution, optimal_move):
    return str(optimal_move) == solution.strip()

# ------------------------------------------------------------------------------------------------
# Josephus Problem
def make_josephus_example(
    min_people: int = 5,
    max_people: int = 20,
    min_k: int = 2,
    max_k: int = 7,
    message_style: str = 'default',
    verbose_prompt: bool = False,
):
    # Randomly choose number of people and the step size
    n = randint(min_people, max_people)
    k = randint(min_k, max_k)
    suffix = "th" if k in [0,4,5,6,7,8,9,10] or k > 10 else {1:"st", 2:"nd", 3:"rd"}[k]

    def josephus(n: int, k: int) -> int:
        safe_pos = 0  # for 1 person, the safe position (0-indexed) is 0
        for i in range(2, n + 1):
            safe_pos = (safe_pos + k) % i
        return safe_pos + 1  # convert to 1-indexed
    safe_position = josephus(n, k)

    if verbose_prompt:
        prompt = (
            f"In a circle of {n} people numbered 1 through {n}, people are eliminated in a specific pattern. "
            f"Starting from position 1 and counting clockwise, every {k}{suffix} person is eliminated. "
            f"This continues around the circle until only one person survives. "
            f"Which position (1 through {n}) should you stand in to be the last survivor? "
            f"Answer with just the number of your chosen position."
        )
    else:
        prompt = (
            f"Josephus problem: In a circle of {n} people numbered 1 through {n}, every {k}{suffix} person "
            f"is eliminated in order until only one person remains. Which position should you stand in to "
            f"be the last remaining person? Answer with a number."
        )

    messages = make_messages(prompt, message_style)
    eval_fn = partial(evaluate_josephus_solution, safe_position=safe_position)

    return messages, eval_fn

def evaluate_josephus_solution(solution: str, safe_position: int) -> bool:
    return str(safe_position) == solution.strip()


# ------------------------------------------------------------------------------------------------
# Dataset
def gen_dataset(tasks: str='all', task_preset: str = 'default'):
    """Generate an infinite stream of task examples.

    Args:
        tasks: Optional task or comma-separated list of tasks to use. If "all", uses all available tasks.
            Other options include "countdown", "nim", and "josephus".
        task_preset: Prompt style and difficulty preset to use when constructing examples.
    """
    if task_preset not in task_presets:
        raise ValueError(f'Unknown task preset: {task_preset}')
    preset = task_presets[task_preset]

    default_tasks = {
        'countdown': partial(make_countdown_example, message_style=preset.message_style),
        'nim': partial(
            make_single_pile_nim_example,
            max_pile_size=preset.nim_max_pile_size,
            message_style=preset.message_style,
        ),
        'josephus': partial(
            make_josephus_example,
            min_people=preset.josephus_min_people,
            max_people=preset.josephus_max_people,
            min_k=preset.josephus_min_k,
            max_k=preset.josephus_max_k,
            message_style=preset.message_style,
            verbose_prompt=preset.verbose_prompts,
        ),
        'linear_equations': partial(
            make_linear_equation_example,
            max_coefficient=preset.linear_max_coefficient,
            max_constant=preset.linear_max_constant,
            message_style=preset.message_style,
            verbose_prompt=preset.verbose_prompts,
        ),
    }

    if tasks == "all":
        task_functions = list(default_tasks.values())
    elif isinstance(tasks, list):
        task_functions = [task for name, task in default_tasks.items() if name in tasks]
    elif isinstance(tasks, str):
        if ',' in tasks:
            task_list = [t.strip() for t in tasks.split(',')]
            task_functions = [task for name, task in default_tasks.items() if name in task_list]
        else:
            task_functions = [default_tasks[tasks]]
    else:
        raise ValueError

    while True:
        task_fn = choice(task_functions)
        messages, eval_fn = task_fn()
        yield messages, eval_fn
