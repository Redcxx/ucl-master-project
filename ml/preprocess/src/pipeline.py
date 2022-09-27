import functools
from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool

from tqdm import tqdm


# run the given function
# this exists because multiprocessing can only pickle top level function
def process(func):
    return func()


# this is the actual processing function
def _process(extra_args, functions, early_termination):
    prev_output = None
    for func, extra_arg in zip(functions, extra_args):
        prev_output = func(prev_output, extra_arg)
        if prev_output is None:
            if early_termination is not None and func != functions[-1]:
                early_termination(extra_args)
            return None
    return prev_output


class Pipeline:

    def __init__(self, workers=4, multi_process=True):
        self.functions = []
        self.arguments = []
        self.pool = ProcessPool if multi_process else ThreadPool
        self.workers = workers
        self.post_process_func = None
        self.early_termination = None

    def add(self, func, args=None):
        self.functions.append(func)
        self.arguments.append(args)
        return self

    def on_early_terminate(self, func):
        self.early_termination = func
        return self

    def post_process(self, func):
        self.post_process_func = func
        return self

    def get_worker_inputs(self):
        assert isinstance(self.arguments[0], list), "input to the first function should be a list"
        size = len(self.arguments[0])
        worker_inputs = [[] for _ in range(size)]
        for func_index, args in enumerate(self.arguments):
            # args is args for each function
            if not isinstance(args, list):
                args = [args for _ in range(size)]

            assert len(args) >= size, f"len of arguments need to be enough: {len(args)} < {size}"
            for i, arg in enumerate(args[:size]):
                worker_inputs[i].append(arg)

        worker_funcs = [functools.partial(_process, worker_input, self.functions, self.early_termination) for worker_input in worker_inputs]

        return worker_funcs

    def run(self, desc=''):
        process_outs = []
        worker_inputs = self.get_worker_inputs()
        with self.pool(self.workers) as pool:
            for item in tqdm(
                    pool.imap_unordered(process, worker_inputs),
                    total=len(worker_inputs),
                    desc=desc
            ):
                process_outs.append(item)
        if self.post_process_func is not None:
            self.post_process_func(process_outs)
