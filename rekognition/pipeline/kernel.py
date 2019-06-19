import multiprocessing
from abc import ABC, abstractmethod

class Kernel(ABC):
    def __init__(self, run_as_process=True):
        self._run_as_process = run_as_process

    # Pure virtual function
    def run(self, *args):
        results = []
        parent_conn, child_conn = multiprocessing.Pipe()

        if self._run_as_process:
            process = multiprocessing.Process(target=self.predict, args = (parent_conn, ) + args)

            # Make main process completely close child processes
            process.daemon = True

            process.start()
            results = child_conn.recv()
            process.join()
        else:
            self.predict(parent_conn, *args)
            results = child_conn.recv()

        return results

    @abstractmethod
    def predict(self, *args):
        pass