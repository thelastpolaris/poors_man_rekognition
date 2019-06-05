import multiprocessing
from abc import ABC, abstractmethod

class Kernel(ABC):
    def __init__(self):
        pass

    # Pure virtual function
    def run(self, *args):
        parent_conn, child_conn = multiprocessing.Pipe()
        process = multiprocessing.Process(target=self.predict, args = (parent_conn, ) + args)

        # Make main process completely close child processes
        process.daemon = True

        process.start()
        results = child_conn.recv()
        process.join()

        return results

    @abstractmethod
    def predict(self, *args):
        pass