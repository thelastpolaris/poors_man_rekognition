import multiprocessing
import abc

class Kernel():
    def __init__(self):
        pass

    # Pure virtual function
    def run(self, args):
        parent_conn, child_conn = multiprocessing.Pipe()
        process = multiprocessing.Process(target=self.predict, args = (parent_conn, ) + args)

        process.start()
        results = child_conn.recv()
        process.join()

        return results

    @abc.abstractmethod
    def predict(self, args):
        pass