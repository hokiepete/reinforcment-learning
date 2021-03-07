import itertools
import threading
import time 
import multiprocessing
import numpy as np

class Worker:
    def __init__(self, id_, global_counter):
        self.id = id_
        self.global_counter = global_counter
        self.local_counter = itertools.count()

    def run(self):
        while True:
            time.sleep(np.random.rand()*2)
            global_step = next(self.global_counter)
            local_step = next(self.local_counter)
            print(f"Worker({self.id}): {local_step}")
            if global_step >= 20:
                break


global_counter = itertools.count()
NUM_WORKERS = multiprocessing.cpu_count()
print("Build Workers")
workers = []
for worker_id in range(NUM_WORKERS):
    workers.append(
        Worker(worker_id, global_counter)
    )
print("Gen Threads")
worker_threads = []
for worker in workers:
    t = threading.Thread(
        target=lambda: worker.run()
    )
    t.start()
    worker_threads.append(t)
print("JOIN")
for t in worker_threads:
    t.join()

print("DONE")
    