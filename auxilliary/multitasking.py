import threading

from auxilliary.utility_funcs import UtilityFuncs
from global_constants import GlobalConstants


class ThreadWorker(threading.Thread):
    def __init__(self, thread_id, task_list):
        threading.Thread.__init__(self)
        self.threadId = thread_id
        self.taskList = task_list
        self.resultList = []

    def run(self):
        pass


class MultiTaskRunner:
    @staticmethod
    def run_task(runner_type, tasks, **kwargs):
        results = []
        tasks_dict = UtilityFuncs.distribute_evenly_to_threads(num_of_threads=GlobalConstants.THREAD_COUNT,
                                                               list_to_distribute=tasks)
        threads_dict = {}
        for thread_id in range(GlobalConstants.THREAD_COUNT):
            threads_dict[thread_id] = runner_type(thread_id, tasks_dict[thread_id], **kwargs)
            threads_dict[thread_id].start()
        for thread in threads_dict.values():
            thread.join()
        for thread in threads_dict.values():
            results.extend(thread.resultList)
        return results
