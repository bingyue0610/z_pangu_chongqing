import multiprocessing
import os, time, random


def cal(some_list, num):
    start_time = time.time()
    tmp_max = num
    for i in some_list:
        if i > tmp_max:
            tmp_max = i
    end_time = time.time()
    print(tmp_max)
    print(end_time - start_time)
    return tmp_max, 10

def read(some_list, num):
    print('fuck')

def get_cpu_num():
    return multiprocessing.cpu_count()

if __name__=='__main__':
    L = [x*x for x in range(1000000)]
    #
    # one_process_test = cal(L, 0)
    # # 999998000001
    # # 0.05000019073486328

    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cores)
    pool_list = []
    result_list = []
    for i in range(4):
        print('this is the %s th term', i)
        if i < 3:
            pool_list.append(pool.apply_async(cal, args=(L[int(len(L)*i/4.0): int(len(L)*(i+1)/4.0)], 0)))
        else:
            pool_list.append(pool.apply_async(cal, args=(L[int(len(L)*i/4.0):], 0)))
    result_list = [xx.get() for xx in pool_list]
    print(type(pool_list))
    print('pool list', pool_list)
    print('result_list', result_list)
    print(L[-1])
    pool.close()
    pool.join()
