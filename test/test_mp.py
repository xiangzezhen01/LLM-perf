import multiprocessing
import time
import os
import random
import traceback

def worker_success(task_id):
    """正常工作的子进程"""
    print(f"子进程 {task_id} (PID:{os.getpid()}) 开始工作")
    time.sleep(random.uniform(0.5, 1.5))
    print(f"子进程 {task_id} (PID:{os.getpid()}) 完成工作")
    return task_id * 10

def worker_error(task_id):
    """会抛出异常的子进程"""
    print(f"子进程 {task_id} (PID:{os.getpid()}) 开始工作")
    time.sleep(random.uniform(0.5, 1.0))
    
    # 模拟不同类型的错误
    if task_id % 3 == 0:
        raise ValueError(f"子进程 {task_id} 值错误")
    elif task_id % 3 == 1:
        # 除以零错误
        result = 10 / (task_id - task_id)
    else:
        # 导入错误
        import non_existent_module
    
    return task_id * 10

def safe_worker(task_id):
    """带有错误处理的子进程"""
    try:
        print(f"子进程 {task_id} (PID:{os.getpid()}) 开始工作")
        time.sleep(random.uniform(0.5, 1.0))
        
        # 随机决定是否抛出错误
        if random.random() < 0.3:  # 30%几率出错
            raise RuntimeError(f"子进程 {task_id} 随机错误")
        
        print(f"子进程 {task_id} (PID:{os.getpid()}) 完成工作")
        return task_id * 10
    except Exception as e:
        print(f"子进程 {task_id} (PID:{os.getpid()}) 捕获到错误: {str(e)}")
        # 记录错误但不传播到主进程
        return f"ERROR: {str(e)}"

def run_test(scenario):
    """运行不同场景的测试"""
    print(f"\n{'='*50}")
    print(f"测试场景: {scenario}")
    print(f"{'='*50}")
    
    tasks = list(range(1, 7))  # 6个任务
    
    try:
        with multiprocessing.Pool(processes=3) as pool:
            if scenario == 1:
                print("场景1: 所有子进程正常执行")
                results = pool.map(worker_success, tasks)
            elif scenario == 2:
                print("场景2: 部分子进程抛出未处理错误")
                results = pool.map(worker_error, tasks)
            elif scenario == 3:
                print("场景3: 子进程处理自身错误")
                results = pool.map(safe_worker, tasks)
            elif scenario == 4:
                print("场景4: 使用imap处理错误")
                results = []
                for res in pool.imap(safe_worker, tasks):
                    results.append(res)
            
            print("\n所有任务完成，结果:")
            for i, res in enumerate(results, 1):
                print(f"任务 {i}: {res}")
                
    except Exception as e:
        print(f"\n主进程捕获到异常: {str(e)}")
        print("错误跟踪:")
        traceback.print_exc()
    finally:
        print("测试结束")

if __name__ == '__main__':
    print("多进程错误处理验证")
    print("="*50)
    
    # 运行不同测试场景
    run_test(1)  # 所有成功
    run_test(2)  # 有错误未处理
    run_test(3)  # 子进程自己处理错误
    run_test(4)  # 使用imap处理错误