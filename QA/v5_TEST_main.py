import sys
from multiprocessing import Process
import v5_hash_TEST
import v5_ip_TEST
import v5_hostname_TEST
import v5_url_TEST
import datetime
from apscheduler.schedulers.background import BackgroundScheduler
import time
from datetime import datetime

def call_process(API_KEY, users):
    procs = []
    func_list = [v5_hash_TEST.hash_test,v5_ip_TEST.ip_test,v5_hostname_TEST.hostname_test,v5_url_TEST.url_test]
    print("execute_process")
    for v in func_list:
        for i in range(users):
            proc = Process(target=v, args=(API_KEY,))
            procs.append(proc)
    for v in procs:
        v.start()

    for v in procs:
        v.join()

if __name__ == "__main__":
    API_KEY = sys.argv[1]
    users = int(sys.argv[2])
    period_time = int(sys.argv[3])
    try:
        start_time = datetime.now()
        loop_count = int(sys.argv[4]) * period_time
        end_time = datetime.fromtimestamp(start_time.timestamp()+float(loop_count))
    except:
        print("require endtime")

    sched = BackgroundScheduler()
    sched.start()

    sched.add_job(call_process, 'interval', seconds=period_time, args=[API_KEY, users], start_date=start_time, end_date=end_time, max_instances=1000)
    print("running...")

    try:
        while True:
            # import pdb; pdb.set_trace()
            if not sched.get_jobs():
                sched.remove_all_jobs()
                sched.shutdown()
                break
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        sched.shutdown()
