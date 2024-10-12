import argparse
import json
import time
import subprocess
import sys
import os
import yaml
from shutil import copyfile
from datetime import date
#from knockknock import slack_sender

import pymongo

myclient = pymongo.MongoClient(
    "mongodb://dongshengy:dongshen20230809@nfs.yangdsh.lrbplus-pg0.clemson.cloudlab.us/dongshengyDB?authSource=admin")
mydb = myclient["dongshengyDB"]
env = "cloudlab"
if env == "cloudlab":
    root = "/proj/lrbplus-PG0/workspaces/yangdsh/CacheLib_heuristic/"
    cachebench_loc_ = "/proj/lrbplus-PG0/workspaces/yangdsh/CacheLib_heuristic/build-cachelib/cachebench/cachebench"
    cachebench_loc = "/proj/lrbplus-PG0/workspaces/yangdsh/CacheLib_heuristic/build-cachelib/cachebench/cachebench_"
    os.system(f"cp {cachebench_loc_} {cachebench_loc}")
    temp_dir = "/proj/lrbplus-PG0/workspaces/yangdsh/CacheLib_heuristic/build-cachelib/cachebench"
elif env == "cloudlab2":
    root = "/proj/lrbplus-PG0/workspaces/yangdsh/cachelib-sosp23/"
    cachebench_loc_ = "/proj/lrbplus-PG0/workspaces/yangdsh/cachelib-sosp23/build-cachelib/cachebench/cachebench"
    cachebench_loc = "/proj/lrbplus-PG0/workspaces/yangdsh/cachelib-sosp23/build-cachelib/cachebench/cachebench_"
    os.system(f"cp {cachebench_loc_} {cachebench_loc}")
    temp_dir = "/proj/lrbplus-PG0/workspaces/yangdsh/cachelib-sosp23/build-cachelib/cachebench"

ts = int(time.time())
debug_nfs = 0
upload_mode = False
should_upload = True
sharding_mode = False
multi_mode = -1
n_cores = 16
if upload_mode:
    # nsdi_cachelib_stress_replay_full_sleep3: 16 instances, 16 threads, sleep 87us
    # nsdi_cachelib_stress_replay_full_sleep4: 8 instances, 24 threads, sleep 87us
    # nsdi_cachelib_stress_replay_full_spin3: 8 instances, 16 threads, sleep 87us
    #ts = "1680477513"
    #ts = "1680478435" # 2393270
    #ts = "1680481441" #pin 32 lru
    #ts = "1680481369" # 1824375
    #ts = "1680484405" # pin 64 lru 3355988
    #ts = "1680484419" # 2639940
    #ts = "1680494473" # 64 lru 3646487
    #ts = "1680481369" # 2941604
    # 1680631155, 32xxxxxx
    # 1680631125, 29xxxxxx
    ts = "1708534403"


def to_task_config(task, task_id):
    with open(task['json_test_config']) as f:
        config = json.load(f)
        config['cache_config']['cacheSizeMB'] = task['cache_size']
        for k in ('useEvictionControl', 'MLConfig', 'allocator', 'lruRefreshSec',
                  'rebalanceStrategy', 'poolRebalanceIntervalSec', 'allocFactor', 'lruRefreshRatio'):
            if k in task:
                config['cache_config'][k] = task[k]
        for k in ('numOps', 'numThreads', "wallTimeReplaySpeed", "cacheSetLatency"
                  , 'mlReqUs', 'admissionThreshold', 'fixedSize'):
            if k in task:
                config['test_config'][k] = task[k]
        if 'ampFactor' in task:
            config['test_config']['replayGeneratorConfig']['ampFactor'] = task['ampFactor']
            config['cache_config']['MLConfig'] += ',time_unit,' + str(task['ampFactor'])
        #config['test_config']['traceFileName'] = config['test_config']['traceFileName_' + env]
        if sharding_mode:
            config['cache_config']['cacheSizeMB'] = str(int(config['cache_config']['cacheSizeMB']) // 56)
            config['test_config']['traceFileName'] += '.' + str(task_id)
        if multi_mode != 0 and 'printToFile' in task:
            config['test_config']['printToFile'] = True #(task['printToFile'] and task_id < 16)
            if 'logFileName' in task and len(task['logFileName']) > 1:
                config['test_config']['logFileName'] = task['logFileName'] + str(ts) + '.' + str(task_id)
            else:
                config['test_config']['logFileName'] = ''
        if 'nvmCacheSizeMB' in config['cache_config'] and config['cache_config']['nvmCacheSizeMB'] != 0:
            if multi_mode < 0:
                device = '/dev/nvme0n1p4'
            else:
                device = '/dev/nvme1n1'
            config['cache_config']['nvmCachePaths'] = [device] #['/dev/nvme1n1', '/dev/nvme0n1p4']
        config['test_config']['numOps'] = int(
            config['test_config']['numOps'] / config['test_config']['numThreads'])
        fout = open(f'{temp_dir}/{ts}/{task_id}.json', 'w')
        json.dump(config, fout)
        task['json_test_config'] = f'{temp_dir}/{ts}/{task_id}.json'
        task['progress_stats_file'] = f'{temp_dir}/{ts}/{task_id}.stat'
    return task


def to_task_str(task, task_id):
    """
    split deterministic args and nodeterminstics args. Add _ prefix to later
    """
    task = to_task_config(task, task_id)
    params = {}
    for k, v in task.items():
        if k in ['progress_stats_file', 'json_test_config'] and v is not None:
            params[k] = str(v)

    # use timestamp as task id
    params = [f'--{k}={v}'for k, v in params.items()]
    params = ' '.join(params)
    res = f'{cachebench_loc} {params}'
    return res


def parse_cmd_args():
    # how to schedule parallel simulations
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',
                        help='debug mode only run 1 task locally',
                        type=bool,
                        choices=[True, False])
    parser.add_argument('--job_file', type=str, nargs='?', help='job config file', required=True)
    parser.add_argument('--algorithm_param_file', type=str, help='algorithm parameter config file', required=True)
    parser.add_argument('--trace_param_file', type=str, help='trace parameter config file', required=True)
    args = parser.parse_args()
    args_dict = vars(args)
    if upload_mode:
        args_dict['job_file'] = f'{temp_dir}/{ts}/jobs.yaml'
        args_dict['algorithm_param_file'] = f'{temp_dir}/{ts}/algo.yaml'
        args_dict['trace_param_file'] = f'{temp_dir}/{ts}/trace.yaml'

    return args_dict

def cartesian_product(param: dict):
    worklist = [param]
    res = []

    while len(worklist):
        p = worklist.pop()
        split = False
        for k in p:
            if type(p[k]) == list:
                _p = {_k: _v for _k, _v in p.items() if _k != k}
                for v in p[k]:
                    worklist.append({
                        **_p,
                        k: v,
                    })
                split = True
                break

        if not split:
            res.append(p)

    return res

def get_task(args):
    """
    convert job config to list of task
    @:returns dict/[dict]
    """
    # job config file
    with open(args['job_file']) as f:
        file_params = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in file_params.items():
        if args.get(k) is None:
            args[k] = v

    # load algorithm parameters
    assert args.get('algorithm_param_file') is not None
    with open(args['algorithm_param_file']) as f:
        default_algorithm_params = yaml.load(f, Loader=yaml.FullLoader)

    assert args.get('trace_param_file') is not None
    with open(args['trace_param_file']) as f:
        trace_params = yaml.load(f, Loader=yaml.FullLoader)
    tasks = []
    for trace_file in args['trace_files']:
        for cache_type in args['cache_types']:
            for cache_size_or_size_parameters in trace_params[trace_file]['cache_sizes']:
                # element can be k: v or k: list[v], which would be expanded with cartesian product
                # priority: default < per trace < per trace per algorithm < per trace per algorithm per cache size
                parameters = {}
                if cache_type in default_algorithm_params:
                    parameters = {**parameters, **default_algorithm_params[cache_type]}
                per_trace_params = {}
                for k, v in trace_params[trace_file].items():
                    if k not in ['cache_sizes'] and k not in default_algorithm_params and v is not None:
                        per_trace_params[k] = v
                parameters = {**parameters, **per_trace_params}
                if cache_type in trace_params[trace_file]:
                    # trace parameters overwrite default parameters
                    parameters = {**parameters, **trace_params[trace_file][cache_type]}
                if isinstance(cache_size_or_size_parameters, dict):
                    # only 1 key (single cache size) is allowed
                    assert (len(cache_size_or_size_parameters) == 1)
                    cache_size = list(cache_size_or_size_parameters.keys())[0]
                    if cache_type in cache_size_or_size_parameters[cache_size]:
                        # per cache size parameters overwrite other parameters
                        parameters = {**parameters, **cache_size_or_size_parameters[cache_size][cache_type]}
                else:
                    cache_size = cache_size_or_size_parameters
                parameters_list = cartesian_product(parameters)
                for parameters in parameters_list:
                    task = {
                        'trace_file': trace_file,
                        'cache_type': cache_type,
                        'cache_size': cache_size,
                        **parameters,
                    }
                    for k, v in args.items():
                        if k not in [
                            'cache_types',
                            'trace_files',
                            'algorithm_param_file',
                            'trace_param_file',
                            'job_file',
                            'debug',
                            'nodes',
                        ] and v is not None:
                            task[k] = v
                    if multi_mode > 1:
                        for i in range(multi_mode):
                            tasks.append(task)
                    else:
                        tasks.append(task)
    return tasks


#webhook_url = "https://hooks.slack.com/services/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"
#@slack_sender(webhook_url=webhook_url, channel="Dongsheng Yang")
def run(args: dict, tasks: list):
    # debug mode, only 1 task
    if args["debug"]:
        tasks = tasks[:1]
    if not os.path.exists(f'{temp_dir}/{ts}/'):
        os.mkdir(f'{temp_dir}/{ts}/')
        os.system(f"cp -r {root}/cachelib/allocator/ {temp_dir}/{ts}/")
    if not upload_mode:
        copyfile(args['job_file'], f'{temp_dir}/{ts}/jobs.yaml')
        copyfile(args['algorithm_param_file'], f'{temp_dir}/{ts}/algo.yaml')
        copyfile(args['trace_param_file'], f'{temp_dir}/{ts}/trace.yaml')

    print(f'n_task: {len(tasks)}\n '
          f'generating job file to {temp_dir}/{ts}/job')
    with open(f'{temp_dir}/{ts}/job', 'w') as f:
        for i, task in enumerate(tasks):
            if debug_nfs == 1:
                task_str = "[ -d \"/nfs/fb/\" ] && sleep 1"
            elif debug_nfs:
                #multi_mode = 0
                task_str = f"sudo bash {root}cachelib/cachebench/script/fix_host.sh"
                #task_str = "sudo bash /proj/lrbplus-PG0/workspaces/yangdsh/fetch_file.sh"
            else:
                task_str = to_task_str(task, i)
            if multi_mode == 1:
                task_str = f'taskset -c 16-31 {task_str}'
            if multi_mode == -1 or multi_mode == 0:
                task_str = f'taskset -c 0-{n_cores-1} {task_str}'
            if multi_mode > 1:
                j = i # % multi_mode
                if multi_mode == 64:
                    task_str = f'taskset -c {j} {task_str}'
                elif multi_mode > 8:
                    task_str = f'taskset -c {j*4}-{j*4+3} {task_str}'
                elif multi_mode <= 8:
                    task_str = f'sudo blkdiscard {device} && taskset -c {j*8}-{j*8+7} {task_str}'
            env_str = f'LD_LIBRARY_PATH="{root}opt/cachelib/lib" && export LD_LIBRARY_PATH'
            task_str = f'bash --login -c "{env_str} && {task_str}" &> {temp_dir}/{ts}/{i}.log\n'
            if i == 0:
                print(f'first task: {task_str}')
            f.write(task_str)
    with open(f'{temp_dir}/{ts}/job') as f:
        command = ['parallel', '-v', '--eta', '--shuf', '--keep-order', '--sshdelay', '0.1']
        for n in args['nodes']:
            command.extend(['-S', n])
        print(f"{' '.join(command)} < {temp_dir}/{ts}/job")
        subprocess.run(command,
                       stdin=f)

def get_multi_results(tasks, timestamp):
    throughput = 0
    set_throughput = 0
    for i in range(len(tasks)):
        with open(f'{temp_dir}/{timestamp}/{i}.log') as f:
            print(f'{temp_dir}/{timestamp}/{i}.log')
            lines = f.readlines()
            for line in lines:
                if line.startswith('get       :'):
                    line = line.replace(' ', '').replace('%', '')
                    hit_ratio = float(line.split(':')[-1])
                    throughput += float(line.replace(',', '').split(':')[1].split('/')[0])
                if line.startswith('set       :'):
                    line = line.replace(' ', '').replace('%', '')
                    hit_ratio = float(line.split(':')[-1])
                    set_throughput += float(line.replace(',', '').split(':')[1].split('/')[0])
                if line.startswith('Cache Request API Latency avg'):
                    result_dict['avg latency'] = float(line.split()[-2])
                if line.startswith('Cache Request API Latency p50      :'):
                    result_dict['p50 latency'] = float(line.split()[-2])
                if line.startswith('Cache Request API Latency p90      :'):
                    result_dict['p90 latency'] = float(line.split()[-2])
                if line.startswith('Cache Request API Latency p99      :'):
                    result_dict['p99 latency'] = float(line.split()[-2])
                if line.startswith('Cache Request API Latency p999     :'):
                    result_dict['p999 latency'] = float(line.split()[-2])
    print(throughput, set_throughput)

def upload_results(tasks, timestamp):
    mycol = mydb[tasks[0]['dbcollection']]
    today = str(date.today())
    for i in range(len(tasks)):
        with open(f'{temp_dir}/{timestamp}/{i}.json') as f:
            result_dict = json.load(f)['cache_config']
            result_dict['date'] = today
            # test_config:
            result_dict['trace_file'] = tasks[i]['trace_file']
            result_dict['numOps'] = tasks[i]['numOps']
            if 'mlReqUs' in tasks[i]:
                result_dict['mlReqUs'] = tasks[i]['mlReqUs']
            if 'wallTimeReplaySpeed' in tasks[i]:
                result_dict['wallTimeReplaySpeed'] = tasks[i]['wallTimeReplaySpeed']
            if 'cacheSetLatency' in tasks[i]:
                result_dict["cacheSetLatency"] = tasks[i]['cacheSetLatency']
            result_dict['cache_type'] = tasks[i]['cache_type']
            if 'numThreads' in tasks[i]:
                result_dict['numThreads'] = tasks[i]['numThreads']
            if 'admissionThreshold' in tasks[i]:
                result_dict['admissionThreshold'] = tasks[i]['admissionThreshold']
            if 'fixedSize' in tasks[i]:
                result_dict['fixedSize'] = tasks[i]['fixedSize']
            if 'ampFactor' in tasks[i]:
                result_dict['ampFactor'] = tasks[i]['ampFactor']
        '''with open(f'{temp_dir}/{timestamp}/{i}.stat') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('Total Ops'):
                    pass
                if line.startswith('get'):
                    pass'''
        if multi_mode > 1:
            result_dict['timestamp_id'] = i
        result_dict['timestamp'] = timestamp
        result_dict['cpu_list'] = []
        with open(f'{temp_dir}/{timestamp}/{i}.json') as f:
            conf = json.load(f)['test_config']
            if 'logFileName' in conf and multi_mode != 0:
                logloc = conf['logFileName']
                with open(f'{logloc}.top') as fin:
                    print(f'{logloc}.top')
                    lines = fin.readlines()
                    for line in lines:
                        if line.endswith('cachebench\n'): #cachebench
                            cpu = float(line.split()[-4])
                            result_dict['cpu_list'].append(cpu)
        with open(f'{temp_dir}/{timestamp}/{i}.log') as f:
            print(f'{temp_dir}/{timestamp}/{i}.log')
            #if i == 43:
            #    continue
            lines = f.readlines()
            result_dict['hit_ratio_list'] = []
            result_dict['byte_requested_list'] = []
            result_dict['byte_miss_list'] = []
            result_dict['ram_hit_ratio_list'] = []
            result_dict['nvm_hit_ratio_list'] = []
            result_dict['ops_list'] = []
            result_dict['time_list'] = []
            result_dict['throughput_list'] = []
            result_dict['target_throughput_list'] = []
            result_dict['miss_ratio_list'] = []
            result_dict['latency_list'] = []
            result_dict['wall_ops_list'] = []
            result_dict['wall_hit_ratio'] = []
            result_dict['peak sec'] = []
            result_dict['peak sec p50 latency'] = []
            result_dict['peak sec p90 latency'] = []
            result_dict['peak sec p95 latency'] = []
            result_dict['peak sec p99 latency'] = []
            for line in lines:
                if 'ml on: ' in line:
                    result_dict['ml_on'] = line
                    continue
                if 'ml off: ' in line:
                    result_dict['ml_off'] = line
                    continue
                if 'ml evict: ' in line:
                    result_dict['ml_evict'] = line
                    continue
                if 'sec miss' in line:
                    result_dict['second_miss_list'] = line
                    print(line)
                    continue
                if 'sec lat' in line:
                    result_dict['second_latency_list'] = line
                    continue
                if 'sec cnt' in line:
                    result_dict['second_req_cnt'] = line
                    continue
                if 'peak latency: ' in line:
                    result_dict['peak_latency_list'] = line
                    continue
                if 'sec single lat' in line:
                    result_dict['second_single_latency_list'] = line
                    continue
                if 'cache latency percentile: ' in line :
                    result_dict['cache latency percentile'] = line
                    continue
                if 'trace latency percentile: ' in line :
                    result_dict['trace latency percentile'] = line
                    continue
                #if 'completed' in line:
                #    v = float(line.split()[1].split('M')[0])
                #    result_dict['wall_ops_list'].append(v)
                #    v = float(line.split()[-1].split('%')[0])
                #    result_dict['wall_hit_ratio'].append(v)
                if line.startswith('throughput'):
                    v = float(line.split(' ')[-1])
                    result_dict['throughput_list'].append(v)
                if line.startswith('target throughput'):
                    v = float(line.split(' ')[-1])
                    result_dict['target_throughput_list'].append(v)
                if line.startswith('miss ratio'):
                    v = float(line.split(' ')[-1])
                    result_dict['miss_ratio_list'].append(v)
                if line.startswith('latency'):
                    v = float(line.split(' ')[-1])
                    result_dict['latency_list'].append(v)
                if line.startswith('Seq'):
                    print(line)
                    v = float(line.split(' ')[-1].split(':')[-1])
                    result_dict['time_list'].append(v)
                if line.startswith('RAM Evictions :'):
                    v = float(line.replace(',', '').split(' ')[-1])
                    result_dict['evictions'] = v
                if line.startswith('RAM Reinsertions'):
                    v = float(line.replace(',', '').split(' ')[-1])
                    result_dict['reinsertions'] = v
                if line.startswith('Byte Hit Ratio:'):
                    line = line.replace(' ', '').replace('%', '')
                    v = float(line.split(':')[-1])
                    result_dict['byte_hit_ratio'] = v
                '''if line.startswith('elpased time'):
                    line = line.replace(' ', '') 
                    v = float(line.split(':')[-1])
                    result_dict['time_list'].append(v)'''
                if 'ops completed. Hit Ratio' in line:
                    line = line.replace('%', '').replace(',', '').replace(')', '').replace('M', '')
                    if len(line.split()) == 11 and ':' not in line.split()[-3] and ':' not in line.split()[-1]\
                    and ':' not in line.split()[1]:
                        v = float(line.split()[-3])
                        result_dict['ram_hit_ratio_list'].append(v)
                        v = float(line.split()[-1])
                        result_dict['nvm_hit_ratio_list'].append(v)
                        v = float(line.split()[1])
                        result_dict['ops_list'].append(v)
                if line.startswith('set       :'):
                    line = line.replace(' ', '').replace('%', '')
                    hit_ratio = float(line.split(':')[-1])
                    result_dict['throughput_of_set'] = float(line.replace(',', '').split(':')[1].split('/')[0])
                if line.startswith('get       :'):
                    line = line.replace(' ', '').replace('%', '')
                    hit_ratio = float(line.split(':')[-1])
                    result_dict['hit_ratio'] = hit_ratio
                    result_dict['throughput'] = float(line.replace(',', '').split(':')[1].split('/')[0])
                if line.startswith('Cache Gets'):
                    line = line.replace(' ', '').replace(',', '')
                    v = float(line.split(':')[-1])
                    result_dict['total_gets'] = v
                if line.startswith('peak 50th: '):
                    if 'completed' not in line:
                        result_dict['peak p50 latency'] = float(line.split()[-1])
                    else:
                        result_dict['peak p50 latency'] = 0
                if line.startswith('peak 90th: '):
                    result_dict['peak p90 latency'] = float(line.split()[-1])
                if line.startswith('peak 95th: '):
                    result_dict['peak p95 latency'] = float(line.split()[-1])
                if line.startswith('peak 99th: '):
                    result_dict['peak p99 latency'] = float(line.split()[-1])
                #
                if line.startswith('peak sec 50th: '):
                    #result_dict['peak sec'].append(float(line.split().split(',')[0]))
                    result_dict['peak sec p50 latency'].append(float(line.split(',')[-1]))
                if line.startswith('peak sec 90th: '):
                    result_dict['peak sec p90 latency'].append(float(line.split(',')[-1]))
                if line.startswith('peak sec 95th: '):
                    result_dict['peak sec p95 latency'].append(float(line.split(',')[-1]))
                if line.startswith('peak sec 99th: '):
                    result_dict['peak sec p99 latency'].append(float(line.split(',')[-1]))
                #
                if line.startswith('Cache Find API latency avg '):
                    result_dict['avg latency'] = float(line.split()[-2])
                if line.startswith('Cache Find API latency p50 '):
                    result_dict['p50 latency'] = float(line.split()[-2])
                if line.startswith('Cache Find API latency p90 '):
                    result_dict['p90 latency'] = float(line.split()[-2])
                if line.startswith('Cache Find API latency p99 '):
                    result_dict['p99 latency'] = float(line.split()[-2])
                if line.startswith('Cache Find API latency p999 '):
                    result_dict['p999 latency'] = float(line.split()[-2])
            print(result_dict)
        if should_upload:
            mycol.insert_one(result_dict)


def main():
    args = parse_cmd_args()
    tasks = get_task(args)
    # print(tasks)
    if not upload_mode:
        run(args, tasks)
    if debug_nfs:
        return
    #if multi_mode:
    #    get_multi_results(tasks, ts)
    #else:
    upload_results(tasks, ts)
    print(ts)


if __name__ == '__main__':
    main()
