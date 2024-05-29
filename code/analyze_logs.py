import os
import sys
import gc
import re
from pprint import pprint
from collections import namedtuple
from pathlib import Path

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

LOGS_DIR = Path("./logs/logs")
JOBLIST_PATH = Path("./logs/logs/joblist.txt")

log_line = namedtuple("log_line", ["JobID", "JobName", "State"])

def parse_joblist(path=JOBLIST_PATH):
    unfiltered_joblist = []
    with open(path, mode="r", encoding="utf8") as joblist_file:
        for i, line in enumerate(joblist_file.readlines()):
            if re.search("-", line):
                continue
            if re.search("JobID", line):
                continue
            job_id, job_name, state = line.split()
            unfiltered_joblist.append(log_line(job_id, job_name, state))
    filtered_joblist = set(filter(lambda ll: (ll.State!="FAILED" and ll.State!="OUT_OF_ME+") and re.search("unet", ll.JobName), unfiltered_joblist))
    return filtered_joblist

def parse_logs(filtered_joblist):
    fold_re = re.compile(r"Fold Number: (\d+)")
    epoch_re = re.compile(r"Epoch (\d+)/(\d+) \(Train\. Loss: ([\d.]+); +Time: (\d+)sec; Steps Completed: (\d+)\)")
    valid_perf_re = re.compile(r"Valid\. Performance \[Benign or Indolent PCa \(n=(\d+)\) +vs\. csPCa \(n=(\d+)\)\]:")
    ranking_score_re = re.compile(r"Ranking Score = ([\d.]+), +AP = -*([\d.]+), AUROC = ([\d.]+)")
    # lr_re = re.compile(r"Learning Rate Updated! New Value: ([\d.]+)")

    train_infos = {job_.JobName: [] for job_ in filtered_joblist}
    valid_infos = {job_.JobName: [] for job_ in filtered_joblist}
    for job in filtered_joblist:
        # print("ID ", job.JobID)
        log_path = LOGS_DIR / f"slurm-{job.JobID}.out"
        train_info = []
        valid_info = []
        try: 
            with open(log_path, mode="r", encoding="utf8") as log:
                for log_line in log.readlines():
                    global train_info_
                    train_info_ = {}
                    global valid_info_
                    valid_info_ = {}
                    if fold_re.search(log_line): 
                        global fold
                        fold = int(fold_re.search(log_line).group(1))
                    if epoch_re.search(log_line):
                        train_info_["Fold"] = fold
                        train_info_["Epoch"] = int(epoch_re.search(log_line).group(1))
                        global epoch
                        epoch = train_info_["Epoch"]
                        train_info_["Max Epoch"] = int(epoch_re.search(log_line).group(2))
                        train_info_["Train loss"] = float(epoch_re.search(log_line).group(3))
                        train_info_["Epoch Time"] = int(epoch_re.search(log_line).group(4))
                        train_info_["Steps"] = int(epoch_re.search(log_line).group(5))
                    if valid_perf_re.search(log_line):
                        global benign
                        benign = int(valid_perf_re.search(log_line).group(1))
                        global cspca
                        cspca = int(valid_perf_re.search(log_line).group(2))
                    
                    if ranking_score_re.search(log_line):
                        valid_info_["Fold"] = fold
                        valid_info_["Epoch"] = epoch
                        valid_info_["Benign"] = benign
                        valid_info_["csPCa"] = cspca
                        valid_info_["ranking_score"] = float(ranking_score_re.search(log_line).group(1))
                        valid_info_["AP"] = float(ranking_score_re.search(log_line).group(2))
                        valid_info_["AUROC"] = float(ranking_score_re.search(log_line).group(3))
                        valid_info.append(valid_info_)
                    if len(train_info_):
                        train_info.append(train_info_)
        except Exception:
            continue
        # print(job.JobID, " - train info:", len(train_info), " - valid info:", len(valid_info), "\n\n")                    
        train_infos[job.JobName].extend(train_info)
        valid_infos[job.JobName].extend(valid_info)
    return train_infos, valid_infos



if __name__ == "__main__":
    joblist = parse_joblist()
    train_infos, valid_infos = parse_logs(joblist)
    # pprint(valid_infos)
    # print(train_infos.keys())
    # for k, v in train_infos.items():
    #     print(k, len(v))



    num_plots = 8
    num_rows = 2 
    fig, axes = plt.subplots(num_rows, int(num_plots/num_rows), figsize=(20, 30))
    id1, id2 = 0,0 
    for idx, name in enumerate(train_infos): 
        if len(train_infos[name])==0:
            continue
    
        log = train_infos[name]
        
        # ??? THING ARE IN THERE DOUBLE? 
        # for t in thing: 
        #     if t['Epoch'] == 10: 
        #         print(t)

        result = {}
        for entry in log:
            epoch = entry['Epoch']
            train_loss = entry['Train loss']
            
            if epoch in result:
                result[epoch].append(train_loss)
            else:
                result[epoch] = [train_loss]
        
        sort_dict = sorted(result.items())
        average_values = [sum(item[1]) / len(item[1]) for item in sort_dict]

        axes[int(id1//4),int(id2%4)].plot(average_values)
        axes[int(id1//4),int(id2%4)].set_title(f'{name}')
        axes[int(id1//4),int(id2%4)].set_xlabel("epoch")
        axes[int(id1//4),int(id2%4)].set_ylabel("loss value")

    
        id1+=1
        id2+=1 

    plt.show()



