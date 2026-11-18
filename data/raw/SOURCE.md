# Raw data source

`batch_task_sample.csv` is a uniform random sample of rows from the **Alibaba Cluster Trace v2018** `batch_task`
table, kept after the filters in `src/preprocess_trace.py`. It is stored headerless, in the original trace format:

task_name, instance_num, job_name, task_type, status, start_time, end_time, plan_cpu, plan_mem

- Repository: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018
- Full file: http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/batch_task.tar.gz (about 125 MB)
- Units: `plan_cpu` 100 = 1 core; `plan_mem` normalised to a machine's memory of 100; times in seconds.

Re-running `python -m src.preprocess_trace --input data/raw/batch_task_sample.csv` reproduces `data/processed/` exactly.
