# Sources and attribution

## Dataset
- **Alibaba Cluster Trace v2018** (cluster-trace-v2018), `batch_task` table. Alibaba Group, Alibaba Open Cluster Trace Program.
  Repository and documentation: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018
  Download (from the official `fetchData.sh`): http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/batch_task.tar.gz
  Schema: https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/schema.txt
  The trace is published for study and research use; see the repository for its terms.

## Papers (from the project proposal)
- Braun, T. D., Siegel, H. J., Beck, N., et al. (2001). A comparison of eleven static heuristics for mapping a class of independent tasks onto heterogeneous distributed computing systems. *Journal of Parallel and Distributed Computing*, 61(6), 810–837. https://doi.org/10.1006/jpdc.2000.1714 — Min-Min definition.
- Hou, H., Agos Jawaddi, S. N., & Ismail, A. (2024). Energy efficient task scheduling based on deep reinforcement learning in cloud environment: A specialized review. *Future Generation Computer Systems*, 151, 214–231. https://doi.org/10.1016/j.future.2023.10.002 — MDP formulation, reward design, research gaps.
- Hou, H., & Ismail, A. (2024). EETS: An energy-efficient task scheduler in cloud computing based on improved DQN algorithm. *Journal of King Saud University – Computer and Information Sciences*, 36(8), 102177. https://doi.org/10.1016/j.jksuci.2024.102177
- Yu, X., Mi, J., Tang, L., Long, L., & Qin, X. (2025). Dynamic multi objective task scheduling in cloud computing using reinforcement learning for energy and cost optimization. *Scientific Reports*, 15, 45387. https://doi.org/10.1038/s41598-025-29280-z — weighted multi-objective reward (adapted).
- Zhang, W., & Ou, H. (2025). Reinforcement learning based multi objective task scheduling for energy efficient and cost effective cloud edge computing. *Scientific Reports*, 15, 41716. https://doi.org/10.1038/s41598-025-25666-1 — baselines (FCFS, Min-Min), metrics, MDP elements.
- Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533. https://doi.org/10.1038/nature14236 — DQN, experience replay, target network.

## Software
- PyTorch (https://pytorch.org), NumPy, pandas, Matplotlib, Streamlit.

## Code
No external code was copied or adapted. The simulator, schedulers, DQN and UI were written for this project,
following the algorithm descriptions in the papers above and the libraries' documentation.
