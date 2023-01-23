#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/12/9 23:15
# @Author  : Binquan Guo (Allen Guo)
# @Email   : 13468897661@163.com
# @Profile : https://wilixx.github.io
# @File    : smart_grid_model_ICCTS_method.py
# @Function: To formulate the Mixed Integer Problems our paper entitled below:
# "Data Volume-aware Computation Task Scheduling for Smart Grid Data Analytic Applications".

""" Citation:
@inproceedings{guo2022optimal,
  title={Optimal Job Scheduling and Bandwidth Augmentation in Hybrid Data Center Networks},
  author={Guo, Binquan and Zhang, Zhou and Yan, Ye and Li, Hongyan},
  booktitle={GLOBECOM 2022-2022 IEEE Global Communications Conference},
  pages={5686--5691},
  year={2022},
  organization={IEEE}
}
"""
import numpy as np
import cvxpy as cp
import datetime
import time

""" An example job.  """
Job = np.array([j for j in range(9)])
p = np.array([1 for j in range(len(Job))])

# S: edges for data dependencies, or possible data transfers.
S = np.array([[1, 2], [2, 3], [3, 4], [4, 8], [1, 6], [5, 6], [6, 7], [7, 8], [7, 9]])-1
q = np.array([1 for e in range(len(S))])
print(S)

""" Step-1: Preparation """

""" Auxiliary matrix for ease of constraints construction. """
size = len(set([n for e in S for n in e]))
E_uv = np.zeros((size, size))
for index, uv in enumerate(S):
    E_uv[uv[0]][uv[1]] = int(index)
print("E_uv=", E_uv)

m_param = 3
Machine = np.array([i for i in range(m_param)])  # Machines
Transmitter = np.array([r for r in range(1)])  # Network channels

# T_upper_bound = np.sum(p)  + np.sum(q) # Initial weaker T_upper_bound
T_upper_bound = np.sum(p)    # Initial strict T_upper_bound

T = np.array([r for r in range(T_upper_bound)])  # Time horizon
M = np.array([i for i in range(len(Machine))])  # Machines axis
J = np.array([i+1 for i in range(len(Job))])  # task axis
print("J=", J)
E = np.array([i+1 for i in range(len(S))])  # network flow axis
print("T=", T)

""" Define BIG_M"""
BIG_M_M = len(Machine)  # + 1
BIG_M_T = T_upper_bound  # + 1

""" Step-2: Problem formulation """

""" Define variables: 3-D matrix"""
C_jit = {}
for j in range(len(Job)):
    C_jit[j] = cp.Variable((len(Machine), len(T)), boolean=True)

M_jjm = {}
for j in range(len(Job)):
    M_jjm[j] = cp.Variable((len(Job), len(Machine)), boolean=True)

N_ket = {}
for e in range(len(S)):       # the virtual machine, which is a concept mentioned in our paper.
    N_ket[e] = cp.Variable((len(Transmitter)+1, len(T)), boolean=True)

print("C_ijt", C_jit)
print("N_ket", N_ket)

makespan_C = cp.Variable(1, integer=True)  # 总工时是整数，因为考虑时隙型的任务

""" Constraints"""
constraints = []

""" Computing resource constraints """
for j in range(len(Job)):
    constraints.append(cp.sum(C_jit[j]) == 1)

exp = 0
for j in range(len(Job)):
    exp = exp + C_jit[j]
constraints.append(exp <= 1)
exp = 0

""" Communication resource constraints """
for e in range(len(S)):
    constraints.append(cp.sum(N_ket[e]) == 1)

exp = 0
for e in range(len(S)):
    exp = exp + N_ket[e][0:-1]   # [:-1]
constraints.append(exp <= 1)
exp = 0

""" Precedence constraints: this part may be a little hard to follow. 
Make sure you understand the techniques in our paper. """
for u, v in S:
    print("***")
    constraints.append(
        T * cp.sum(C_jit[u], axis=0) + p[u]  # + 1
        <= T * cp.sum(C_jit[v], axis=0)
    )

    constraints.append(
        cp.sum(M_jjm[u][v]) <= 1)

    constraints.append(
         cp.sum(N_ket[int(E_uv[u][v])][:-1])  # * p[u]
         <= T * cp.sum(N_ket[int(E_uv[u][v])], axis=0) - T * cp.sum(C_jit[u], axis=0)  # + p[u]
    )

    constraints.append(
         1 <= T * cp.sum(C_jit[v], axis=0) - T * cp.sum(N_ket[int(E_uv[u][v])], axis=0)
    )

    constraints.append(
        cp.sum(M_jjm[u][v]) == cp.sum(N_ket[int(E_uv[u][v])][-1]))

    constraints.append(
        0 <= cp.sum(C_jit[u], axis=1) + cp.sum(C_jit[v], axis=1) - 2 * M_jjm[u][v])
    constraints.append(
        cp.sum(C_jit[u], axis=1) + cp.sum(C_jit[v], axis=1) - 2 * M_jjm[u][v] <= 1)

""" Objective function linearizing """

for j in Job:
    constraints.append(makespan_C >= (T * cp.sum(C_jit[j], axis=0) + p[j]))

for e in range(len(S)):
    constraints.append(makespan_C >= (T * cp.sum(N_ket[e][:-1], axis=0) + q[e]))

problem = cp.Problem(cp.Minimize(makespan_C), constraints)
print("problem is DCP:", problem.is_dcp())

""" Step-3: Problem solving and result extraction """

start_time = datetime.datetime.now().time().strftime('%H:%M:%S')
print("The start time is ", start_time)
print("#########################")
start_milliseconds = int(round(time.time() * 1000))

problem.solve(solver=cp.GUROBI, verbose=True)

""" TO """
stop_milliseconds = int(round(time.time() * 1000))
duration_time_1 = (stop_milliseconds - start_milliseconds) / 1000
end_time = datetime.datetime.now().time().strftime('%H:%M:%S')
total_time=(datetime.datetime.strptime(end_time,'%H:%M:%S') - datetime.datetime.strptime(start_time,'%H:%M:%S'))

print("The stop time is ", end_time)
print("The computation time is ", total_time)

print("Status: ", problem.status)
print("The optimal value is", problem.value)
print("A solution x is", makespan_C.value)

computing_job_schedule = np.sum(np.array([(j+1) * C_jit[j].value for j in range(len(Job))]), axis=0)
networking_job_schedule = np.sum(np.array([(e+1) * N_ket[e].value for e in range(len(S))]), axis=0)

print("compute job result:")
print(computing_job_schedule)

print("Network result:")
print(networking_job_schedule)
