import copy
import datetime
import numpy as np
import random
import re
import matplotlib.pyplot as plt
import math

# la16
jobs = 20
machines = 5

t_table = [[6, 40, 81, 37, 19], [40, 32, 55, 81, 9], [46, 65, 70, 55, 77], [21, 65, 64, 25, 15], [85, 40, 44, 24, 37], [89, 29, 83, 31, 84], [59, 38, 80, 30, 8], [80, 56, 77, 41, 97], [56, 91, 50, 71, 17], [40, 88, 59, 7, 80], [45, 29, 8, 77, 58], [36, 54, 96, 9, 10], [28, 73, 98, 92, 87], [70, 86, 27, 99, 96], [95, 59, 56, 85, 41], [81, 92, 32, 52, 39], [7, 22, 12, 88, 60], [45, 93, 69, 49, 27], [21, 84, 61, 68, 26], [82, 33, 71, 99, 44]]
m_table = []
m = list(range(1, machines + 1))
for i in range(jobs):
    m_table.append(random.shuffle(m))

m_table= [[0, 2, 1, 3, 4], [2, 3, 0, 4, 1], [1, 4, 2, 3, 0], [2, 4, 0, 3, 1], [2, 0, 1, 3, 4], [0, 4, 1, 3, 2], [4, 3, 1, 2, 0], [0, 2, 1, 4, 3], [4, 0, 3, 2, 1], [1, 0, 4, 2, 3], [0, 1, 2, 4, 3], [2, 0, 3, 1, 4], [0, 2, 1, 3, 4], [0, 3, 2, 1, 4], [1, 0, 4, 3, 2], [1, 2, 4, 0, 3], [1, 4, 2, 0, 3], [3, 0, 2, 4, 1], [0, 1, 2, 3, 4], [1, 2, 4, 0, 3]]

# # abz9
# jobs = 20
# machines = 15
#
# t_table =  [[14, 21, 13, 11, 11, 35, 20, 17, 18, 11, 23, 13, 15, 11, 35], [35, 31, 13, 26, 14, 17, 38, 20, 19, 12, 16, 34, 15, 12, 14], [30, 35, 40, 35, 30, 23, 29, 37, 38, 40, 26, 11, 40, 36, 17], [40, 18, 12, 23, 23, 14, 16, 14, 23, 12, 16, 32, 40, 25, 29], [35, 15, 31, 28, 32, 30, 27, 29, 38, 11, 23, 17, 27, 37, 29], [33, 33, 19, 40, 19, 33, 26, 31, 28, 36, 38, 21, 25, 40, 35], [25, 32, 33, 18, 32, 28, 15, 35, 14, 34, 23, 32, 17, 26, 19], [16, 33, 34, 30, 40, 12, 26, 26, 15, 21, 40, 32, 14, 30, 35], [17, 16, 20, 24, 26, 36, 22, 14, 11, 20, 23, 29, 23, 15, 40], [27, 37, 40, 14, 25, 30, 34, 11, 15, 32, 36, 12, 28, 31, 23], [25, 22, 27, 14, 25, 20, 18, 14, 19, 17, 27, 22, 22, 27, 21], [34, 15, 22, 29, 34, 40, 17, 32, 20, 39, 31, 16, 37, 33, 13], [12, 27, 17, 24, 11, 19, 11, 17, 25, 11, 31, 33, 31, 12, 22], [22, 15, 16, 32, 20, 22, 11, 19, 30, 33, 29, 18, 34, 32, 18], [27, 26, 28, 37, 18, 12, 11, 26, 27, 40, 19, 24, 18, 12, 34], [15, 28, 25, 32, 13, 38, 11, 34, 25, 20, 32, 23, 14, 16, 20], [15, 13, 37, 14, 22, 24, 26, 22, 34, 22, 19, 32, 29, 13, 35], [36, 33, 28, 20, 30, 33, 29, 34, 22, 12, 30, 12, 35, 13, 35], [26, 31, 35, 38, 19, 35, 27, 29, 39, 13, 14, 26, 17, 22, 15], [36, 34, 33, 17, 38, 39, 16, 27, 29, 16, 16, 19, 40, 35, 39]]
# m_table = []
# m = list(range(1, machines + 1))
# for i in range(jobs):
#     m_table.append(random.shuffle(m))
#
# m_table= [[6, 5, 8, 4, 1, 14, 13, 11, 10, 12, 2, 3, 0, 7, 9], [1, 5, 0, 3, 6, 9, 7, 12, 10, 13, 8, 4, 11, 14, 2], [0, 4, 2, 10, 6, 14, 8, 13, 7, 3, 9, 12, 1, 11, 5], [7, 5, 4, 8, 0, 9, 13, 12, 10, 3, 6, 14, 1, 11, 2], [2, 3, 12, 11, 6, 4, 10, 7, 0, 13, 1, 14, 5, 9, 8], [5, 3, 6, 12, 10, 0, 13, 2, 11, 7, 4, 1, 14, 9, 8], [13, 0, 11, 12, 4, 6, 5, 3, 9, 2, 7, 10, 1, 14, 8], [2, 12, 9, 11, 13, 8, 14, 5, 6, 3, 1, 4, 0, 7, 10], [2, 10, 14, 6, 8, 3, 12, 0, 13, 9, 7, 1, 11, 4, 5], [4, 9, 3, 11, 13, 7, 0, 2, 5, 12, 1, 10, 14, 8, 6], [13, 0, 3, 8, 5, 6, 14, 7, 1, 2, 4, 9, 12, 11, 10], [14, 10, 0, 3, 13, 6, 7, 2, 12, 5, 4, 11, 1, 8, 9], [6, 12, 4, 2, 8, 5, 14, 3, 9, 1, 11, 13, 7, 10, 0], [5, 14, 0, 8, 7, 4, 9, 13, 1, 12, 6, 11, 3, 10, 2], [5, 3, 10, 6, 4, 12, 11, 13, 7, 9, 14, 1, 2, 0, 8], [8, 5, 9, 6, 1, 7, 11, 2, 4, 0, 10, 3, 12, 14, 13], [1, 4, 8, 3, 10, 5, 12, 7, 9, 14, 11, 13, 0, 2, 6], [7, 5, 13, 9, 10, 4, 14, 0, 3, 11, 6, 8, 1, 2, 12], [14, 11, 5, 2, 13, 10, 4, 8, 3, 9, 6, 7, 0, 1, 12], [1, 7, 11, 8, 14, 6, 5, 3, 13, 2, 0, 4, 9, 12, 10]]

def com_tr(t_table):
    topo_order = []
    for j_num, o_num in enumerate(t_table):  # 根據工作時間獲取工件數的index和工序列表
        topo_order = topo_order+(np.ones([1, len(o_num)], int) * (j_num + 1)).tolist()
    combin = []
    for li in topo_order:
        combin = combin+li

    random.shuffle(combin)  # 隨機打亂
    return combin

class Cij:
    def __init__(self, name, StartTime, LoadTime):
        self.name = name
        self.StartTime = StartTime
        self.LoadTime = LoadTime
        self.EndTime = StartTime + LoadTime

def c_max(combin):
    # 函數給值，將工件數、機器數與加工時間進行綁定
    # enumerate() 函數用於將一個可遍歷的資料物件(如列表、元組或字符串)組合為一個索引序列
    for i in range(len(combin)):  # 工件索引
        job = combin[i]  # 工件
        no_job = combin[:i+1].count(job)  # 工序
        machine = m_table[job-1][no_job-1]  # 機器
        loadtime = t_table[job-1][no_job-1]  # 該工序加工時間
        locals()['c{}_{}_{}'.format(job, no_job, machine)] = Cij(name='c{}_{}_{}'.format(job, no_job, machine),
                                                                  StartTime=0, LoadTime=loadtime, )
    load_time_tables = []
    # M_time = np.zeros(max(max(m_table)))
    M_time=np.zeros(machines)
    for i in range(len(combin)):
        job = combin[i]  # 工件号
        no_job = combin[:i+1].count(job)
        machine = m_table[job-1][no_job-1]
        if no_job == 1:  # 工序編號，開始時間為機器完成上個工件任務的時間或0
            locals()['c{}_{}_{}'.format(job, no_job, machine)].StartTime = M_time[machine-1]
        else:
            locals()['c{}_{}_{}'.format(job, no_job, machine)].StartTime = max(
                M_time[machine-1],  # 該工序所在加工位置機器的時間
                locals()['c{}_{}_{}'.format(job, no_job-1, m_table[job-1][no_job-2])].EndTime)  # 前道工序的完成時間
        locals()['c{}_{}_{}'.format(job, no_job, machine)].EndTime = locals()['c{}_{}_{}'.format(job, no_job, machine)].StartTime + \
                                                          locals()['c{}_{}_{}'.format(job, no_job, machine)].LoadTime
        M_time[machine - 1] = locals()['c{}_{}_{}'.format(job, no_job, machine)].EndTime
        load_time_tables.append([locals()['c{}_{}_{}'.format(job, no_job, machine)].name, [
            locals()['c{}_{}_{}'.format(job, no_job, machine)].StartTime,
            locals()['c{}_{}_{}'.format(job, no_job, machine)].EndTime]])
        T=[]
        for i in load_time_tables:
            T.append(i[-1][-1])
    # print(load_time_tables)
    return load_time_tables, max(T)  # load_time_tables 所有工件中每個工序加工位置的開始與結束時間。

# 种群初始化
def init_population(pop_size, chrom):
    pop = []
    for i in range(pop_size):
        c = copy.deepcopy(chrom)
        random.shuffle(c)
        pop.append(c)
    return pop

# 適應度
def fitness(combin):
    return 1/(c_max(combin)[1])

class node:
    def __init__(self, state):
        self.state = state
        self.load_table = c_max(state)[0]  # 求出染色體上每個工序所在機器的開始結束時間表
        self.makespan = c_max(state)[1]
        self.fitness = fitness(state)

def two_points_cross(chro1, chro2):
    # 不改變位置
    chro1_1 = copy.deepcopy(chro1)
    chro2_1 = copy.deepcopy(chro2)
    # 交叉位置，point1<point2
    point1 = random.randint(0, len(chro1_1))
    point2 = random.randint(0, len(chro1_1))
    while point1 > point2 or point1 == point2:
        point1 = random.randint(0, len(chro1_1))
        point2 = random.randint(0, len(chro1_1))

    # 紀錄交叉片段
    frag1 = chro1[point1:point2]
    frag2 = chro2[point1:point2]
    random.shuffle(frag1)
    random.shuffle(frag2)
    # 交叉
    chro1_1[point1:point2], chro2_1[point1:point2] = chro2_1[point1:point2], chro1_1[point1:point2]

    child1 = chro1_1[:point1] + frag1 + chro1_1[point2:]
    child2 = chro2_1[:point1] + frag2 + chro2_1[point2:]

    return child1, child2


# 交換變異
def gene_exchange(n):
    point1 = random.randint(0, len(n) - 1)
    point2 = random.randint(0, len(n) - 1)
    while point1 == point2 or point1 > point2:
        point1 = random.randint(0, len(n) - 1)
        point2 = random.randint(0, len(n) - 1)
    n[point1], n[point2] = n[point2], n[point1]
    return n


# 插入變異
def gene_insertion(n):
    point1 = random.randint(0, len(n) - 1)
    point2 = random.randint(0, len(n) - 1)
    while point1 == point2:
        point1 = random.randint(0, len(n) - 1)
        point2 = random.randint(0, len(n) - 1)
    x = n.pop(point1)
    n.insert(point2, x)
    return n


# 部分逆序變異
def gene_reverse(n):
    point1 = random.randint(0, len(n) - 1)
    point2 = random.randint(0, len(n) - 1)
    while point1 == point2 or point1 > point2:
        point1 = random.randint(0, len(n) - 1)
        point2 = random.randint(0, len(n) - 1)
    ls_res = n[point1:point2]
    ls_res.reverse()
    l1 = n[:point1]
    l2 = n[point2:]
    n_res_end = l1 + ls_res + l2
    return n_res_end

def select(population):
    pop_fit=[]
    for i in population:
        pop_fit.append(fitness(i))
    best_chrom = min(pop_fit)

    return best_chrom

def update_AK(A0, r0, K, t, c_r0, mu0):  # 更新參數
    A = K/(1+(K/A0-1)*np.exp(-r0*t))
    r = r0*(1-A/K)
    c_r = c_r0*(1-A/K)
    mu = mu0*(1-A/K)
    pop_size = math.ceil(math.log(K / A)) + 2  # 種群規模
    return A, r, c_r, mu, pop_size
def update_solution(population):
    solution_list = []
    # 可行解集，包含开始结束时间等信息
    for i in population:
        # locals()['solution{}'.format(population.index(i))] = node(i)
        solution_list.append(node(i))
    solution_list.sort(key=lambda x: x.makespan)  # 排序後首個染色體為最佳解
    pops = [i.state for i in solution_list]  # 把solution_list的染色體複製到pops中
    f_list = [i.makespan for i in solution_list]
    Xb, fb, fave = pops[0], f_list[0], np.mean(f_list)  # 最佳個體與平均適應度
    return solution_list, Xb, fb, fave, f_list

# 改進GA算法
A0 = 100
r0 = 0.01
K = 10000
t = 1
c_r0 = 0.8  # 交叉機率
mu0 = 0.9  # 變異機率
A,r,c_r,mu,pop_size = update_AK(A0,r0,K,t,c_r0,mu0)
target_points = [1, 2, 3]

combin = com_tr(t_table)  # 工件編碼
# population = init_population(pop_size, combin)

# 開始循環運作
start = datetime.datetime.now()
best_fit = []
# iters=100
population = init_population(pop_size, combin)
solution_list = []
for i in population:
    # locals()['solution{}'.format(population.index(i))] = node(i)
    solution_list.append(node(i))
solution_list.sort(key=lambda x: x.makespan)  # 排序後首個染色體為最佳解
pops = [i.state for i in solution_list]  # 把solution_list的染色體複製到pops中
f_list = [i.makespan for i in solution_list]
Xb, fb, fave = pops[0], f_list[0], np.mean(f_list)  # 最佳個體與平均適應度
best_fit.append(fb)

while A < K*0.98:
    if t % 10 == 0:
        print('第{}次進化後的Cmax為{}'.format(t, fb))
    # pop_new = init_population(pop_size, combin)
    pop_new = copy.deepcopy(pops)
    for k in range(1, len(pop_new)):
        pk = np.exp((f_list[k]-fb)/(A*r))
        if pk < random.random():
            if mu > random.random():
                target = random.choice(target_points)
                if target == 1:
                    pop_new[k] = gene_exchange(pop_new[k])
                elif target == 2:
                    pop_new[k] = gene_insertion(pop_new[k])
                else:
                    pop_new[k] = gene_reverse(pop_new[k])
            elif c_r > random.random():
                pop_new[k], pop_new[k-1] = two_points_cross(pop_new[random.randint(0, int(pop_size/2))], pop_new[random.randint(math.ceil(pop_size/2), pop_size-1)])
            else:
                random.shuffle(pop_new[k])
        else:
            Xb, fb = pop_new[k], f_list[k]
    t += 1
    best_fit.append(fb)
    cross_population = pop_new
    cross_solution = [node(i) for i in cross_population]
    A, r, c_r, mu, pop_size = update_AK(A0, r0, K, t, c_r0, mu0)
    population = init_population(pop_size, combin)
    solution_list = [node(i) for i in population]
    solution_list = solution_list + cross_solution
    solution_list.sort(key=lambda x: x.makespan)
    del solution_list[pop_size:]  # 刪除popsize後面的可行解，使其大小穩定
    pops = [i.state for i in solution_list]
    f_list = [i.makespan for i in solution_list]
    Xb, fb, fave = pops[0], f_list[0], np.mean(f_list)
print('演化完成，最終Cmax為：', fb)
end = datetime.datetime.now()
print('耗時{}'.format(end - start))
print(solution_list[0].load_table)


# 繪製甘特圖
def color():
    color_ls = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    col = ''
    for i in range(6):
        col += random.choice(color_ls)
    return '#'+col
colors = [color() for i in range(len(t_table))]
for i in node(Xb).load_table:
    # print(i)
    y = eval(re.findall('_(\d+)', i[0])[1])

    label=re.findall(r'(\d*?)_', i[0])[0]
    plt.barh(y=y, left=i[1][0], width=i[1][-1] - i[1][0], height=0.5, color=colors[eval(label) - 1],
             label=f'job{label}')
# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
plt.title('Scheduling Gnatt GA')
plt.xlabel('Time')
plt.ylabel('Machine')
handles, labels = plt.gca().get_legend_handles_labels()

from collections import OrderedDict
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.show()