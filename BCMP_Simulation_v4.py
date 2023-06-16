import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import random
import sys
import time
from mpi4py import MPI
import collections
import itertools

class BCMP_Simulation:
    
    def __init__(self, N, R, K, U, mu, type_list, p, theoretical, sim_time, rank, size, m):
        self.N = N #網内の拠点数(プログラム内部で指定する場合は1少なくしている)
        self.R = R #クラス数
        self.K = K #網内の客数 K = [K1, K2]のようなリスト。トータルはsum(K)
        self.U = U #最大窓口数
        self.mu = mu #サービス率 FCFSはクラス別で変えられない
        self.type_list = type_list #Type1(FCFS),Type2プロセッサシェアリング（Processor Sharing: PS）,Type3無限サーバ（Infinite Server: IS）,Type4後着順割込継続型（LCFS-PR)
        self.p = p
        print('p : {0}'.format(self.p))
        self.m = m.values #窓口配列
        print('m : {0}'.format(self.m))
        #print(self.p.shape)
        self.event = [[] for i in range(self.N)] #各拠点で発生したイベント(arrival, departure)を格納
        self.eventclass = [[] for i in range(self.N)] #各拠点でイベント発生時の客クラス番号
        self.eventqueue = [[] for i in range(N)] #各拠点でイベント発生時のqueueの長さ
        self.eventtime = [[] for i in range(N)] #各拠点でイベントが発生した時の時刻
        self.timerate = np.zeros((self.N, sum(self.K)+1))#拠点での人数分布(0~K人の分布)、0人の場合も入る
        self.timerateclass = np.zeros((self.N, self.R, sum(self.K)+1))#拠点での人数分布(0~K人の分布)、0人の場合も入る、クラス別
        self.time = sim_time #シミュレーション時間
        self.theoretical = theoretical
        print('Theoretical Values : {0}'.format(self.theoretical.values))
        #print(self.p.iloc[0,1])
        self.rank = rank
        self.size = size
        self.process_text = './process/process_N'+str(self.N)+'_R'+str(self.R)+'_K'+str(sum(self.K))+',U_'+str(self.U)+'_Time'+str(self.time)+'.txt'
        self.start = time.time()
        
        
    def getSimulation(self):
        queue = np.zeros(self.N) #各拠点のサービス中を含むqueueの長さ(クラスをまとめたもの)
        queueclass = np.zeros((self.N, self.R)) #各拠点のサービス中を含むqueueの長さ(クラス別)
        classorder = [[] for i in range(self.N)] #拠点に並んでいる順のクラス番号
        service = np.zeros((self.N, self.U)) #サービス中の客の残りサービス時間
        total_length = np.zeros(self.N) #各拠点の延べ系内人数(クラスをまとめたもの)
        total_lengthclass = np.zeros((self.N, self.R)) #各拠点の延べ人数(クラス別)
        total_waiting = np.zeros(self.N) #延べ待ち人数(クラスをまとめたもの)
        total_waitingclass = np.zeros((self.N, self.R))#延べ待ち人数(クラス別)
        L = np.zeros(self.N) #平均系内人数(結果)
        Lc = np.zeros((self.N, self.R)) #平均系内人数(結果)(クラス別)
        Q = np.zeros(self.N) #平均待ち人数(結果)
        Qc = np.zeros((self.N, self.R)) #平均待ち人数(結果)(クラス別)
        rmse = [] #100単位時間でのrmseの値を格納
        rmse_time = [] #rmseを登録した時間
        regist_time = 50 #rmseの登録時刻
        regist_span = 50 #50単位で登録
        #確認用(plot)
        departure_Node = [] #退去が発生する拠点を記録(最小のサービス時間を持つ拠点でもある)
        arrival_Node = [] #到着が発生する拠点を記録
        
        elapse = 0
        initial_node = 0
        #Step1 開始時の客の分配 (開始時のノードは拠点番号0)
        for i in range(R):
            for j in range(K[i]):
                initial_node = random.randrange(self.N)#20220320 最初はランダムにいる拠点を決定
                self.event[initial_node].append("arrival")
                self.eventclass[initial_node].append(i) #到着客のクラス番号
                self.eventqueue[initial_node].append(queue[initial_node])#イベントが発生した時のqueueの長さ(到着客は含まない)
                self.eventtime[initial_node].append(elapse) #(移動時間0)
                queue[initial_node] +=1 #最初はノード0にn人いるとする
                queueclass[initial_node][i] += 1 #拠点0にクラス別人数を追加
                classorder[initial_node].append(i)#拠点0にクラス番号を追加
                if queue[initial_node] <= self.m[initial_node][0]:#窓口数よりqueueが小さいとき
                    service[initial_node][int(queue[initial_node] - 1)] = self.getExponential(self.mu[initial_node]) #窓口客のサービス時間設定
        
        '''
        print('Step1 開始時の客の分配 (開始時のノードは拠点番号0)')
        print('event : {0}'.format(self.event))
        print('eventclass : {0}'.format(self.eventclass))
        print('eventqueue : {0}'.format(self.eventqueue))
        print('eventtime : {0}'.format(self.eventtime))
        print('queue : {0}'.format(queue))
        print('queueclass : {0}'.format(queueclass))
        print('classorder : {0}'.format(classorder))
        print('service : {0}'.format(service))
        print('----- シミュレーション開始 -----')
        #'''
        
        #print('Simulation Start')
        #Step2 シミュレーション開始
        while elapse < self.time:
            #print('経過時間 : {0} / {1}'.format(elapse, self.time))
            mini_service = 100000#最小のサービス時間
            mini_index = -1 #最小のサービス時間をもつノード
            m_index = -1 #最小のサービス時間の窓口
           
            #print('Step2.1 次に退去が起こる拠点を検索')
            #Step2.1 次に退去が起こる拠点を検索
            for i in range(self.N):#待ち人数がいる中で最小のサービス時間を持つノードの窓口を算出
                if queue[i] > 0:
                    for j in range(self.m[i][0]):
                        if mini_service > service[i][j] and queue[i] > j:
                            mini_service = service[i][j]
                            mini_index = i
                            m_index = j
            departure_class = classorder[mini_index].pop(m_index) #退去する客のクラスを取り出す(指定窓口)(service配列とずれる)
            departure_Node.append(mini_index)

            '''
            print('現在時刻(elapse) : {0}'.format(elapse))
            print('最小のサービス時間(mini_service) : {0}'.format(mini_service))
            print('最小のサービス時間を持つ拠点番号(mini_index) : {0}'.format(mini_index))
            print('最小のサービス時間を持つ窓口番号(m_index) : {0}'.format(m_index))
            print('最小のサービス時間を持つ拠点のクラス(departure_class) : {0}'.format(departure_class))
            #'''
            
            #Step2.2 イベント拠点確定後、全ノードの情報更新(サービス時間、延べ人数)
            for i in range(self.N):#ノードiから退去(全拠点で更新)
                total_length[i] += queue[i] * mini_service #ノードでの延べ系内人数
                for r in range(self.R): #クラス別延べ人数更新
                    total_lengthclass[i,r] += queueclass[i,r] * mini_service
                if queue[i] > 0: #系内人数がいる場合(サービス中の客がいるとき)
                    for j in range(self.m[i][0]):
                        if service[i][j] > 0:
                            service[i][j] -= mini_service #サービス時間を減らす
                    total_waiting[i] += ( queue[i] - 1 ) * mini_service #ノードでの延べ待ち人数
                    for r in range(R):
                        if queueclass[i,r] > 0: #クラス別延べ待ち人数の更新
                            total_waitingclass[i,r] += ( queueclass[i,r] - 1 ) * mini_service 
                elif queue[i] == 0: #いらないかも
                    total_waiting[i] += queue[i] * mini_service
                self.timerate[i, int(queue[i])] += mini_service #人数分布の時間帯を更新
                for r in range(R):
                    self.timerateclass[i, r, int(queueclass[i,r])] += mini_service #人数分布の時間帯を更新
            '''
            print('Step2.2 イベント拠点確定後、全ノードの情報更新(サービス時間、延べ人数)')
            print('queue : {0}'.format(queue))
            print('queueclass : {0}'.format(queueclass))
            print('最小のサービス時間を引いたあとのservice : {0}'.format(service))
            print('total_length : {0}'.format(total_length))
            print('total_lengthclass : {0}'.format(total_lengthclass))
            print('timerate : {0}'.format(self.timerate))
            print('timerateclass : {0}'.format(self.timerateclass))
            #'''
        
            #Step2.3 退去を反映
            self.event[mini_index].append("departure") #退去を登録
            self.eventclass[mini_index].append(departure_class)
            self.eventqueue[mini_index].append(queue[mini_index]) #イベント時の系内人数を登録
            #self.eventqueueclass[mini_index, departure_class].append(queueclass[mini_index, departure_class]) #イベント時の系内人数を登録
            queue[mini_index] -= 1 #ノードの系内人数を減らす
            queueclass[mini_index, departure_class] -= 1 #ノードの系内人数を減らす(クラス別)
            elapse += mini_service
            self.eventtime[mini_index].append(elapse) #経過時間の登録はイベント後
            #serviceの修正
            for k in range(m_index, self.m[mini_index][0]):
                if (self.m[mini_index][0] - 1) == k:
                    break
                else:
                    service[mini_index][k] = service[mini_index][int(k + 1)]
                    service[mini_index][int(k + 1)] = 0
            if queue[mini_index] > 0: #窓口がすべて埋まっている -> １つ空いた
                for i in range(self.m[mini_index][0]):
                    if service[mini_index][i] == 0 and i < queue[mini_index]:#一応queueの長さを考慮
                        service[mini_index][i] = self.getExponential(self.mu[mini_index])#退去後まだ待ち人数がある場合、サービス時間設定
                        #print('サービス開始')
                        #print('queue[{0}] = {1}  窓口 {2} に追加'.format(mini_index, queue[mini_index], i)) #i=3ならOK
                        break

            '''
            print('Step2.3 退去の反映')
            print('退去後queue[{0}] : {1}'.format(mini_index, queue[mini_index]))
            print('退去後queueclass[{0}][{1}] : {2}'.format(mini_index, departure_class, queueclass[mini_index][departure_class]))
            print('退去後service : {0}'.format(service))
            #'''
            
            #Step2.4 退去客の行き先決定
            #推移確率行列が N*R × N*Rになっている。departure_class = 0の時は最初のN×N (0~N-1の要素)を見ればいい
            #departure_class = 1の時は (N~2N-1の要素)、departure_class = 2の時は (2N~3N-1の要素)
            #departure_class = rの時は (N*r~N*(r+1)-1)を見ればいい
            rand = random.random()
            sum_rand = 0
            destination_index = -1
            pr = np.zeros((self.N, self.N))#今回退去する客クラスの推移確率行列を抜き出す
            for i in range(self.N * departure_class, self.N * (departure_class + 1)):
                for j in range(self.N * departure_class, self.N * (departure_class + 1)):
                    pr[i - self.N * departure_class, j - self.N * departure_class] = self.p.iloc[i,j]
            #'''
            print('今回退去する客クラスの推移確率行列')
            print(pr)
            print(pr.shape)
            #'''
            
            for i in range(len(pr)): #11   
                sum_rand += pr[mini_index][i]
                if rand < sum_rand:
                    destination_index = i
                    break
            if destination_index == -1: #これは確率が1になってしまったとき用
                destination_index = len(pr) -1 #一番最後のノードに移動することにする
            self.event[destination_index].append("arrival") #イベント登録
            self.eventclass[destination_index].append(departure_class) #移動する客クラス番号登録
            self.eventqueue[destination_index].append(queue[destination_index])
            self.eventtime[destination_index].append(elapse) #(移動時間0)
            queue[destination_index] += 1 #推移先の待ち行列に並ぶ
            queueclass[destination_index][departure_class] += 1 #推移先の待ち行列(クラス別)に登録 
            classorder[destination_index].append(departure_class)
            #推移先で待っている客がいなければサービス時間設定(即時サービス)
            if queue[destination_index] <= self.m[destination_index][0]: #ここが間違ってるかも
                for i in range(self.m[destination_index][0]):
                    if service[destination_index][i] == 0 and i < queue[destination_index]:
                        service[destination_index][i] = self.getExponential(self.mu[destination_index]) #サービス時間設定
                        break
            arrival_Node.append(destination_index)
            
            '''
            print('Step2.4 退去客の行き先決定')
            print('destination_index : {0}'.format(destination_index))
            print('queue : {0}'.format(queue))
            print('queueclass : {0}'.format(queueclass))
            print('classorder : {0}'.format(classorder))
            print('service : {0}'.format(service))
            #'''
           
            #Step2.5 RMSEの計算
            if elapse > regist_time:
                rmse_sum = 0
                theoretical_value = self.theoretical.values
                lc = total_lengthclass / elapse #今までの時刻での平均系内人数
                for n in range(self.N):
                    for r in range(self.R):
                        rmse_sum += (theoretical_value[n,r] - lc[n,r])**2
                rmse_sum /= self.N * self.R
                rmse_value = math.sqrt(rmse_sum)
                rmse.append(rmse_value)
                rmse_time.append(regist_time)
                regist_time += regist_span
                print('Rank = {0}, Calculation Time = {1}'.format(self.rank, time.time() - self.start))
                print('Elapse = {0}, RMSE = {1}'.format(elapse, rmse_value))
                print('Elapse = {0}, Lc = {1}'.format(elapse, lc))
                if self.rank == 0:
                    with open(self.process_text, 'a') as f:
                        print('Rank = {0}, Calculation Time = {1}'.format(self.rank, time.time() - self.start), file=f)
                        print('Elapse = {0}, RMSE = {1}'.format(elapse, rmse_value), file=f)
                        print('Elapse = {0}, Lc = {1}'.format(elapse, lc), file=f)

        #plot(確認) -> 並列で最大・最小を除いた平均をとるように変更したい(まだ)
        plt.clf()
        plt.plot(rmse_time, rmse)
        plt.title('U = '+str(self.U))
        plt.xlabel('Time')
        plt.ylabel('RMSE')
        plt.savefig('./check/N_'+str(self.N)+',R_'+str(self.R)+',K_'+str(self.K)+',U_'+str(self.U)+',Time_'+str(self.time)+',Rank_'+str(self.rank)+'.png', format='png', dpi=300)

        L = total_length / self.time
        Lc = total_lengthclass / self.time
        Q = total_waiting / self.time
        Qc = total_waitingclass / self.time
        
        print('Rank = {0}, Calculation Time = {1}'.format(self.rank, time.time() - self.start))
        print('平均系内人数L : {0}'.format(L))
        print('平均系内人数(クラス別)Lc : {0}'.format(Lc))
        print('平均待ち人数Q : {0}'.format(Q))
        print('平均待ち人数(クラス別)Qc : {0}'.format(Qc))
        if self.rank == 0:
            with open(self.process_text, 'a') as f:
                print('Rank = {0}, Calculation Time = {1}'.format(self.rank, time.time() - self.start), file=f)
                print('平均系内人数L : {0}'.format(L), file=f)
                print('平均系内人数(クラス別)Lc : {0}'.format(Lc), file=f)
                print('平均待ち人数Q : {0}'.format(Q), file=f)
                print('平均待ち人数(クラス別)Qc : {0}'.format(Qc), file=f)
                   
        pd.DataFrame(L).to_csv('./csv/L(N_'+str(self.N)+',R_'+str(self.R)+',K_'+str(self.K)+',U_'+str(self.U)+',Time_'+str(self.time)+',Rank_'+str(self.rank)+').csv')
        pd.DataFrame(Lc).to_csv('./csv/Lc(N_'+str(self.N)+',R_'+str(self.R)+',K_'+str(self.K)+',U_'+str(self.U)+',Time_'+str(self.time)+',Rank_'+str(self.rank)+').csv')
        pd.DataFrame(Q).to_csv('./csv/Q(N_'+str(self.N)+',R_'+str(self.R)+',K_'+str(self.K)+',U_'+str(self.U)+',Time_'+str(self.time)+',Rank_'+str(self.rank)+').csv')
        pd.DataFrame(Qc).to_csv('./csv/Qc(N_'+str(self.N)+',R_'+str(self.R)+',K_'+str(self.K)+',U_'+str(self.U)+',Time_'+str(self.time)+',Rank_'+str(self.rank)+').csv')
        rmse_index = {'time': rmse_time, 'RMSE': rmse}
        df_rmse = pd.DataFrame(rmse_index)
        df_rmse.to_csv('./csv/RMSE(N_'+str(self.N)+',R_'+str(self.R)+',K_'+str(self.K)+',U_'+str(self.U)+',Time_'+str(self.time)+',Rank_'+str(self.rank)+').csv')

        print('service = {0}'.format(service)) #確認
        
    def getExponential(self, param):
        return - math.log(1 - random.random()) / param
    
    
    
if __name__ == '__main__':
   
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #推移確率行列に合わせる
    N = int(sys.argv[1])
    R = int(sys.argv[2])
    K_total = int(sys.argv[3])
    U = int(sys.argv[4])
    sim_time = int(sys.argv[5])
    p_file = sys.argv[6]
    theoretical_file = sys.argv[7]
    m_file = sys.argv[8]
    #N = 33 #33
    #R = 2
    #K_total = 500
    #U = 4
    K = [(K_total + i) // R for i in range(R)] #クラス人数を自動的に配分する
    mu = np.full(N, 1) #サービス率を同じ値で生成(サービス率は調整が必要)
    m = pd.read_csv(m_file, header=None, dtype=int)
    type_list = np.full(N, 1) #サービスタイプはFCFS
    #m = np.full(N, 1) #窓口数
    #p = pd.read_csv('csv/transition_probability_N33_R2_K500_Core128.csv')
    p = pd.read_csv(p_file, header=None)
    #p = pd.read_csv('csv/transition33.csv')
    #theoretical = pd.read_csv('csv/L_N33_R2_K500_Core128.csv')
    theoretical = pd.read_csv(theoretical_file, header=None)
    #theoretical = pd.read_csv('csv/Theoretical33.csv')
    #time = 10000
    bcmp = BCMP_Simulation(N, R, K, U, mu, type_list, p, theoretical, sim_time, rank, size, m) 
    start = time.time()
    bcmp.getSimulation()
    elapsed_time = time.time() - start
    print ("rank : {1}, calclation_time:{0}".format(elapsed_time, rank) + "[sec]")
    # python3 BCMP_Simulation_v1.py 33 1 500 100000 csv/transition_probability_N33_R1_K500_Core128.csv csv/L_N33_R1_K500_Core128.csv > result_33_1_500_100000.txt &
    # python3 BCMP_Simulation_v1.py 33 2 500 100000 csv/transition_probability_N33_R2_K500_Core128.csv csv/L_N33_R2_K500_Core128.csv > result_33_2_500_100000.txt &
    # python3 BCMP_Simulation_v1.py 33 3 500 100000 csv/transition_probability_N33_R3_K500_Core128.csv csv/L_N33_R3_K500_Core128.csv > result_33_3_500_100000.txt &
    #並列計算用
    #mpiexec -n 4 python BCMP_Simulation_v4.py 11 2 200 1 15000 tp/transition_probability_N11_R2_K200_Core16.csv tp/L_N11_R2_K200_Core16.csv > result_11_2_200_15000.txt
    #mpiexec -n 4 python BCMP_Simulation_v4.py 5 2 100 4 1000 tp/transition_probability_N5_R2_K100_Core16.csv tp/L_N5_R2_K100_Core16.csv > result_5_2_100_1000.txt
    #mpiexec -n 4 python BCMP_Simulation_v4.py 33 2 500 4 20000 tp/transition_probability_N33_R2_K500_Core8_v10.csv tp/L_N33_R2_K500_Core8_v10.csv > result_33_2_500_20000.txt
    #python BCMP_Simulation_v4.py 33 2 500 4 20000 transition_probability_std_33_2_500_16_10_4.csv ga_L_std_33_2_500_16_10_4.csv ga_Node_std_33_2_500_16_10.csv > result_33_2_500_20000.txt
    