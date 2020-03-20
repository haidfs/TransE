# -*- coding: UTF-8 -*-
import timeit
from random import uniform, sample, choice
import numpy as np
from copy import deepcopy
import logging

LOG_FORMAT = "%(asctime)s - %(name)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

INITIAL_LEARNING_RATE = 0.01


def get_details_of_entityOrRels_list(file_path, split_delimeter="\t"):
    num_of_file = 0
    lyst = []
    with open(file_path) as file:
        # 确实是直接使用readlines的，低内存模式是在read_csv api中使用 csv_data = pd.read_csv(csv_file, low_memory=False)
        # 读取时可以不写r的参数，因为mode参数默认即为r
        lines = file.readlines()
        for line in lines:
            details_and_id = line.strip().split(split_delimeter)
            lyst.append(details_and_id[0])
            num_of_file += 1
    return num_of_file, lyst


def get_details_of_triplets_list(file_path, split_delimeter="\t"):
    num_of_file = 0
    lyst = []
    with open(file_path) as file:
        lines = file.readlines()
        for line in lines:
            triple = line.strip().split(split_delimeter)
            if len(triple) < 3:
                continue
            lyst.append(tuple(triple))
            num_of_file += 1
    return num_of_file, lyst


def norm(lyst):
    # 归一化 单位向量
    var = np.linalg.norm(lyst)
    i = 0
    while i < len(lyst):
        lyst[i] = lyst[i] / var
        i += 1
    # 需要返回array值 因为list不支持减法
    return np.array(lyst)


def dist_L1(h, t, l):
    s = h + l - t
    # 曼哈顿距离/出租车距离， |x-xi|+|y-yi|直接对向量的各个维度取绝对值相加
    return np.fabs(s).sum()


def dist_L2(h, t, l):
    s = h + l - t
    # 欧氏距离,是向量的平方和未开方。一定要注意，归一化公式和距离公式的错误书写，会引起收敛的失败
    return (s * s).sum()


class TransE(object):
    def __init__(
            self,
            entity_list,
            rels_list,
            triplets_list,
            margin=1,
            learing_rate=INITIAL_LEARNING_RATE,
            dim=100,
            normal_form="L1",
            batch_size=10000):
        self.learning_rate = learing_rate
        self.loss = 0
        self.entity_list = entity_list
        self.rels_list = rels_list
        self.triplets_list = triplets_list
        self.margin = margin
        self.dim = dim
        self.normal_form = normal_form
        self.batch_size = batch_size
        self.entity_vector_dict = {}
        self.rels_vector_dict = {}
        self.loss_list = []
        self.initialize()

    def initialize(self):
        '''
        对论文中的初始化稍加改动
        初始化l和e，对于原本的l和e的文件中的/m/06rf7字符串标识转化为定义的dim维向量，对dim维向量进行uniform和norm归一化操作
        :return:
        '''
        entity_vector_dict, rels_vector_dict = {}, {}
        # component的意思是向量的分量，当达到向量维数之后，对向量进行归一化，就完成了伪码中的初始化部分。
        entity_vector_compo_list, rels_vector_compo_list = [], []
        for item, dyct, compo_list, name in zip([self.entity_list, self.rels_list],
                                                [entity_vector_dict, rels_vector_dict],
                                                [entity_vector_compo_list, rels_vector_compo_list],
                                                ["entity_vector_dict", "rels_vector_dict"]):
            for entity_or_rel in item:
                n = 0
                compo_list = []
                while n < self.dim:
                    random = uniform(-6 / (self.dim ** 0.5),
                                     6 / (self.dim ** 0.5))
                    compo_list.append(random)
                    n += 1
                compo_list = norm(compo_list)
                dyct[entity_or_rel] = compo_list
        self.entity_vector_dict = entity_vector_dict
        self.rels_vector_dict = rels_vector_dict

    def transE(self, cycle_index=20):
        for i in range(cycle_index):
            start = timeit.default_timer()
            Sbatch = self.sample(self.batch_size)
            Tbatch = []  # 元组对（原三元组，打碎的三元组）的列表 ：{((h,r,t),(h',r,t'))}
            for sbatch in Sbatch:
                # 这里的pos_neg_triplets代表正负例三元组对，positive，negative
                pos_neg_triplets = (
                    sbatch, self.get_corrupted_triplets(sbatch))
                if pos_neg_triplets not in Tbatch:
                    Tbatch.append(pos_neg_triplets)
            self.update(Tbatch)
            if i % 1 == 0:
                # 可以更改i值考虑使用ema，即指数滑动平均
                # self.learning_rate = INITIAL_LEARNING_RATE * (pow(0.96, i / 100))
                end = timeit.default_timer()
                logging.info(
                    "Simple TransE, After %d training epoch(s):\nbatch size is %d, cost time is %g s, loss on batch data is %g" %
                    (i, self.batch_size, end - start, self.loss))
                # 查看最后的结果收敛情况
                self.loss_list.append(self.loss)
                if i % 100 == 0:
                    self.write_vector("data/entityVector.txt", "entity")
                    self.write_vector("data/relationVector.txt", "rels")
                self.loss = 0

    def sample(self, size):
        return sample(self.triplets_list, size)

    def get_corrupted_triplets(self, triplets):
        '''training triplets with either the head or tail replaced by a random entity (but not both at the same time)
        :param triplet:单个（h,t,l）
        :return corruptedTriplet:'''
        coin = choice([True, False])
        # 由于这个时候的(h,t,l)是从train文件里面抽出来的，要打坏的话直接随机寻找一个和头实体不等的实体即可
        if coin:  # 抛硬币 为真 打破头实体，即第一项
            while True:
                # 取第一个元素是因为sample返回的是一个列表类型
                searching_entity = sample(self.entity_vector_dict.keys(), 1)[0]
                if searching_entity != triplets[0]:
                    break
            corrupted_triplets = (searching_entity, triplets[1], triplets[2])
        else:  # 反之，打破尾实体，即第二项
            while True:
                searching_entity = sample(self.entity_vector_dict.keys(), 1)[0]
                if searching_entity != triplets[1]:
                    break
            corrupted_triplets = (triplets[0], searching_entity, triplets[2])
        return corrupted_triplets

    def update(self, Tbatch):
        entity_vector_copy = self.entity_vector_dict
        rels_vector_copy = self.rels_vector_dict

        # 这里的h,t,r代表头实体向量、尾实体向量、关系向量，h2和t2代表论文中的h'和t'，即负例三元组中的头尾实体向量
        # Tbatch是元组对（原三元组，打碎的三元组）的列表
        # ：[((h,r,t),(h',r,t'))...]，这里由于data文件的原因是(h,t,r)
        for pos_neg_triplets in Tbatch:
            h = entity_vector_copy[pos_neg_triplets[0][0]]
            t = entity_vector_copy[pos_neg_triplets[0][1]]
            r = rels_vector_copy[pos_neg_triplets[0][2]]
            # 损坏三元组中的头实体向量与尾实体向量
            h2 = entity_vector_copy[pos_neg_triplets[1][0]]
            t2 = entity_vector_copy[pos_neg_triplets[1][1]]
            # 这里原本定义了beforebatch，但是个人认为没有必要，这里已经进入到batch里面了，走的就是单个处理
            if self.normal_form == "L1":
                dist_triplets = dist_L1(h, t, r)
                dist_corrupted_triplets = dist_L1(h2, t2, r)
            else:
                dist_triplets = dist_L2(h, t, r)
                dist_corrupted_triplets = dist_L2(h2, t2, r)
            eg = self.margin + dist_triplets - dist_corrupted_triplets
            if eg > 0:  # 大于0取原值，小于0则置0.即合页损失函数margin-based ranking criterion
                self.loss += eg
                temp_positive = 2 * self.learning_rate * (t - h - r)
                temp_negative = 2 * self.learning_rate * (t2 - h2 - r)
                if self.normal_form == "L1":
                    temp_positive_L1 = [1 if temp_positive[i] >= 0 else -1 for i in range(self.dim)]
                    temp_negative_L1 = [1 if temp_negative[i] >= 0 else -1 for i in range(self.dim)]
                    temp_positive = np.array(temp_positive_L1) * self.learning_rate
                    temp_negative = np.array(temp_negative_L1) * self.learning_rate

                # 对损失函数的5个参数进行梯度下降， 随机体现在sample函数上
                h += temp_positive
                t -= temp_positive
                r = r + temp_positive - temp_negative
                h2 -= temp_negative
                t2 += temp_negative

                # 归一化刚才更新的向量，减少计算时间
                entity_vector_copy[pos_neg_triplets[0][0]] = norm(h)
                entity_vector_copy[pos_neg_triplets[0][1]] = norm(t)
                rels_vector_copy[pos_neg_triplets[0][2]] = norm(r)
                entity_vector_copy[pos_neg_triplets[1][0]] = norm(h2)
                entity_vector_copy[pos_neg_triplets[1][1]] = norm(t2)

            self.entity_vector_dict = entity_vector_copy
            self.rels_vector_dict = rels_vector_copy

    def write_vector(self, file_path, option):
        if option.strip().startswith("entit"):
            logging.info(
                "Write entities vetor into file      : {}".format(file_path))
            # dyct = deepcopy(self.entity_vector_dict)
            dyct = self.entity_vector_dict
        if option.strip().startswith("rel"):
            logging.info(
                "Write relationships vector into file: {}".format(file_path))
            # dyct = deepcopy(self.rels_vector_dict)
            dyct = self.rels_vector_dict
        with open(file_path, 'w') as file:  # 写文件，每次覆盖写 用with自动调用close
            for dyct_key in dyct.keys():
                file.write(dyct_key + "\t")
                file.write(str(dyct[dyct_key].tolist()))
                file.write("\n")

    def write_loss(self, file_path, num_of_col):
        with open(file_path, 'w') as file:
            lyst = deepcopy(self.loss_list)
            for i in range(len(lyst)):
                if num_of_col == 1:
                    # 保留4位小数
                    file.write(str(int(lyst[i] * 10000) / 10000) + "\n")
                else:
                    file.write(str(int(lyst[i] * 10000) / 10000) + "    ")
                    if (i + 1) % num_of_col == 0 and i != 0:
                        file.write("\n")


def prepare_fb15k_train_data():
    entity_file = "data/FB15k/entity2id.txt"
    num_entity, entity_list = get_details_of_entityOrRels_list(entity_file)
    logging.info("The number of entity_list is %d." % num_entity)
    rels_file = "data/FB15k/relation2id.txt"
    num_rels, rels_list = get_details_of_entityOrRels_list(rels_file)
    logging.info("The num of rels_list is %d." % num_rels)
    train_file = "data/FB15k/train.txt"
    num_triplets, train_triplets_list = get_details_of_triplets_list(
        train_file)
    logging.info("The num of train_triplets_list is %d." % num_triplets)
    return entity_list, rels_list, train_triplets_list


def main():
    # 对应TrainMain中的 --multi_process "None"的测试代码
    entity_list, rels_list, train_triplets_list = prepare_fb15k_train_data()

    transE = TransE(
        entity_list,
        rels_list,
        train_triplets_list,
        margin=1,
        dim=100,
        learing_rate=0.003)
    logging.info("TransE is initializing...")
    transE.transE(5000)

    # transE.transE2(num_of_epochs=1000, epoch_triplets=15000, num_of_batches=10)
    logging.info("********** End TransE training ***********\n")


if __name__ == "__main__":
    main()