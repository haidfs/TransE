from random import uniform, sample, choice
import numpy as np
from copy import deepcopy


def get_details_of_entityOrRels_list(file_path, split_delimeter="\t"):
    num_of_file = 0
    lyst = []
    with open(file_path) as file:
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
    # return list
    return np.array(lyst)


def dist_L1(h, t, l):
    s = h + l - t
    # 曼哈顿距离/出租车距离， |x-xi|+|y-yi|直接对向量的各个维度取绝对值相加
    # dist = np.fabs(s).sum()
    return np.fabs(s).sum()


def dist_L2(h, t, l):
    s = h + l - t
    # 欧氏距离,是向量的平方和未开方。一定要注意，归一化公式和距离公式的错误书写，会引起收敛的失败
    # dist = (s * s).sum()
    return (s * s).sum()


class TransE(object):
    def __init__(self, entity_list, rels_list, triplets_list, margin=1, learing_rate=0.01, dim=50, normal_form="L1"):
        self.learning_rate = learing_rate
        self.loss = 0
        self.entity_list = entity_list  # entityList是entity的list；初始化后，变为字典，key是entity，values是其向量（使用narray）。
        self.rels_list = rels_list
        self.triplets_list = triplets_list
        self.margin = margin
        self.dim = dim
        self.normal_form = normal_form
        self.entity_vector_dict = {}
        self.rels_vector_dict = {}
        self.loss_list = []

    def initialize(self):
        """对论文中的初始化稍加改动
        初始化l和e，对于原本的l和e的文件中的/m/06rf7字符串标识转化为定义的dim维向量，对dim维向量进行uniform和norm归一化操作
        """
        entity_vector_dict, rels_vector_dict = {}, {}
        entity_vector_compo_list, rels_vector_compo_list = [], []
        for item, dict, compo_list, name in zip(
                [self.entity_list, self.rels_list], [entity_vector_dict, rels_vector_dict],
                [entity_vector_compo_list, rels_vector_compo_list], ["entity_vector_dict", "rels_vector_dict"]):
            for entity_or_rel in item:
                n = 0
                compo_list = []
                while n < self.dim:
                    random = uniform(-6 / (self.dim ** 0.5), 6 / (self.dim ** 0.5))
                    compo_list.append(random)
                    n += 1
                compo_list = norm(compo_list)
                dict[entity_or_rel] = compo_list
            print("The " + name + "'s initialization is over. It's number is %d." % len(dict))
        self.entity_vector_dict = entity_vector_dict
        self.rels_vector_dict = rels_vector_dict

    def transE(self, cycle_index=20):
        print("\n********** Start TransE training **********")
        for i in range(cycle_index):

            if i % 100 == 0:
                print("----------------The {} batchs----------------".format(i))
                print("The loss is: %.4f" % self.loss)
                # 查看最后的结果收敛情况
                self.loss_list.append(self.loss)
                # self.write_vector("data/entityVector.txt", "entity")
                # self.write_vector("data/relationVector.txt", "rels")
                self.loss = 0

            Sbatch = self.sample(150)
            Tbatch = []  # 元组对（原三元组，打碎的三元组）的列表 ：{((h,r,t),(h',r,t'))}
            for sbatch in Sbatch:
                triplets_with_corrupted_triplets = (sbatch, self.get_corrupted_triplets(sbatch))
                if triplets_with_corrupted_triplets not in Tbatch:
                    Tbatch.append(triplets_with_corrupted_triplets)
            self.update(Tbatch)

    def sample(self, size):
        return sample(self.triplets_list, size)

    def get_corrupted_triplets(self, triplets):
        '''training triplets with either the head or tail replaced by a random entity (but not both at the same time)
        :param triplet:单个（h,t,l）
        :return corruptedTriplet:'''
        # i = uniform(-1, 1) if i
        coin = choice([True, False])
        # 由于这个时候的(h,t,l)是从train文件里面抽出来的，要打坏的话直接随机寻找一个和头实体不等的实体即可
        if coin:  # 抛硬币 为真 打破头实体，即第一项
            while True:
                searching_entity = sample(self.entity_vector_dict.keys(), 1)[0]  # 取第一个元素是因为sample返回的是一个列表类型
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
        entity_vector_copy = deepcopy(self.entity_vector_dict)
        rels_vector_copy = deepcopy(self.rels_vector_dict)

        for triplets_with_corrupted_triplets in Tbatch:
            head_entity_vector = entity_vector_copy[triplets_with_corrupted_triplets[0][0]]
            tail_entity_vector = entity_vector_copy[triplets_with_corrupted_triplets[0][1]]
            relation_vector = rels_vector_copy[triplets_with_corrupted_triplets[0][2]]

            head_entity_vector_with_corrupted_triplets = entity_vector_copy[triplets_with_corrupted_triplets[1][0]]
            tail_entity_vector_with_corrupted_triplets = entity_vector_copy[triplets_with_corrupted_triplets[1][1]]

            head_entity_vector_before_batch = self.entity_vector_dict[triplets_with_corrupted_triplets[0][0]]
            tail_entity_vector_before_batch = self.entity_vector_dict[triplets_with_corrupted_triplets[0][1]]
            relation_vector_before_batch = self.rels_vector_dict[triplets_with_corrupted_triplets[0][2]]

            head_entity_vector_with_corrupted_triplets_before_batch = self.entity_vector_dict[
                triplets_with_corrupted_triplets[1][0]]
            tail_entity_vector_with_corrupted_triplets_before_batch = self.entity_vector_dict[
                triplets_with_corrupted_triplets[1][1]]

            if self.normal_form == "L1":
                dist_triplets = dist_L1(head_entity_vector_before_batch, tail_entity_vector_before_batch,
                                        relation_vector_before_batch)
                dist_corrupted_triplets = dist_L1(head_entity_vector_with_corrupted_triplets_before_batch,
                                                  tail_entity_vector_with_corrupted_triplets_before_batch,
                                                  relation_vector_before_batch)
            else:
                dist_triplets = dist_L2(head_entity_vector_before_batch, tail_entity_vector_before_batch,
                                        relation_vector_before_batch)
                dist_corrupted_triplets = dist_L2(head_entity_vector_with_corrupted_triplets_before_batch,
                                                  tail_entity_vector_with_corrupted_triplets_before_batch,
                                                  relation_vector_before_batch)
            eg = self.margin + dist_triplets - dist_corrupted_triplets
            if eg > 0:  # 大于0取原值，小于0则置0.即合页损失函数margin-based ranking criterion
                self.loss += eg
                temp_positive = 2 * self.learning_rate * (
                        tail_entity_vector_before_batch - head_entity_vector_before_batch - relation_vector_before_batch)
                temp_negative = 2 * self.learning_rate * (
                        tail_entity_vector_with_corrupted_triplets_before_batch - head_entity_vector_with_corrupted_triplets_before_batch - relation_vector_before_batch)
                if self.normal_form == "L1":
                    temp_positive_L1 = [1 if temp_positive[i] >= 0 else -1 for i in range(self.dim)]
                    temp_negative_L1 = [1 if temp_negative[i] >= 0 else -1 for i in range(self.dim)]
                    temp_positive_L1 = [float(f) for f in temp_positive_L1]
                    temp_negative_L1 = [float(f) for f in temp_negative_L1]
                    temp_positive = np.array(temp_positive_L1) * self.learning_rate
                    temp_negative = np.array(temp_negative_L1) * self.learning_rate
                    # temp_positive = norm(temp_positive_L1) * self.learning_rate
                    # temp_negative = norm(temp_negative_L1) * self.learning_rate

                # 对损失函数的5个参数进行梯度下降， 随机体现在sample函数上
                head_entity_vector += temp_positive
                tail_entity_vector -= temp_positive
                relation_vector = relation_vector + temp_positive - temp_negative
                head_entity_vector_with_corrupted_triplets -= temp_negative
                tail_entity_vector_with_corrupted_triplets += temp_negative

                # 归一化刚才更新的向量，减少计算时间
                entity_vector_copy[triplets_with_corrupted_triplets[0][0]] = norm(head_entity_vector)
                entity_vector_copy[triplets_with_corrupted_triplets[0][1]] = norm(tail_entity_vector)
                rels_vector_copy[triplets_with_corrupted_triplets[0][2]] = norm(relation_vector)
                entity_vector_copy[triplets_with_corrupted_triplets[1][0]] = norm(
                    head_entity_vector_with_corrupted_triplets)
                entity_vector_copy[triplets_with_corrupted_triplets[1][1]] = norm(
                    tail_entity_vector_with_corrupted_triplets)

                # self.entity_vector_dict = deepcopy(entity_vector_copy)
                # self.rels_vector_dict = deepcopy(rels_vector_copy)
            self.entity_vector_dict = entity_vector_copy
            self.rels_vector_dict = rels_vector_copy

    def write_vector(self, file_path, option):
        if option.strip().startswith("entit"):
            print("Write entities vetor into file      : {}".format(file_path))
            # dyct = deepcopy(self.entity_vector_dict)
            dyct = self.entity_vector_dict
        if option.strip().startswith("rel"):
            print("Write relationships vector into file: {}".format(file_path))
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
                    # file.write(str(lyst[i]).split('.')[0] + '.' + str(lyst[i]).split('.')[1][:4] + "\n")
                else:
                    # file.write(str(lyst[i]).split('.')[0] + '.' + str(lyst[i]).split('.')[1][:4] + "\t")
                    file.write(str(int(lyst[i] * 10000) / 10000) + "    ")
                    if (i + 1) % num_of_col == 0 and i != 0:
                        file.write("\n")


if __name__ == "__main__":
    entity_file_path = "data/FB15k/entity2id.txt"
    num_of_entity, entity_list = get_details_of_entityOrRels_list(entity_file_path)
    rels_file_path = "data/FB15k/relation2id.txt"
    num_of_rels, rels_list = get_details_of_entityOrRels_list(rels_file_path)
    train_file_path = "data/FB15k/train.txt"
    num_of_triplets, triplets_list = get_details_of_triplets_list(train_file_path)

    transE = TransE(entity_list, rels_list, triplets_list, margin=1, dim=50)
    print("\nTransE is initializing...")
    transE.initialize()
    transE.transE(500000)
    print("********** End TransE training ***********\n")
    # 训练的批次并不一定是100的整数倍，将最后更新的向量写到文件
    transE.write_vector("data/entityVector.txt", "entity")
    transE.write_vector("data/relationVector.txt", "relationship")
    transE.write_loss("data/lossList_25cols.txt", 25)
    transE.write_loss("data/lossList_1cols.txt", 1)
