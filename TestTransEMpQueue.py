# -*- coding: UTF-8 -*-
from numpy import *
import operator
import logging
from TrainTransESimple import get_details_of_triplets_list
from multiprocessing import Queue, JoinableQueue, Process
import timeit

LOG_FORMAT = "%(asctime)s - %(name)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)


class Test:
    '''���������۹���
��������֪ʶ����һ����n��ʵ�壬��ô���۹������£�
����ÿһ�����Ե���Ԫ��a�е�ͷʵ�����βʵ�壬�����滻Ϊ����֪ʶ���е���������ʵ�壬Ҳ���ǻ����n����Ԫ�顣
�ֱ������n����Ԫ�����������ֵ(distֵ)����transE�У����Ǽ���h+r-t��ֵ���������Եõ�n������ֵ���ֱ��Ӧ����n����Ԫ�顣
������n������ֵ������������
��¼ԭ������Ԫ��a������ֵ��������š�
�����д��ڲ��Լ��еĲ�����Ԫ���ظ��������̡�
ÿ����ȷ��Ԫ�������ֵ�����������ƽ�����õ���ֵ���ǳ�ΪMean Rank��
������ȷ��Ԫ����������������С��10�ı������õ���ֵ���ǳ�ΪHits@10��
�����������۵Ĺ��̣���������ָ�꣺Mean Rank��Hits@10������Mean RankԽСԽ�ã�Hits@10Խ��Խ�á��ô���δ����Hits@10����Python�������ִ��������ٶȺ�����
������ߺ���ʹ���廪��ѧ���Fast_TransX���룬ʹ��C++��д�����ܸߣ��ܹ����ٵó�ѵ���Ͳ��Խ����
'''

    def __init__(self, entity_dyct, relation_dyct, train_triple_list,
                 test_triple_list,
                 label="head", is_fit=False, n_rank_calculator=24):
        self.entity_dyct = entity_dyct
        self.relation_dyct = relation_dyct
        self.train_triple_list = train_triple_list
        self.test_triple_list = test_triple_list
        self.rank = []
        self.label = label
        self.is_fit = is_fit
        self.hit_at_10 = 0
        self.count = 0
        self.n_rank_calculator = n_rank_calculator

    def write_rank(self, file_path):
        logging.info("Write int to %s" % file_path)
        file = open(file_path, 'w')
        for r in self.rank:
            file.write(str(r[0]) + "\t")
            file.write(str(r[1]) + "\t")
            file.write(str(r[2]) + "\t")
            file.write(str(r[3]) + "\n")
        file.close()

    def get_rank_part(self, triplet):
        rank_dyct = {}
        for ent in self.entity_dyct.keys():
            if self.label == "head":
                corrupted_triplet = (ent, triplet[1], triplet[2])
                if self.is_fit and (
                        corrupted_triplet in self.train_triple_list):
                    continue
                rank_dyct[ent] = distance(self.entity_dyct[ent], self.entity_dyct[triplet[1]],
                                          self.relation_dyct[triplet[2]])
            else:  # ���ݱ�ǩ�滻ͷʵ������滻βʵ��������
                corrupted_triplet = (triplet[0], ent, triplet[2])
                if self.is_fit and (
                        corrupted_triplet in self.train_triple_list):
                    continue
                rank_dyct[ent] = distance(self.entity_dyct[triplet[0]], self.entity_dyct[ent],
                                          self.relation_dyct[triplet[2]])
        sorted_rank = sorted(rank_dyct.items(),
                             key=operator.itemgetter(1))  # ����Ԫ�صĵ�һ���������������
        if self.label == 'head':
            num_tri = 0
        else:
            num_tri = 1
        ranking = 1
        for i in sorted_rank:
            if i[0] == triplet[num_tri]:
                break
            ranking += 1
        if ranking < 10:
            self.hit_at_10 += 1
        self.rank.append(
            (triplet, triplet[num_tri], sorted_rank[0][0], ranking))
        logging.info(
            "Count:{} triplet {} {} ranks {}".format(
                self.count, triplet, self.label, ranking))
        self.count += 1

    def get_relation_rank(self):
        count = 0
        self.rank = []
        self.hit_at_10 = 0
        for triplet in self.test_triple_list:
            rank_dyct = {}
            for rel in self.relation_dyct.keys():
                corrupted_triplet = (triplet[0], triplet[1], rel)
                if self.is_fit and (
                        corrupted_triplet in self.train_triple_list):
                    continue
                rank_dyct[rel] = distance(self.entity_dyct[triplet[0]], self.entity_dyct[triplet[1]],
                                          self.relation_dyct[rel])
            sorted_rank = sorted(rank_dyct.items(), key=operator.itemgetter(1))
            ranking = 1
            for i in sorted_rank:
                if i[0] == triplet[2]:
                    break
                ranking += 1
            if ranking < 10:
                self.hit_at_10 += 1
            self.rank.append((triplet, triplet[2], sorted_rank[0][0], ranking))
            logging.info(
                "Count:{} triplet {} relation ranks {}".format(
                    count, triplet, ranking))
            count += 1

    def get_mean_rank_and_hit(self):
        total_rank = 0
        for r in self.rank:
            total_rank += r[3]
        num = len(self.rank)
        return total_rank / num, self.hit_at_10 / num

    def calculate_rank(self, in_queue, out_queue):
        while True:
            test_triplet = in_queue.get()
            if test_triplet is None:
                in_queue.task_done()
                return
            else:
                out_queue.put(test_triplet)
                in_queue.task_done()

    def launch_test(self):
        eval_result_queue = JoinableQueue()
        rank_result_queue = Queue()
        print('-----Start evaluation-----')
        start = timeit.default_timer()
        for _ in range(self.n_rank_calculator):
            Process(
                target=self.calculate_rank,
                kwargs={
                    'in_queue': eval_result_queue,
                    'out_queue': rank_result_queue}).start()
        n_used_eval_triple = 0
        for test_triplet in self.test_triple_list:
            eval_result_queue.put(test_triplet)
            n_used_eval_triple += 1
        for _ in range(self.n_rank_calculator):
            eval_result_queue.put(None)
        print('-----Joining all rank calculator-----')
        # eval_result_queue.join()
        for i in range(n_used_eval_triple):
            test_triplet = rank_result_queue.get()
            self.get_rank_part(test_triplet)
        print('-----All rank calculation accomplished-----')
        print('-----Obtaining evaluation results-----')


def distance(h, t, r):
    h = array(h)
    t = array(t)
    r = array(r)
    s = h + r - t
    return linalg.norm(s)


def get_dict_from_vector_file(file_path):
    file = open(file_path)
    dyct = {}
    for line in file.readlines():
        name_vector = line.strip().split("\t")
        # �����vectorʹ��[1:-1]����Ϊvector��'[0.11,0.22,..]'������str���ͣ�[1:-1]��Ϊ��ȥ���б��������
        vector = [float(s) for s in name_vector[1][1:-1].split(", ")]
        name = name_vector[0]
        dyct[name] = vector
    return dyct


def main():
    train_file = "data/FB15k/train.txt"
    num_train_triple, train_triple_list = get_details_of_triplets_list(
        train_file)
    logging.info("Num of Train:%d" % num_train_triple)
    test_file = "data/FB15k/test.txt"
    num_test_triple, test_triple_list = get_details_of_triplets_list(test_file)
    logging.info("Num of Test:%d" % num_test_triple)
    entity_vector_file = "data/entityVector.txt"
    entity_vector_dyct = get_dict_from_vector_file(entity_vector_file)
    relation_vector_file = "data/relationVector.txt"
    relation_vector_dyct = get_dict_from_vector_file(relation_vector_file)
    logging.info("********** Start Test **********")

    test_head_raw = Test(
        entity_vector_dyct,
        relation_vector_dyct,
        train_triple_list,
        test_triple_list)
    test_head_raw.launch_test()

    logging.info(
        "=========== Test Head Raw MeanRank: %g Hits@10: %g ===========" %
        test_head_raw.get_mean_rank_and_hit())
    test_head_raw.write_rank("data/test/" + "test_head_raw" + ".txt")
    logging.info("********** End Test **********")


if __name__ == '__main__':
    main()