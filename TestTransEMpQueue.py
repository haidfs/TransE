from numpy import *
import operator
import logging
from TrainTransESimple import get_details_of_triplets_list
from multiprocessing import Queue, JoinableQueue, Process
import timeit

LOG_FORMAT = "%(asctime)s - %(name)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)


class Test:
    '''基本的评价过程
假设整个知识库中一共有n个实体，那么评价过程如下：
对于每一个测试的三元组a中的头实体或者尾实体，依次替换为整个知识库中的所有其它实体，也就是会产生n个三元组。
分别对上述n个三元组计算其能量值(dist值)，在transE中，就是计算h+r-t的值。这样可以得到n个能量值，分别对应上述n个三元组。
对上述n个能量值进行升序排序。
记录原本的三元组a的能量值排序后的序号。
对所有处在测试集中的测试三元组重复上述过程。
每个正确三元组的能量值排序后的序号求平均，得到的值我们称为Mean Rank。
计算正确三元组的能量排序后的序号小于10的比例，得到的值我们称为Hits@10。
上述就是评价的过程，共有两个指标：Mean Rank和Hits@10。其中Mean Rank越小越好，Hits@10越大越好。该代码未计算Hits@10，且Python对于这种大量计算速度很慢。
建议读者后续使用清华大学库的Fast_TransX代码，使用C++编写，性能高，能够快速得出训练和测试结果。
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
            else:  # 根据标签替换头实体或者替换尾实体计算距离
                corrupted_triplet = (triplet[0], ent, triplet[2])
                if self.is_fit and (
                        corrupted_triplet in self.train_triple_list):
                    continue
                rank_dyct[ent] = distance(self.entity_dyct[triplet[0]], self.entity_dyct[ent],
                                          self.relation_dyct[triplet[2]])
        sorted_rank = sorted(rank_dyct.items(),
                             key=operator.itemgetter(1))  # 按照元素的第一个域进行升序排序
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
        # 这里的vector使用[1:-1]是因为vector是'[0.11,0.22,..]'这样的str类型，[1:-1]是为了去掉列表的中括号
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