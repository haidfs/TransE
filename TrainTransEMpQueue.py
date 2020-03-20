# -*- coding: UTF-8 -*-
import numpy as np
from multiprocessing import Process, Queue
import logging
import timeit
from TrainTransESimple import TransE as TransESimple
from TrainTransESimple import norm, dist_L1, dist_L2
from TrainTransESimple import prepare_fb15k_train_data

LOG_FORMAT = "%(asctime)s - %(name)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

INITIAL_LEARNING_RATE = 0.01


class TransE(TransESimple):
    def __init__(
            self,
            entity_list,
            rels_list,
            triplets_list,
            margin=1,
            learing_rate=INITIAL_LEARNING_RATE,
            dim=100,
            normal_form="L1",
            batch_size=10000,
            n_generator=24):
        TransESimple.__init__(self, entity_list, rels_list, triplets_list, margin=margin, learing_rate=learing_rate,
                              dim=dim, normal_form=normal_form, batch_size=batch_size)
        self.n_generator = n_generator

    def generate_training_batch(
            self,
            sbatch_queue: Queue,
            tbatch_queue: Queue):
        while True:
            raw_batch = sbatch_queue.get()
            if raw_batch is None:
                return
            else:
                pos_triplet = raw_batch
                neg_triplet = self.get_corrupted_triplets(pos_triplet)
                pos_neg_triplets = (pos_triplet, neg_triplet)
                tbatch_queue.put(pos_neg_triplets)

    def launch_training(self):
        raw_batch_queue = Queue()
        training_batch_queue = Queue()
        for _ in range(self.n_generator):
            Process(
                target=self.generate_training_batch,
                kwargs={
                    'sbatch_queue': raw_batch_queue,
                    'tbatch_queue': training_batch_queue}).start()
        start = timeit.default_timer()
        Sbatch = self.sample(self.batch_size)
        n_batch = 0
        for raw_batch in Sbatch:
            raw_batch_queue.put(raw_batch)
            n_batch += 1
        for _ in range(self.n_generator):
            raw_batch_queue.put(None)
        epoch_loss = 0
        self.loss = 0
        for i in range(n_batch):
            batch_pos, batch_neg = training_batch_queue.get()
            self.update_part(batch_pos, batch_neg)
        epoch_loss += self.loss
        print("batch size %d, cost time %g s, loss on batch data is %g" % (
            n_batch, timeit.default_timer() - start, epoch_loss))

    def update_part(self, pos_triplet, neg_triplet):
        entity_vector_copy = self.entity_vector_dict
        rels_vector_copy = self.rels_vector_dict

        # 这里的h,t,r代表头实体向量、尾实体向量、关系向量，h2和t2代表论文中的h'和t'，即负例三元组中的头尾实体向量
        # Tbatch是元组对（原三元组，打碎的三元组）的列表
        # ：[((h,r,t),(h',r,t'))...]，这里由于data文件的原因是(h,t,r)
        h = entity_vector_copy[pos_triplet[0]]
        t = entity_vector_copy[pos_triplet[1]]
        r = rels_vector_copy[pos_triplet[2]]
        # 损坏三元组中的头实体向量与尾实体向量
        h2 = entity_vector_copy[neg_triplet[0]]
        t2 = entity_vector_copy[neg_triplet[1]]
        # 在这里原本定义了beforebatch，但是个人认为没有必要，这里已经进入到batch里面了，走的就是单个处理
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
            entity_vector_copy[pos_triplet[0]] = norm(h)
            entity_vector_copy[pos_triplet[1]] = norm(t)
            rels_vector_copy[pos_triplet[2]] = norm(r)
            entity_vector_copy[neg_triplet[0]] = norm(h2)
            entity_vector_copy[neg_triplet[1]] = norm(t2)

        self.entity_vector_dict = entity_vector_copy
        self.rels_vector_dict = rels_vector_copy


def main():
    entity_list, rels_list, train_triplets_list = prepare_fb15k_train_data()

    transE = TransE(
        entity_list,
        rels_list,
        train_triplets_list,
        margin=1,
        dim=100,
        learing_rate=0.003)
    logging.info("TransE is initializing...")
    for epoch in range(2000):
        print("Mp Queue TransE, After %d training epoch(s):\n" % epoch)
        transE.launch_training()
        if epoch % 100 == 0:
            transE.write_vector("data/entityVectorMpQueue.txt", "entity")
            transE.write_vector("data/relationVectorMpQueue.txt", "rels")
    logging.info("********** End TransE training ***********\n")


if __name__ == "__main__":
    main()