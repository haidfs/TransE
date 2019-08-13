import timeit
import numpy as np
import tensorflow as tf
import multiprocessing as mp
from TestDatasetTF import KnowledgeGraph


class TransE:
    def __init__(self, kg: KnowledgeGraph,
                 score_func,
                 n_rank_calculator, entity_vector_dict, rels_vector_dict):
        self.kg = kg
        self.score_func = score_func
        self.n_rank_calculator = n_rank_calculator

        self.entity_vector_dict = entity_vector_dict
        self.rels_vector_dict = rels_vector_dict
        self.entity_embedding = None
        self.relation_embedding = None

        '''ops for evaluation'''
        self.eval_triple = tf.placeholder(dtype=tf.int32, shape=[3])
        self.idx_head_prediction = None
        self.idx_tail_prediction = None
        self.build_entity_embedding()
        self.build_eval_graph()

    def build_entity_embedding(self):
        self.entity_embedding = np.array(
            list(self.entity_vector_dict.values()))
        self.relation_embedding = np.array(
            list(self.rels_vector_dict.values()))

    def build_eval_graph(self):
        with tf.name_scope('evaluation'):
            self.idx_head_prediction, self.idx_tail_prediction = self.evaluate(
                self.eval_triple)

    def evaluate(self, eval_triple):
        with tf.name_scope('lookup'):
            head = tf.nn.embedding_lookup(
                self.entity_embedding, eval_triple[0])
            tail = tf.nn.embedding_lookup(
                self.entity_embedding, eval_triple[1])
            relation = tf.nn.embedding_lookup(
                self.relation_embedding, eval_triple[2])
        with tf.name_scope('link'):
            # 并不太明确这里的用途，h,r,t应该都是[1,dim]维度的向量， self.entity_embedding应该是[n,dim]维度的向量，做加减法得到的是什么类型？
            # 如果是list类型，对于不同维度是不能直接加减的。但是对于np.array或者tf的embedding，是可以直接相减的，等同于 self.entity_embedding
            # 的每一行都在和h,r,t做运算
            distance_head_prediction = self.entity_embedding + relation - tail
            distance_tail_prediction = head + relation - self.entity_embedding
        with tf.name_scope('rank'):
            if self.score_func == 'L1':  # L1 score
                _, idx_head_prediction = tf.nn.top_k(tf.reduce_sum(
                    tf.abs(distance_head_prediction), axis=1), k=self.kg.n_entity)
                _, idx_tail_prediction = tf.nn.top_k(tf.reduce_sum(
                    tf.abs(distance_tail_prediction), axis=1), k=self.kg.n_entity)
            else:  # L2 score
                _, idx_head_prediction = tf.nn.top_k(tf.reduce_sum(
                    tf.square(distance_head_prediction), axis=1), k=self.kg.n_entity)
                _, idx_tail_prediction = tf.nn.top_k(tf.reduce_sum(
                    tf.square(distance_tail_prediction), axis=1), k=self.kg.n_entity)
        return idx_head_prediction, idx_tail_prediction

    def launch_evaluation(self, session):
        eval_result_queue = mp.JoinableQueue()
        rank_result_queue = mp.Queue()
        print('-----Start evaluation-----')
        start = timeit.default_timer()
        for _ in range(self.n_rank_calculator):
            mp.Process(
                target=self.calculate_rank,
                kwargs={
                    'in_queue': eval_result_queue,
                    'out_queue': rank_result_queue}).start()
        n_used_eval_triple = 0
        for eval_triple in self.kg.test_triples:
            idx_head_prediction, idx_tail_prediction = session.run(
                fetches=[
                    self.idx_head_prediction, self.idx_tail_prediction], feed_dict={
                    self.eval_triple: eval_triple})
            eval_result_queue.put((eval_triple, idx_head_prediction, idx_tail_prediction))
            n_used_eval_triple += 1
            print(
                '[{:.3f}s] #evaluation triple: {}/{}'.format(
                    timeit.default_timer() - start,
                    n_used_eval_triple,
                    self.kg.n_test_triple),
                end='\r')
        print()
        for _ in range(self.n_rank_calculator):
            eval_result_queue.put(None)
        print('-----Joining all rank calculator-----')
        eval_result_queue.join()
        print('-----All rank calculation accomplished-----')
        print('-----Obtaining evaluation results-----')
        '''Raw'''
        head_meanrank_raw = 0
        head_hits10_raw = 0
        tail_meanrank_raw = 0
        tail_hits10_raw = 0
        '''Filter'''
        head_meanrank_filter = 0
        head_hits10_filter = 0
        tail_meanrank_filter = 0
        tail_hits10_filter = 0
        for _ in range(n_used_eval_triple):
            head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter = rank_result_queue.get()
            head_meanrank_raw += head_rank_raw
            if head_rank_raw < 10:
                head_hits10_raw += 1
            tail_meanrank_raw += tail_rank_raw
            if tail_rank_raw < 10:
                tail_hits10_raw += 1
            head_meanrank_filter += head_rank_filter
            if head_rank_filter < 10:
                head_hits10_filter += 1
            tail_meanrank_filter += tail_rank_filter
            if tail_rank_filter < 10:
                tail_hits10_filter += 1
        print('-----Raw-----')
        head_meanrank_raw /= n_used_eval_triple
        head_hits10_raw /= n_used_eval_triple
        tail_meanrank_raw /= n_used_eval_triple
        tail_hits10_raw /= n_used_eval_triple
        print('-----Head prediction-----')
        print(
            'MeanRank: {:.3f}, Hits@10: {:.3f}'.format(
                head_meanrank_raw,
                head_hits10_raw))
        print('-----Tail prediction-----')
        print(
            'MeanRank: {:.3f}, Hits@10: {:.3f}'.format(
                tail_meanrank_raw,
                tail_hits10_raw))
        print('------Average------')
        print(
            'MeanRank: {:.3f}, Hits@10: {:.3f}'.format(
                (head_meanrank_raw + tail_meanrank_raw) / 2,
                (head_hits10_raw + tail_hits10_raw) / 2))
        print('-----Filter-----')
        head_meanrank_filter /= n_used_eval_triple
        head_hits10_filter /= n_used_eval_triple
        tail_meanrank_filter /= n_used_eval_triple
        tail_hits10_filter /= n_used_eval_triple
        print('-----Head prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(
            head_meanrank_filter, head_hits10_filter))
        print('-----Tail prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(
            tail_meanrank_filter, tail_hits10_filter))
        print('-----Average-----')
        print(
            'MeanRank: {:.3f}, Hits@10: {:.3f}'.format(
                (head_meanrank_filter + tail_meanrank_filter) / 2,
                (head_hits10_filter + tail_hits10_filter) / 2))
        print('cost time: {:.3f}s'.format(timeit.default_timer() - start))
        print('-----Finish evaluation-----')

    def calculate_rank(self, in_queue, out_queue):
        while True:
            idx_predictions = in_queue.get()
            if idx_predictions is None:
                in_queue.task_done()
                return
            else:
                eval_triple, idx_head_prediction, idx_tail_prediction = idx_predictions
                head, tail, relation = eval_triple
                head_rank_raw = 0
                tail_rank_raw = 0
                head_rank_filter = 0
                tail_rank_filter = 0
                for candidate in idx_head_prediction[::-1]:
                    if candidate == head:
                        break
                    else:
                        head_rank_raw += 1
                        if (candidate, tail,
                                relation) in self.kg.golden_triple_pool:
                            continue
                        else:
                            head_rank_filter += 1
                for candidate in idx_tail_prediction[::-1]:
                    if candidate == tail:
                        break
                    else:
                        tail_rank_raw += 1
                        if (head, candidate,
                                relation) in self.kg.golden_triple_pool:
                            continue
                        else:
                            tail_rank_filter += 1
                out_queue.put(
                    (head_rank_raw,
                     tail_rank_raw,
                     head_rank_filter,
                     tail_rank_filter))
                in_queue.task_done()

    def check_norm(self):
        print('-----Check norm-----')
        entity_embedding = self.entity_embedding
        relation_embedding = self.relation_embedding
        entity_norm = np.linalg.norm(entity_embedding, ord=2, axis=1)
        relation_norm = np.linalg.norm(relation_embedding, ord=2, axis=1)
        # print('entity norm: {} relation norm: {}'.format(entity_norm, relation_norm))