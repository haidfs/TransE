# -*- coding: UTF-8 -*-
import timeit
from TrainTransESimple import prepare_fb15k_train_data
from TrainTransESimple import TransE
from TrainTransEMpQueue import TransE as fastTransE
from TrainTransEMpManager import TransE as managerTransE
import argparse
from TrainTransEMpManager import Manager2, func1, MyManager

import logging
from multiprocessing import Process, Lock

LOG_FORMAT = "%(asctime)s - %(name)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)


def main():
    parser = argparse.ArgumentParser(description='TransE')
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--margin_value', type=float, default=1.0)
    parser.add_argument('--normal_form', type=str, default='L1')
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--learning_rate', type=float, default=0.003)
    parser.add_argument('--n_generator', type=int, default=24)
    parser.add_argument('--max_epoch', type=int, default=2000)
    parser.add_argument('--multi_process', type=str, default="MpQueue")
    args = parser.parse_args()
    print(args)
    entity_list, rels_list, train_triplets_list = prepare_fb15k_train_data()
    logging.info("********** Start TransE training ***********")

    if args.multi_process == "Simple":
        transE = TransE(
            entity_list,
            rels_list,
            train_triplets_list,
            margin=args.margin_value,
            dim=args.embedding_dim,
            learing_rate=args.learning_rate,
            normal_form=args.normal_form,
            batch_size=args.batch_size)
        logging.info("TransE is initializing...")
        transE.transE(args.max_epoch)
    elif args.multi_process == "MpQueue":
        transE = fastTransE(
            entity_list,
            rels_list,
            train_triplets_list,
            margin=args.margin_value,
            dim=args.embedding_dim,
            learing_rate=args.learning_rate,
            normal_form=args.normal_form,
            batch_size=args.batch_size,
            n_generator=args.n_generator)
        logging.info("TransE is initializing...")
        for epoch in range(args.max_epoch):
            logging.info(
                "Mp Queue TransE: After %d training epoch(s):" %
                epoch)
            transE.launch_training()
    else:
        MyManager.register('managerTransE', managerTransE)
        manager = Manager2()

        transE = manager.managerTransE(
            entity_list,
            rels_list,
            train_triplets_list,
            batch_size=args.batch_size,
            learing_rate=args.learning_rate,
            margin=1,
            dim=50,
            normal_form=args.normal_form)
        logging.info("TransE is initializing...")
        start = timeit.default_timer()
        for i in range(args.max_epoch):  # epoch�Ĵ���
            lock = Lock()
            proces = [Process(target=func1, args=(transE, lock)) for j in range(10)]  # 10������̣��������У����Ի�ܿ�
            for p in proces:
                p.start()
            for p in proces:
                p.join()
            end = timeit.default_timer()
            logging.info(
                "Mp Manager TransE: After %d training epoch(s):\nbatch size %d, cost time %g s, loss on batch data is %g"
                % (i, 10000, end - start, transE.get_loss()))
            start = end
            transE.clear_loss()
    logging.info("********** End TransE training ***********\n")
    # ѵ�������β���һ����100�����������������µ�����д���ļ�
    transE.write_vector("data/entityVector.txt", "entity")
    transE.write_vector("data/relationVector.txt", "relationship")


if __name__ == '__main__':
    main()