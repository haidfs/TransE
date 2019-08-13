import logging

import tensorflow as tf
import argparse
from TestDatasetTF import KnowledgeGraph
from TestModelTF import TransE
from TestTransEMpQueue import get_dict_from_vector_file


def main():
    parser = argparse.ArgumentParser(description='TransE')
    parser.add_argument('--data_dir', type=str, default=r'./data/FB15k/')
    parser.add_argument('--score_func', type=str, default='L1')
    parser.add_argument('--n_rank_calculator', type=int, default=24)
    args = parser.parse_args()
    print(args)
    kg = KnowledgeGraph(data_dir=args.data_dir)

    entity_vector_file = "data/entityVector.txt"
    entity_vector_dyct = get_dict_from_vector_file(entity_vector_file)
    relation_vector_file = "data/relationVector.txt"
    relation_vector_dyct = get_dict_from_vector_file(relation_vector_file)
    logging.info("********** Start Test **********")

    kge_model = TransE(
        kg=kg,
        score_func=args.score_func,
        n_rank_calculator=args.n_rank_calculator,
        entity_vector_dict=entity_vector_dyct,
        rels_vector_dict=relation_vector_dyct)

    gpu_config = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(gpu_options=gpu_config)
    with tf.Session(config=sess_config) as sess:
        print('-----Initializing tf graph-----')
        tf.global_variables_initializer().run()
        print('-----Initialization accomplished-----')
        kge_model.check_norm()
        kge_model.launch_evaluation(session=sess)


if __name__ == '__main__':
    main()