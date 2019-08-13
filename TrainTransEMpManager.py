from multiprocessing import Process, Lock
from multiprocessing.managers import BaseManager
import logging
from TrainTransESimple import TransE as TransESimple
from TrainTransESimple import prepare_fb15k_train_data

LOG_FORMAT = "%(asctime)s - %(name)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

INITIAL_LEARNING_RATE = 0.01


class TransE(TransESimple):

    def get_loss(self):
        # 参考清华的Fast-TransX的C++代码，确实速度很快，Python接近10个小时的训练C++大概在十几分钟即可完成。粗略的看了一下代码，
        # 它对原本的论文中的Sbatch做了修改，直接进行了（总数量为训练三元组数，一个epoch分为多个batch完成，每个batch的每一个三元组都随机采样），随机梯度下降。多线程并发，n个batch对应n个线程
        # Python由于历史遗留问题，使用了GIL，全局解释锁，使得Python的多线程近似鸡肋，无法跑满多核cpu，所以考虑使用多进程优化
        # 为了使用多进程，使用了manager将transE封装为Proxy对象。由于Proxy对象无法获取封装的TransE类的属性，所以需要写get函数将loss传出。
        # 另外值得注意的是，Python的多进程性能不一定优于for循环。基本开销就包括了进程的创建和销毁、上下文切换（进程间需要RPC远程通信以做到类变量共享）。
        # 至少在trainTransE和trainTransE_MultiProcess对比来看，trainTransE的for循环一批10个耗时在8s-9s，trainTransE_MultiProcess的一个epoch即一批，耗时在12-13s。
        # 进一步优化方法：进程池，实现进程复用？框架：tf？？
        return self.loss

    def clear_loss(self):
        # 该函数也是为了Proxy对象外部将损失置0
        self.loss = 0

    def transE(self):
        Sbatch = self.sample(self.batch_size // 10)
        Tbatch = []  # 元组对（原三元组，打碎的三元组）的列表 ：{((h,r,t),(h',r,t'))}
        for sbatch in Sbatch:
            pos_neg_triplets = (sbatch, self.get_corrupted_triplets(sbatch))
            if pos_neg_triplets not in Tbatch:
                Tbatch.append(pos_neg_triplets)
        self.update(Tbatch)


class MyManager(BaseManager):
    pass


def Manager2():
    m = MyManager()
    m.start()
    return m


MyManager.register('TransE', TransE)


def func1(em, lock):
    with lock:
        em.transE()


def main():
    manager = Manager2()
    entity_list, rels_list, train_triplets_list = prepare_fb15k_train_data()

    transE = manager.TransE(
        entity_list,
        rels_list,
        train_triplets_list,
        batch_size=10000,
        margin=1,
        dim=50)
    logging.info("TransE is initializing...")
    for i in range(20000):  # epoch的次数
        lock = Lock()
        proces = [Process(target=func1, args=(transE, lock))
                  for j in range(10)]  # 10个多进程，谨慎运行，电脑会很卡
        for p in proces:
            p.start()
        for p in proces:
            p.join()
        if i != 0:
            logging.info(
                "After %d training epoch(s), loss on batch data is %g" %
                (i * 10, transE.get_loss()))
        transE.clear_loss()
    # transE.transE(100000)
    logging.info("********** End TransE training ***********\n")


if __name__ == "__main__":
    main()