该工程代码主要是实现自己阅读过的和知识图谱相关的经典算法的代码：
1.TransE是知识图谱中知识表示的经典算法，工程实现了训练代码（多进程通信版）和测试代码
后续如继续进行论文阅读会补充相应的代码
2.由于data文件过大，无法上传，请至https://github.com/thunlp/KB2E下载data.zip并解压至工程的data路径
3.TransE论文地址： https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf
###训练部分
####Simple版本
./train_fb15k.sh 0
仅仅使用Python完成对应的训练代码
####Manager版本
./train_fb15k.sh 1
将TransE类的实例在多进程之间传递
####Queue版本
./train_fb15k.sh 2
将TransE类的训练数据传入队列，减小进程开销，加快训练速度
当训练完成之后，再进行测试
###测试部分
####TestTransEMqQueue
python TestTransEMpQueue.py
多进程队列测试加速，效果不明显，单个测试例0.5s，测试结束需要近5h。
####TestMainTF
 python TestMainTF.py
tf与多进程测试加速，效果显著，测试结束仅需要8min左右。
###最终测试结果
 	            FB15k
epochs:2000		MeanRank		Hits@10
			raw	     filter		raw	   filter
head			320.743	     192.152		29.7	41.2
tail			236.984	     153.431		36.1	46.2
average			278.863	     172.792		32.9	43.7
paper			243	     125	 	34.9	47.1
