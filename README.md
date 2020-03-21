该工程代码主要是实现自己阅读过的和知识图谱相关的经典算法的代码：  
1.TransE是知识图谱中知识表示的经典算法，工程实现了训练代码（多进程通信版）和测试代码  
后续如继续进行论文阅读会补充相应的代码  
2.TransE论文地址： https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf  
3.TransE SGD解释与代码简单解释： https://blog.csdn.net/weixin_42348333/article/details/89598144  

### 训练部分
#### Simple版本
./train_fb15k.sh 0
仅仅使用Python完成对应的训练代码
#### Manager版本
./train_fb15k.sh 1
将TransE类的实例在多进程之间传递
#### Queue版本
./train_fb15k.sh 2
将TransE类的训练数据传入队列，减小进程开销，加快训练速度  
### 注意事项
1. 当训练完成之后，再进行测试。  
2. 测试代码需要在Linux环境执行，Windows环境多进程速度慢，且多进程有bug！！！
### 测试部分
#### TestMainTF
 python TestMainTF.py
tf与多进程测试加速，效果显著，Linux环境128G服务器，测试结束仅需要8min左右。
### 最终测试结果
![image](https://github.com/haidfs/TransE/blob/master/images/TestResult.png)
### THANKS TO
感谢两位前辈的代码，基本是在他们的基础上学习整理  
https://github.com/wuxiyu/transE  
https://github.com/ZichaoHuang/TransE

