import cv2
import numpy as np
from matplotlib import pyplot as plt

'''1、生成模拟数据'''
#构造 20 组笔试和面试成绩都分布在[95, 100)区间的数据对,A级
a = np.random.randint(95,100, (20, 2)).astype(np.float32)
#构造 20 组笔试和面试成绩都分布在[90, 95)区间的数据对,B级
b = np.random.randint(90,95, (20, 2)).astype(np.float32)
#两组数据合并
data = np.vstack((a,b))
data = np.array(data,dtype='float32')

'''2、构造分组标签'''
#A级标签0
aLabel=np.zeros((20,1))
#B级标签1
bLabel=np.ones((20,1))
#上述标签合并
label = np.vstack((aLabel, bLabel))
label = np.array(label,dtype='int32')

'''3、训练'''
svm = cv2.ml.SVM_create()
result = svm.train(data,cv2.ml.ROW_SAMPLE,label)

'''4、分类'''
test = np.vstack([[97,97],[98,90],[85,85]])
test = np.array(test,dtype='float32')
test_p = svm.predict(test)

'''5、显示分类结果'''
plt.scatter(a[:,0], a[:,1], 80, 'g', 'o')
plt.scatter(b[:,0], b[:,1], 80, 'b', 's')
plt.scatter(test[:,0], test[:,1], 80, 'r', '*')
plt.show()

print("测试样本:",test)
print("预测标签:{}".format(test_p))