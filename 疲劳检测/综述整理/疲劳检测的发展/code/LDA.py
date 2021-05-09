import numpy as np
import matplotlib.pyplot as plt

'''LDA实现二分类'''
'''自己生成一个数据集：'''
def createDataSet():
	#类别1
    X1 = np.mat(np.random.random((8, 2)) * 5 + 15)
    #类别2
    X2 = np.mat(np.random.random((8, 2)) * 5 + 2)
    return X1, X2

'''求各个属性的均值：'''
def average(dataset):
	ave = []
	a, b = np.shape(dataset)
	for i in range(b):
		n = np.sum(dataset[:,i]) / a
		ave.append(n)
	return np.array(ave)

'''求单一类的类内散度矩阵：'''
def compute_sw(dataset, ave):
	sw = 0
	a, b = np.shape(dataset)
	for i in range(a - 1):
		sw += np.dot(dataset[i,:] - ave, (dataset[i,:] - ave).T)
	return np.array(sw)

'''根据方向向量求直线方程 (X - x1)/w[0] = (Y - y1)/w[1]'''
def getLine(w,x1_x):
	'''
	:param w: 方向向量
	:param dot: 某一个样本
	:return: x,y
	'''
	x = np.linspace(0,30,2)
	print("x",x)

	dot_1 = x1_x[0,0]
	dot_2 = x1_x[0,1]

	y = w[1] * (x - dot_1 * 2) / (w[0] * 2) + (dot_2 * 2)
	return x,y

if __name__ == '__main__':
	x1_x, x2_x = createDataSet()
	x1_x_ave = average(x1_x)
	x2_x_ave = average(x2_x)
	print(x1_x)

	#因为有些矩阵是没有逆矩阵的，为了使任意数据集都有结果，我们使用广义逆矩阵：
	x1_sw = compute_sw(x1_x, x1_x_ave)
	x2_sw = compute_sw(x2_x, x2_x_ave)
	Sw = x1_sw + x2_sw

	# 求广义逆
	pinv = np.linalg.pinv(Sw)

	#求最佳方向w：w  = S_w^{−1}(μ1 - μ2)
	w = np.multiply(x1_x_ave - x2_x_ave, pinv)[0, :]
	print(w)

	plt.figure(figsize=(16,8))
	plt.subplot(121)
	plt.title("LDA前")
	plt.scatter(x = [x1_x[:,0]],y = [x1_x[:,1]],label="c1")
	plt.scatter(x = [x2_x[:,0]],y = [x2_x[:,1]],label="c2")
	plt.scatter(x = x1_x_ave[0], y = x1_x_ave[1],label="c1 ave")
	plt.scatter(x = x2_x_ave[0], y = x2_x_ave[1],label="c2 ave")
	plt.legend()

	plt.subplot(122)
	plt.title("LDA后")
	x,y = getLine(w,x1_x)
	plt.plot(x,y,label="LDA line")
	plt.scatter(x=[x1_x[:, 0]], y=[x1_x[:, 1]], label="c1")
	plt.scatter(x=[x2_x[:, 0]], y=[x2_x[:, 1]], label="c2")
	plt.scatter(x=x1_x_ave[0], y=x1_x_ave[1], label="c1 ave")
	plt.scatter(x=x2_x_ave[0], y=x2_x_ave[1], label="c2 ave")
	plt.legend()
	plt.show()