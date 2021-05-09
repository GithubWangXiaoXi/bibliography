from collections import Counter
# 查看所生成的样本类别分布，0和1样本比例9比1，属于类别不平衡数据
print(Counter(y))
# Counter({0: 900, 1: 100})

# 使用imlbearn库中上采样方法中的SMOTE接口 <https://blog.csdn.net/nlpuser/article/details/81265614>
from imblearn.over_sampling import SMOTE
# 定义SMOTE模型，random_state相当于随机数种子的作用
smo = SMOTE(random_state=42)
X_smo, y_smo = smo.fit_resample(X, y)

print(Counter(y_smo))
# Counter({0: 900, 1: 900})