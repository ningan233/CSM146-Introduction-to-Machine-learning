1.先写一个关于data的class，用来导入和plot data
2.再写一个polynomial regression的class，其中有负责不同部分的member function
     generate feature用于形成一个n行（m+1）列的关于x的矩阵，矩阵X的每一行是是从x^0到x^m
     fit_GD 是用gradient descent的方法来fit training data，从而train一个model，得到这个model的W
                    （这一步规定不能用predict function的原因是，predict function此时还没有被定义）
                    （fit_GD的过程是，首先initialize W0，计算W0下的loss function和loss function对W的导数，利用W=W-eta*导数更新一次W0，用更新的W0再次计算loss function和导数，比较两次loss function的差值看是否需要停止循环）
                    （initialize W0 ----> 计算loss和导数 ----> 更新W0 ----> 计算loss和导数 ---->更新W0      一直到两次loss值的差值小于eps，循环终止）
     fit 是用closed-form的方法来通过training data计算W 
     predict是用来计算y_pred(=XW)
     cost是在得到最终的W之后，通过调用predict来计算在最终的W下，loss function的值

3.实体化一个model=polynomialRegressison（m=？）
  再用training data去fit这个model（可以用gradient descent的方法去fit model，也可以用closed-form去fit model来得到最终的W）
  计算cost（计算用时）（看以前的code，今年code删除了得到具体run time的code）

4.RMSE的优点？

5.最后一张图不知道哪里有问题一直搞不对orz
  不能用testing data去fit model！！！是用training data去train model之后直接调用rmse这个function来求training error和testing error

  