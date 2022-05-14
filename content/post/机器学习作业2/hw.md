---
title: "机器学习作业2"
description: "大三下学期机器学习课第二次作业"
date: "2022-05-04 21:37:00+0200"
slug: "机器学习作业2"
image: "image/seele.png"
categories:
    - 作业
tags:
    - 机器学习
    - 作业
math: true
markup: mmmark
---

**1.说明线性模型得原理和实现过程，涉及公式推导。**
假定给定数据集$D=\{(x_i,y_i)\}_{i=1}^m,x_i \in \mathbb{R}^N,y_i \in \mathbb{R}$,有$f(x_i)=w^Tx_i,w^T \in \mathbb{R}^m$.$f(x_i)$的值与$y_i$尽可能接近，我们称其为线性模型。$w^T$作为一个线性算子，对输入样本空间进行线性变换，将其映射到类别空间上，要求相似的样本在空间中距离更近，被分为一类。所以对于线性模型我们要求其输入样本在样本空间的分布是线性可分的。
对于线性模型来说，我们一般会有一个性能评估指标，如线性回归的均方误差，对数几率回归的对数似然，线性判别分析的广义瑞利商。这些性能评价指标就是我们模型的优化指标，我们要选择模型的参数使这些优化指标达到最优。在限制了优化指标的最优条件时，我们就可以通过数学优化的方法求得线性模型的参数。

**2.解释线性判别分析的原理，算法优缺点，可加图说明。**
线性判别分析( Linear Discriminant Analysis,简称LDA)是一种经典的线性学习方法，它的基本思想是将样本投影到一条直线上，使同类样本尽可能近，不同类样本之间的距离尽可能远。前者对应同类样本的协方差尽可能小，后者对应不同样本类别的投影中心的距离尽可能大。
这里先定义几个符号：假设数据集为$D\{(x_i,y_i)\}_{i=1}^m,y_i \in \{0,1\},x_i \in \mathbb{R}^m,m \in \mathbb{N}^+$那么有$X_i,\mu_i,\Sigma_i$分别表示$i \in \{0,1\}$类样本的集合，均值向量，协方差矩阵,投影方向为$w$那么我们的目的就变为让$w^T\Sigma_0w+w^T\Sigma_1w$尽可能小，$||w^T\mu_0-w^T\mu_1 ||_{2}^{2}$尽可能大。整合一下目标变为$J=\frac{w^T(\mu_0-\mu_1)(\mu_0-\mu_1)^Tw}{w^T(\Sigma_0+\Sigma_1)w}$最大化。
如果我们定义$S_w=\Sigma_0+\Sigma_1$为类内散度矩阵，$S_b=(\mu_0-\mu_1)(\mu_0-\mu_1)^T$为类间散度矩阵，则可变换到$J=\frac{w^TS_bw}{w_TS_ww}$，这里的J为广义瑞利商(在第一题中提过)。
对于广义瑞利商，其优化目标为：
$$
\begin{aligned}
\underset{\pmb{w}}{min} \quad &-\pmb{w}^T\pmb{b}_b\pmb{w}\\
s.t. \quad &\pmb{w}^T\pmb{S}_w\pmb{w}=1
\end{aligned}
$$
由拉格朗日乘子法可以求得$\pmb{S}_b\pmb{w}=\lambda\pmb{S}_w\pmb{w}$, $\pmb{S}_b\pmb{w}$的方向恒为$\pmb{\mu}_0-\pmb{\mu}_1$,则$\pmb{w}=\pmb{S}_w^{-1}(\pmb{\mu}_0-\pmb{\mu}_1)$。之后利用$\pmb{w}^T\pmb{\mu}$对特征向量进行投影即可。
优点：线性判别分析是有有监督的，能很好得反应样本间差异
缺点：局限性大，受样本种类限制，投影空间最多为n-1维。

**3.以前三层神经网络为例，说明误差逆传播算法的计算原理。**

假设有一个三层神经网络，如下图:
![](image/MLhw/1.jpg)

对于训练样例$(\pmb{x}_k,\pmb{y}_k)$,假定输出为$\pmb{\hat{y}_k}=(\hat{y}_1^k,\hat{y}_2^k,\cdots,\hat{y}_l^k)$,即$\hat{\pmb{y}}_j^k=f(\beta_j-\theta_j)$,假设使用均方误差作为评估准则，$E_k=\frac{1}{2}\sum_{j=1}^l(\hat{y}_j^k-y_j^k)^2$。
对于误差，我们知道其在相对于各参数负梯度方向上下降最快，所以接下来我们要求误差相对于各参数的梯度，为简化计算，我们使用$w_{hj},\theta_j,\upsilon_{ih},\gamma_h$举例计算，其中$\theta_j,\gamma_h$分别代表输出层第$j$个神经元，隐含第$h$个神经元的阈值，也可以叫偏置。
依据链式法则：
$$
\begin{aligned}
&\Delta w_{hj}=-\eta\frac{\partial E_k}{\partial w_{hj}}=-\eta\frac{\partial E_k}{\partial \hat{y_k}}\frac{\partial  \hat{y_k}}{\partial \beta_j}\frac{\partial  \beta_j}{\partial w_{hj}}\\
&\Delta \theta_j=-\eta \frac{\partial E_k}{\partial \theta_j}=-\eta\frac{\partial E_k}{\partial \hat{y_k}}\frac{\partial  \hat{y_k}}{\partial \beta_j}\frac{\partial  \beta_j}{\partial \theta_j}\\
&\Delta \upsilon_{ih}=-\eta \frac{\partial E_k}{\partial b_h}\frac{b_h}{\alpha_h}\frac{\alpha_h}{\upsilon_{ih}}\\
&\Delta \gamma_h=-\eta \frac{\partial E_k}{\partial b_h}\frac{b_h}{\alpha_h}\frac{\alpha_h}{\gamma_h}\\
&\frac{\partial E_k}{\partial b_h}= \sum_{j=1}^l\frac{\partial E_k}{\partial \beta_j}\frac{\beta_j}{\partial b_h}
\end{aligned}
$$
假设$\upsilon$为任意参数，则参数更新方式为$\upsilon\leftarrow\upsilon+\Delta \upsilon$按照以上公式更新模型参数，模型达到评估要求后，即可进行预测。

**4.解释支持向量机的原理和实现过程，优缺点，适用情况，不同参数的作用，说明自己的学习感悟**
从集合角度看，支持向量机是找一个对于线性可分的数据集正负样本都远的超平面。
我们用$\pmb{w}^T\pmb{x}+b=0$,其中$\pmb{w}=(w_1;w_2;...;w_d)$为超平面法向量，决定了超平面的方向，b为位移项，决定了超平面和原点之间的距离。
样本空间中任意一点到超平面的距离表示为：
$$
\gamma=\frac{|\pmb{w}^T\pmb{x}+b|}{||\pmb{w}||}
$$
使用$
\frac{y_{min}(\pmb{w}^T\pmb{x}_{min}+b)}{||\pmb{x}||}
$表示数据集划分的几何距离。
那么解优化问题：
$$
\begin{aligned}
    \max_{\pmb{w},b}\quad&\frac{y_{min}(\pmb{w}^T\pmb{x}_{min}+b)}{||\pmb{w}||}\\
s.t.\quad&y_i(\pmb{w}^T\pmb{x}_i+b)\geqslant y_{min}(\pmb{w}^T\pmb{x}_{min}+b)
\end{aligned}
$$
为使其解唯一，限制$\frac{y_{min}(\pmb{w}^T\pmb{x}_{min}+b)}{||\pmb{x}||}=1$，得：
$$
\begin{aligned}
    \max_{\pmb{w},b}\quad&\frac{1}{||\pmb{w}||}\\
s.t.\quad&y_i(\pmb{w}^T\pmb{x}_i+b)\geqslant 1
\end{aligned}
$$
解这个优化问题得$\pmb{w}$和$b$,确定超平面。（真不会解，解法空着了）
使用：
$$
y=sign(\pmb{w}^T+b) 
$$
即可对样本进行分类。
优点：支持向量机是一种小样本学习方法，只有少数几个样本起作用，避免了维数灾难，鲁棒性好。
缺点：对于分线性可分得数据集需要借助核函数，核函数的选择是一个未解决的问题。对于大规模训练耗时大，解决多分类问题存在困难。


**5.在夏季，某公园男性穿凉鞋的概率为$\frac{2}{5}$,女性穿凉鞋的概率为$\frac{4}{5}$，并且该公园中男女比例通常为2:1,问题：若你在公园中随机遇到一个穿凉鞋的人，请问他的性别为男性或女性的概率分别为多少，解释计算原理。**

贝叶斯公式：
$$
P(A_i|B)=\frac{P(B|A_i)P(A_i)}{\sum_j P(B|A_j)P(Aj)}
$$
假设事件A表示遇到的人是男或女,事件B表示遇到的人穿凉鞋。
则依以上公式有：
$$
\begin{aligned}
P(男|穿)&=\frac{P(穿|男)P(男)}{P(穿|男)P(男)+P(穿|女)P(女)}\\
&=\frac{\frac{2}{5}\times \frac{2}{3}}{\frac{2}{5}\times\frac{2}{3}+\frac{4}{5}\times\frac{1}{3}}=\frac{1}{2}=0.5\\
P(女|穿)&=1-P(男|穿)=0.5
\end{aligned}
$$
所以在如题所述的公园遇到一个穿凉鞋的人，他(她)是男(女)性的概率为0.5。

**6.说明集成学习算法的主要类别，以Adaboost和随机森林为例，说明算法原理和实现过程**

集成学习通过构建多个学习器并将其结合来完成学习任务的方法。可以根据结合策略可分为串行序列化方法和并行结合方法，前者以Bossting方法为代表，个体学习器之间常常存在强依赖关系，后者则以Bagging和随机森林为代表，个体学习器之间不存在强依赖。
关于Adaboost:
此算法先从初始训练集训练出一个基学习器，在根据这个基学习器的预测结果来调整样本分布，是之前错分的样本在后续学习器的训练样本在后续受到更多关注，然后训练下一个学习器，如此重复，直至学习器数目达到事先指定的$T$,最后将T个学习器进行加权结合。
集成学习器表示为：$H(x)=\sum_{t=1}^T\alpha_t h_t(x)$,其中$h_t$为个体学习器，$\alpha_t$为其对应的权重。
学习器需要最小化损失函数$\mathcal{L}(H|\mathcal{D})=\mathbb{E}_{x\sim \mathcal{D}}[e^{-f(x)H(x)}]$
前向分布求解：每轮求解一个学习器的$h_t$和权重$\alpha_t$,第t轮优化目标
$$
(\alpha_t,h_t)=argmin_{\alpha,h}\mathcal{L}(H_{t-1}+\alpha h|\mathcal{D})
$$
根据指数损失函数的定义式，有
$$
\begin{aligned}
\mathcal{L}(H_{t-1}+\alpha h|\mathcal{D})&=\mathbb{E}_{x \sim \mathcal{D}}[e^{-f(x)(H_{t-1}(x)+\alpha h(x))}]\\
&=\sum_{i=1}^{|\mathcal{D}|}\mathcal{D}(x_i)e^{-f(x_i)(H_{t-1}(x_i)+\alpha h(x_i))}\\
&=\sum_{i=1}^{|\mathcal{D}|}\mathcal{D}(x_i)e^{-f(x_i)H_{t-1}(x_i)}e^{-f(x_i)\alpha h(x_i)}
\end{aligned}
$$
因为$f(x_i)$和$h(x_i)$仅可取值{-1,1}，可以推得
$$
\begin{aligned}
\mathcal{L}(H_{t-1}+\alpha h|\mathcal{D})&=\sum_{i=1}^{|D|}\mathcal{D}(x_i)e^{-f(x_i)H_{t-1}(x_i)}(e^{-\alpha}+(e^{\alpha}-e^{-\alpha})\mathbb{I}(f(x_i)\neq h(x_i)))\\
&=\sum_{i=1}^{|D|}\mathcal{D}(x_i)e^{-f(x_i)H_{t-1}(x_i)}e^{-\alpha}+\sum_{i=1}^{|D|}\mathcal{D}(x_i)e^{-f(x_i)H_{t-1}(x_i)}(e^{\alpha}-e^{-\alpha})\mathbb{I}(f(x_i)\neq h(x_i))
\end{aligned}
$$
做一个简单的符号替换，令$\mathcal{D}_t^{'}(x_i)=\mathcal{D}(x_i)e^{-f(x_i)H_{t-1}(x_i)}$,化简得
$$
\mathcal(L)(H_{t-1}+\alpha h|\mathcal{D})=e^{-\alpha}\sum_{i=1}^{|D|}\mathcal{D}_t^{'}(x_i)+(e^{\alpha}-e^{- \alpha})\sum_{i=1}^{|D}\mathcal{D}_t^{'}(x_i)\mathbb{I}(f(x_i)\neq h(x_i))
$$
忽略和$h_t$无关的项
$$
h_t=argmin_h(e^{\alpha}-e^{-\alpha})\sum_{i=1}^{|D|}\mathcal{D_t{'}(x_i)}\mathbb{I}(f(x_i)\neq h(x_i))
$$
由于$\alpha>\frac{1}{2}$,所以$e^{\alpha}-e^{-\alpha}>0$恒成立。
求解目标简化为
$$
h_t=argmin_h\sum_{i=1}^{|D}\mathcal{D}_t^{'}(x_i)\mathbb{I}(f(x_i)\neq h(x_i))
$$
上式中$D_{t}^{'}(x_i)$可以看作一个分布，我们将其规范化$\mathcal{D}_{t}(x_i)=\frac{\mathcal{D}_t^{'}(x_i)}{\sum_{i=1}^{|D|}\mathcal{D}_t^{'}(x_i)}$。
第t轮的样本权重可以通过第t-1轮的样本权重计算。
$$
\begin{aligned}
    \mathcal{D}_{t+1}(x_i)&=\mathcal{D}(x_i)e^{-f(x_i)H_t(x_i)}\\&=\mathcal{D}(x_i)e^{-f(x_i)(H_{t-1}(x_i)+\alpha_t h_t(x_i))}\\
    &=\mathcal{D}(x_i)e^{-f(x_i)H_{t-1}(x_i)}e^{-f(x_i)\alpha_t h_t(x_i)}\\
    &=\mathcal{D}(x_i)e^{-f(x_i)\alpha_t h_t(x_i)}
\end{aligned}
$$
接下来求解学习器$h_t$权重$\alpha_t$。损失函数$\mathcal{L}(H_{t-1}+\alpha h|D)$对$\alpha$求导有：
$$
\begin{aligned}
\frac{\partial \mathcal{L}(H_{t-1}+\alpha h_t|D)}{\partial \alpha}&=\frac{\partial(e^{-\alpha}\sum_{i=1}^{|D|}\mathcal{D}_t^{'}(x_i)+(e^{\alpha}-e^{-\alpha})\sum_{i=1}^{|D|}\mathcal{D}_{t}^{'}(x_i)\mathbb{I}(f(x_i)\neq h(x_i)))}{\partial \alpha}\\
&=-e^{- \alpha}\sum_{i=1}^{|D|}\mathcal{D}_t^{'}(x_i)+(e^{\alpha}+e^{- \alpha})\sum_{i=1}^{|D|}\mathcal{D}_{t}^{'}(x_i)\mathbb{I}(f(x_i)\neq h(x_i))
\end{aligned}
$$
令导数为0，移向得：
$$
\begin{aligned}
    \frac{e^{- \alpha}}{e^{\alpha}+e^{- \alpha}}&=\frac{\sum_{i=1}^{|D|}\mathcal{D}_{t}^{'}(x_i)\mathbb{I}(f(x_i) \neq h(x_i))}{\sum_{i=1}^{|D|}\mathcal{D}_{t}^{'}(x_i)}\\
    &=\sum_{i=1}^{|D|}\frac{\mathcal{D}(x_i)}{Z_t}\mathbb{I}(f(x_i)\neq h(x_i))\\
    &=\sum_{i=1}^{|D|}\mathcal{D}_t(x_i)\mathbb{I}(f(x_i)\neq h(x_i))\\
    &=\mathbb{E}_{x\sim \mathcal{D}_t}[\mathbb{I}f(x_i)\neq h(x_i)]\\
    &=\epsilon_t
\end{aligned}
$$
求解上式可得
$$
\alpha_t=\frac{1}{2}\ln(\frac{1-\epsilon_t}{\epsilon_t})
$$
以上即为参数求解步骤，算法实现如下：
输入：
* 训练集：$D={(x_1,y_1),(x_2,y_2),\cdots,(x_m,y_m)}$
* 基学习算法$\mathcal{E}$
* 训练轮数$T$

过程：
* $\mathcal{D}_1(x)=\frac{1}{m}$
* for t=1,2,...T do
  * $h_t=\mathcal{E}(D,\mathcal{D}_t)$
  * $\epsilon=P_{x \sim \mathcal{D_t}}(h_t(x)\neq f(x))$
  * if $\epsilon_t>0.5$ then break
  * $\alpha_t=\frac{1}{2}\ln(\frac{1-\epsilon_t}{\epsilon_t})$
  * 按照上面的公式从$\mathcal{D}_t$求得$\mathcal{D}_{t+1}$
* end for
  

输出：
$F(x)=sign(\sum_{t=1}^{T}\alpha_t h_t(x))$

关于随机森林：
Bagging和随机森林都是基学习器为决策树的算法，随机森林对基学习器进行训练时，都用自主采样法对数据集进行采样生成新的训练集并随机选择属性，这样的扰动大大提升了学习器的性能。
具体实现如下：
输入：
* 训练集：$D={(x_1,y_1),(x_2,y_2),\cdots,(x_m,y_m)}$
* 基学习算法$\mathcal{E}$
* 训练轮数$T$
  

过程：
* for t=1,2,...,T do
  * 从$D$采样的$\mathcal{D}_{bs}$
  * 从划分属性采样得pro
  * $h_t=\mathcal{E}(D,\mathcal{D}_{bs},pro)$
* end for

输出：
* $H(x)=argmax_{y \in \mathcal{Y}\sum_{t=1}^T\mathbb{I}(h_t(x)=y)}$
