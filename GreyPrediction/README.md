# 灰色预测模型

**什么时候用灰色预测**？

下面是清风的看法，使用哪种模型进行预测是仁者见 仁智者见智的事情：

- （1）数据是**以年份度量的非负数据**（如果是月份或者季度 数据一定要用我们上一讲学过的时间序列模型）；
- （2）数据**能经过准指数规律的检验**（除了前两期外，后面 **至少90%（可根据情况而定）的期数的光滑比要低于0.5**）；
- （3）**数据的期数较短且和其他数据之间的关联性不强**（**小 于等于10，也不能太短了，比如只有3期数据**，**4-10左右**），要是数据期数较长，一般用传统的时间序列模型比较合适。

**需要的参数：**

- 原始数据
- 预测数量

**TODO:**

- 结果显示改为自动更改
  - 现在实际运用需要将所有的排污总量字眼替换成你的指标，例如：销量、消费等（替换方法：快捷键Ctrl+F ，查找内容填写：排污总量，替换为填写：你自己的指标 然后点击全部替换）
- 代码降重，此代码是清风写的