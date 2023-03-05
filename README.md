# stat-net

StatNet: 统计理论指导下的神经网络

report [here]('./StatNet-统计理论指导下的神经网络.pdf')

（已写好报告，后续再完善这个文档）

## 运行方式

`python train.py`

## test result

MLP

- 100-loss=0.414172, acc=0.724700
- 500-loss=0.147488, acc=0.859800
- 1000-loss=0.112205, acc=0.883800
- 5000-loss=0.056688, acc=0.924700
- 10000-loss=0.026235, acc=0.952100
- 30000-loss=0.008682, acc=0.952000

Stat

- 100-loss=0.148356, acc=0.764300
- 500-loss=0.042105, acc=0.874800
- 1000-loss=0.022027, acc=0.897100
- 5000-loss=0.007350, acc=0.937000
- 10000-loss=0.004028, acc=0.955000
- 30000-loss=0.002041, acc=0.970700
