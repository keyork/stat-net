# stat-net

StatNet: 统计理论指导下的神经网络

report [here](https://github.com/keyork/stat-net/blob/main/StatNet-%E7%BB%9F%E8%AE%A1%E7%90%86%E8%AE%BA%E6%8C%87%E5%AF%BC%E4%B8%8B%E7%9A%84%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.pdf)

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
