# GERL

## Preprocess

构建dict：
- news_id => news index
- user_id => user index
- word => word index
- news_index => title seq

构建两个1-hop关系表
- news index => user index list, 可以提前做好sample
- user index => news index list, 可以提前做好sample

构建两个 2-hop关系表
- news index => 2-hop news index list, 可以提前做好sample
- user index => 2-hop user index list, 可以提前做好sample

## 构建数据集
每个训练数据的格式为: 
```json
{
    user: 123,
    hist_news: [1, 2, 3]
    neighbor_users: [4, 5, 6]
    target_news: [7, 8, 9, 10, 11],
    neighbor_news: [
        [27, 28, 29],
        [30, 31, 32],
        [33, 34, 35],
        [36, 37, 38],
        [39, 40, 41]
    ]
}
```

每个测试数据的格式为: 
```json
{
    imp_id: 000,
    user: 123,
    hist_news: [1, 2, 3]
    neighbor_users: [4, 5, 6]
    target_news: 7,
    neighbor_news: [27, 28, 29],
    y: 1
}
```
