# -*- coding: utf-8 -*-
"""
构建两个1-hop关系表
- news index => user index list, 可以提前做好sample
- user index => news index list, 可以提前做好sample

构建两个 2-hop关系表
- news index => 2-hop news index list, 可以提前做好sample
- user index => 2-hop user index list, 可以提前做好sample

只能利用 hist 信息构建这个表，只保留在train中出现过的 user 和 item
train 和 val的时候共用上边的表，都是基于train的hist来构建
test 单独跑，基于trian + test 的hist来构建
"""
