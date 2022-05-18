---
title: "Py Pathlib Note"
date: 2022-05-16T11:23:45+08:00
draft: false
description: python模块pathlib的笔记
categories:
    - python
tags:
    - python
image: image/椿.jpeg
---
位于Lib/pathlib.py 是面向对象的文件系统路径，与之前的os.path功能相近，从文档来看，这个库主要是用来操作文件系统路径的，与直接操作路径的字符串相比，它可以屏蔽`Unix`和`Windows`的文件路径格式差异。网上说重要的是将os和path解耦，这个我不太懂。
## 用法
### 基础使用
#### 导入主类
~~~python
from pathlib import Path
~~~
#### 列出子目录
~~~python
p=Path('.')
[x for x in p.iterdir() if x.is_dir()]
~~~
#### 列出目录树下指定类型文件
~~~python
list(p.glob('**/*.py'))
~~~
#### 查询路径属性
~~~python
p.exists()
p.is_dir()
~~~