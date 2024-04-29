## git工作区/暂存区/远程仓库管理

### **1.简介**

Git是一个开源的分布式颁布控制系统，可以高效、高速地处理从很小到很大地项目版本管理。

### **2.特点**

#### **2.1****分布式特点**

分布式相比于集中式的最大区别在于开发者可以提交到本地，每个开发者通过克隆（git clone），在本地机器上拷贝一个完整的Git仓库。

![img](C:/Users/yanwei/Documents/markdown/%E5%9B%BE%E7%89%87/17144002111761.png)

#### **2.2功能特点**

从**一般开发者**的角度来看，git有以下功能：

1、从服务器上克隆完整的Git仓库（包括代码和版本信息）到单机上。

2、在自己的机器上根据不同的开发目的，创建分支，修改代码。

3、在单机上自己创建的分支上提交代码。

4、在单机上合并分支。

5、把服务器上最新版的代码fetch下来，然后跟自己的主分支合并。

6、生成补丁（patch），把补丁发送给主开发者。

7、看主开发者的反馈，如果主开发者发现两个一般开发者之间有冲突（他们之间可以合作解决的冲突），就会要求他们先解决冲突，然后再由其中一个人提交。如果主开发者可以自己解决，或者没有冲突，就通过。

8、一般开发者之间解决冲突的方法，开发者之间可以使用pull 命令解决冲突，解决完冲突之后再向主开发者提交补丁。

从**主开发者**的角度（假设主开发者不用开发代码）看，git有以下功能：

1、查看邮件或者通过其它方式查看一般开发者的提交状态。

2、打上补丁，解决冲突（可以自己解决，也可以要求开发者之间解决以后再重新提交，如果是[开源项目](https://baike.baidu.com/item/开源项目/3406069?fromModule=lemma_inlink)，还要决定哪些补丁有用，哪些不用）。

3、向公共服务器提交结果，然后通知所有开发人员。

#### **2.3优缺点**

**优点：**

适合[分布式开发](https://baike.baidu.com/item/分布式开发/4143301?fromModule=lemma_inlink)，强调个体。

公共服务器压力和数据量都不会太大。

速度快、灵活。

任意两个开发者之间可以很容易的解决冲突。

离线工作。

**缺点：**

资料少（起码中文资料很少）。

学习周期相对而言比较长。

不符合[常规思维](https://baike.baidu.com/item/常规思维/9532113?fromModule=lemma_inlink)。

代码[保密性](https://baike.baidu.com/item/保密性/4928247?fromModule=lemma_inlink)差，一旦开发者把整个库克隆下来就可以完全公开所有代码和版本信息。

### 3.基本使用

- 下载Git，注册登录github，网速不行没有魔法时，采用“UU加速器”学术资源加速
- 到需要管理的项目一级下，初始化git，添加readme
    - ```Plain
        git init
        git add README.md
        ```
- 将项目放入暂存区
    - ```Plain
        git add .     # 将整个文件放入暂存区
        git add file  # 将某个文件放入暂存区
        ```
- 提交commit，提交每一次版本的信息，很重要（后续如需恢复版本的依据）
    - ```Plain
        git commit -m "某一次提交，修改，功能"
        ```
- 更改分区名字（optin）
    - ```Plain
        git branch -M main  # 最开始是mastar，后续更改的
        ```
- 建立远程仓库的链接
    - ```Plain
        git remote add origin "github仓库链接"
        ```
- 将项目push到远程仓库
    - ```Plain
        git push -u origin main  # 初次push
        git push                 # 后续push，直接写
        ```
- 如需修改版本，本地恢复到某个节点状态
    - ```Plain
        git log  # 通过log查看历史版本
        git reset --hard 版本id
        ```
- 修改后，强推到远程仓库 <此时如果分支较远或者改动较多，使用git push origin可能会报错失败，此时可使用强推>
    - ```Plain
        git push -f -u origin main
        ```

### 4.总结

以上仅为Git的一些初级使用，如需了解更多Git知识，参考 Git-Book：https://git-scm.com/book/en/v2

了解Git 多人协同开发，可参考 https://blog.csdn.net/whc18858/article/details/133209975 

### 参考

> [1] Git 百度百科  https://baike.baidu.com/item/GIT/12647237
>
> [2] https://blog.csdn.net/hhj13978064496/article/details/99856779