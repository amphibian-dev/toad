# Welcome to Toad contributing guide 

We're so glad you're thinking about contributing to toad project. If you're unsure about anything, just ask @Secbone or submit the issue or pull request anyway. The worst that can happen is you'll be politely asked to change something. We love all friendly contributions.

我们非常开心你乐意为 toad 项目贡献代码。如果你有任何疑问，可以联系 @Secbone 或者提交 issue 和 pull request 都可以。最糟不过是被礼貌地要求你修改一些东西。我们非常愿意看到所有善意的问题。

## Getting Started · 开始吧

### Setup Environment · 设置环境

Setting up the environment is very simple, you just need to run the following command

设置环境非常简单，你只需要执行以下代码

```bash
make install
```

All done! Now you can enjoy your coding~

完成！开始享受你的编码吧~

### About Cython · 关于 Cython

`toad.merge` module is compiled with `cython`, so if you want to change something with `toad.merge`, you need to run

`toad.merge` 模块是使用 `cython` 编译的，所有如果你想要对 `toad.merge` 模块进行改动时，你需要运行

```bash
make build
```
after you updated code.

之后来使你的代码生效。

### Testing · 测试

You can run

你可以执行

```bash
make test
```

for testing the whole package. We recommend that you do this before every commit to avoid new code impacting old functionality.

来测试整个包的代码。我们建议你在每次体检前这么做，以防止新代码对老的功能产生影响。

You can also run

你也可以运行

```bash
make test toad/xxxx_test.py
```

to test only a single module.

来只测试某一个模块。

### Pull Request

When you're finished with the changes, creating a pull request and waiting for merge.

当你完成所有的改动后，就可以创建一个 pull request 并且等它被合并啦~
