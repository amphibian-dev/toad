# 数据探查工具

ESC Team 数据探查工具合集
### 环境配置
```
未安装vc++的，先通过该链接https://visualstudio.microsoft.com/zh-hans/thank-you-downloading-visual-studio/?sku=BuildTools&rel=15下载vc安装工具，
下载好后点击安装，安装过程中根据显示的勾选需要安装的组件vc++生产工具和旁边的可选中的C++/CLI支持即可，然后重启电脑。
```
## Install

```
make install
```
or
```
python setup.py install
```

## Usage

```
import detector


data = pd.read_csv('test.csv')

detector.detect(data)

detect.quality(data, target = 'TARGET', iv_only = True)

detect.IV(feature, target, method = 'dt', min_samples = 0.1)
```

## Documents

working...
