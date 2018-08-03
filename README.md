# 数据探查工具

ESC Team 数据探查工具合集

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
