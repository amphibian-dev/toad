# 数据探查工具

ESC Team 数据探查工具合集

## Install

```
python setup.py install
```

## Usage

```
import detector


data = pd.read_csv('test.csv')

detector.detect(data)

detect.quality(data, target = 'TARGET')

detect.IV(data, feature = 'feature_name', target = 'TARGET', method = 'dt', min_samples = 0.1)
```

## Documents

working...
