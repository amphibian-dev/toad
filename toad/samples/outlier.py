#孤立森林
import toad
from sklearn.ensemble import IsolationForest
from toad.transform import WOETransformer
from toad.tadpole import utils

frame.head()

def outlier_detect(frame, method="iso_forest",x = None, target = 'target',character=True):





def gauss_outlier():
    


def iso_forest(target="target",str=True,prob=True,top=0.01,**kwargs):
    if str:
        woe_transer = WOETransformer()
        woed = woe_transer.fit_transform(frame.drop(columns=["target"]),frame["target"],select_dtypes="object")

    clf = IsolationForest(**kwargs)
    clf.fit(woed)  # fit the added trees

    if prob:
        frame["iso_prob"] = clf.score_samples(woed)
        threshold = frame["iso_prob"].quantile(top)
        res = (frame,)
        res+= (threshold,)
    else:
        frame["iso_mark"] = clf.predict(woed)
        res = frame

    return unpack_tuple(res)
