import utils
from sklearn.ensemble import AdaBoostClassifier

def Ada(x_train, y_train, x_test, y_test):
    model = AdaBoostClassifier(
        algorithm='SAMME.R',
        base_estimator=None,
        learning_rate=0.1,
        n_estimators=500,
        random_state=1,
    )

    model.fit(x_train, y_train)

    prob = model.predict_proba(x_test)[:, 0]

    return utils.KS(prob, y_test)
