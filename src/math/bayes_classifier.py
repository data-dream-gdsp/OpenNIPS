import pandas as pd
import numpy as np


class NaiveBayes:
    def __init__(self):
        self.model = {}

    def fit(self, x: pd.Series, y: pd.Series = pd.Series()) -> "NaiveBayes":
        if y.empty:
            y = x.iloc[:, -1]
        y_counts = y.value_counts()
        y_counts = y_counts.apply(
            lambda val: (val + 1) / (y.size + y_counts.size)
        )

        model = {}
        for class_name, value in y_counts.items():
            model[class_name] = {'PClass': value, 'PFeature': {}}

        prop_names = x.columns[:-1]
        prop_by_feature = {}
        for feature_name in prop_names:
            prop_by_feature[feature_name] = x[feature_name].value_counts().index.tolist()

        for class_name, group in x.groupby(x.columns[-1]):
            for feature_name in prop_names:
                class_p_feature = {}
                prop_summary = group[feature_name].value_counts()
                for prop_name in prop_by_feature[feature_name]:
                    if prop_name not in prop_summary:
                        prop_summary[prop_name] = 0

                n_i = len(prop_by_feature[feature_name])
                prop_summary = prop_summary.apply(
                    lambda val: (val + 1) / (n_i + prop_summary.size)
                )

                for feature_prop_name, p_value in prop_summary.items():
                    class_p_feature[feature_prop_name] = p_value
                model[class_name]['PFeature'][feature_name] = class_p_feature

        self.model = model
        return self

    def _predict_series(self, series: pd.Series) -> pd.Series:
        max_rate = None
        class_select = None

        for class_name, model_data in self.model.items():
            rate = np.log(model_data['PClass'])
            p_feature = model_data['PFeature']

            for feature_name, value in series.items():
                props_rate = p_feature.get(feature_name)
                if props_rate is None:
                    continue

                rate += np.log(props_rate.get(value)) if value in props_rate else 0

            if max_rate is None or rate > max_rate:
                max_rate = rate
                class_select = class_name

        return class_select

    def predict(self, data) -> pd.Series:
        if isinstance(data, pd.Series):
            return self._predict_series(data)
        return data.apply(
            lambda val: self._predict_series(val),
            axis=1,
        )
