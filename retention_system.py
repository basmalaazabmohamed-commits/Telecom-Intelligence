import cloudpickle
import numpy as np
import pandas as pd

from scipy.special import logit


class TelecomRetentionSystem:

    def __init__(self, model_path):

        with open(model_path, "rb") as file:

            self.bundle = cloudpickle.load(file)


    def validate_input(self, data):

        if not isinstance(data, pd.DataFrame):

            raise TypeError(
                "Input must be a pandas DataFrame."
            )


        if data.empty:

            raise ValueError("Input data is empty.")


        if not data.columns.is_unique:

            raise ValueError("Duplicate column names detected.")


        required = self.bundle["raw_features"]


        missing = [

            feature for feature in required

            if feature not in data.columns

        ]


        if missing:

            raise ValueError(

                f"Missing required columns: {missing}"

            )


        result = data.loc[:, required].copy()


        for feature in self.bundle["raw_numeric_features"]:

            original = result[feature]

            converted = pd.to_numeric(

                original,

                errors="coerce"

            )


            invalid = original.notna() & converted.isna()


            if invalid.any():

                raise ValueError(

                    f"Invalid numeric input: {feature}"

                )


            result[feature] = converted


        categorical = [

            feature for feature in required

            if feature not in self.bundle["raw_numeric_features"]

        ]


        for feature in categorical:

            result[feature] = (

                result[feature]

                .astype(object)

                .where(result[feature].notna(), np.nan)

            )


        return result


    def predict_proba(self, data):

        raw = self.validate_input(data)

        bundle = self.bundle

        predictions = []


        for component in bundle["components"].values():

            if component["input_type"] == "original":

                processed = bundle["domain_cleaner"](raw)

            else:

                processed = bundle["feature_engineer"](raw)


            inputs = processed.loc[

                :, component["features"]

            ]


            probabilities = component["estimator"].predict_proba(

                inputs

            )[:, 1]


            predictions.append(probabilities)


        matrix = np.column_stack(predictions)

        champion = bundle["champion"]


        if champion == "Equal Soft Voting":

            probabilities = matrix @ bundle["equal_weights"]


        elif champion == "Weighted Soft Voting":

            probabilities = matrix @ bundle["optimized_weights"]


        elif champion == "OOF Stacking":

            probabilities = (

                bundle["meta_model"]

                .predict_proba(matrix)[:, 1]

            )


        else:

            probabilities = matrix[:, 0]


        calibration = bundle["calibration_method"]


        if calibration == "Sigmoid":

            logits = logit(

                np.clip(probabilities, 1e-7, 1 - 1e-7)

            ).reshape(-1, 1)


            probabilities = (

                bundle["calibrator"]

                .predict_proba(logits)[:, 1]

            )


        elif calibration == "Isotonic":

            probabilities = bundle["calibrator"].predict(

                probabilities

            )


        if not np.isfinite(probabilities).all():

            raise ValueError("Non-finite predictions detected.")


        return np.clip(probabilities, 0, 1)


    def predict(self, data):

        probabilities = self.predict_proba(data)


        return (

            probabilities >= self.bundle["threshold"]

        ).astype(int)


    def score_customers(self, data):

        probabilities = self.predict_proba(data)

        config = self.bundle["business_config"]


        result = pd.DataFrame({

            "Churn_Probability": probabilities,

            "Predicted_Churn": (

                probabilities >= self.bundle["threshold"]

            ).astype(int)

        }, index=data.index)


        revenue = pd.to_numeric(

            data["rev_Mean"],

            errors="coerce"

        )


        result["Monthly_Revenue"] = revenue.where(

            revenue >= 0

        )


        result["Monthly_Revenue_Exposure"] = (

            result["Churn_Probability"]

            *

            result["Monthly_Revenue"]

        )


        result["Expected_Net_Contribution"] = (

            result["Monthly_Revenue_Exposure"]

            *

            config["retention_success_rate"]

            *

            config["retention_months"]

            *

            config["contribution_margin"]

            -

            config["campaign_contact_cost"]

            -

            config["retention_offer_cost"]

        )


        result["Campaign_Selected"] = False


        capacity = int(

            len(result) * config["campaign_capacity"]

        )


        eligible = result.loc[

            result["Expected_Net_Contribution"] > 0

        ].sort_values(

            "Expected_Net_Contribution",

            ascending=False

        )


        selected = eligible.head(capacity).index


        result.loc[

            selected,

            "Campaign_Selected"

        ] = True


        return result
