from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
    _XGBOOST_IMPORT_ERROR = None
except Exception as exc:
    XGBRegressor = None
    _XGBOOST_IMPORT_ERROR = exc

def train_linear_regression(X_train, y_train):

    model = LinearRegression()

    model.fit(X_train, y_train)

    return model


def train_random_forest(X_train, y_train):

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def train_xgboost(X_train, y_train):
    if XGBRegressor is None:
        raise RuntimeError(
            "XGBoost is unavailable. On macOS install the OpenMP runtime with `brew install libomp` "
            "and then reinstall xgboost if needed. Original error: {}".format(_XGBOOST_IMPORT_ERROR)
        )

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model
