"""Tests for classical ML models."""

import numpy as np
import pytest
from src.models.classical import XGBoostModel, RandomForestModel, LogisticRegressionModel


class TestClassicalModels:
    """Test classical ML models."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        X_train = np.random.randn(100, 20)
        y_train = np.random.randint(0, 2, 100)
        X_val = np.random.randn(20, 20)
        y_val = np.random.randint(0, 2, 20)
        return X_train, y_train, X_val, y_val

    def test_xgboost_model(self, sample_data):
        """Test XGBoost model."""
        X_train, y_train, X_val, y_val = sample_data

        model = XGBoostModel()
        model.fit(X_train, y_train, X_val, y_val)

        y_pred = model.predict(X_val)
        assert len(y_pred) == len(y_val)

        y_proba = model.predict_proba(X_val)
        assert y_proba.shape == (len(y_val), 2)

        metrics = model.evaluate(X_val, y_val)
        assert "accuracy" in metrics
        assert "auprc" in metrics

    def test_random_forest_model(self, sample_data):
        """Test Random Forest model."""
        X_train, y_train, X_val, y_val = sample_data

        model = RandomForestModel()
        model.fit(X_train, y_train, X_val, y_val)

        y_pred = model.predict(X_val)
        assert len(y_pred) == len(y_val)

        metrics = model.evaluate(X_val, y_val)
        assert "accuracy" in metrics

    def test_logistic_regression_model(self, sample_data):
        """Test Logistic Regression model."""
        X_train, y_train, X_val, y_val = sample_data

        model = LogisticRegressionModel()
        model.fit(X_train, y_train, X_val, y_val)

        y_pred = model.predict(X_val)
        assert len(y_pred) == len(y_val)

        metrics = model.evaluate(X_val, y_val)
        assert "accuracy" in metrics
