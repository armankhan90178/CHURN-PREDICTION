"""
ChurnShield 2.0 — Advanced Anomaly Detector

Purpose:
Detect abnormal customers, risky behavior,
suspicious business activity, churn spikes,
revenue anomalies, engagement drops,
payment fraud signals, and data inconsistencies.

Capabilities:
- statistical anomaly detection
- ML-based anomaly scoring
- churn-risk anomaly detection
- revenue spike/drop detection
- customer behavior deviation
- outlier scoring
- multi-column anomaly fusion
- business intelligence alerts
- anomaly explanations
- severity classification
- anomaly clustering
- automatic recommendations

Author: ChurnShield AI
"""

import logging
import warnings
from datetime import datetime
from typing import Dict, List, Any

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

logger = logging.getLogger("churnshield.anomaly_detector")


# ============================================================
# MAIN ENGINE
# ============================================================

class AnomalyDetector:

    def __init__(self):

        self.model = None
        self.scaler = StandardScaler()

        self.anomaly_threshold = -0.15

    # ========================================================
    # MAIN ENTRY
    # ========================================================

    def detect(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:

        logger.info("Starting anomaly detection")

        data = df.copy()

        numeric_cols = self._get_numeric_columns(data)

        if len(numeric_cols) == 0:

            return {
                "status": "failed",
                "reason": "No numeric columns available"
            }

        prepared = self._prepare_data(
            data,
            numeric_cols
        )

        isolation_results = self._run_isolation_forest(
            prepared
        )

        zscore_results = self._run_zscore_detection(
            prepared
        )

        business_results = self._run_business_rule_detection(
            data
        )

        cluster_results = self._run_cluster_detection(
            prepared
        )

        combined = self._combine_results(
            data,
            isolation_results,
            zscore_results,
            business_results,
            cluster_results
        )

        summary = self._generate_summary(
            combined
        )

        alerts = self._generate_alerts(
            combined
        )

        recommendations = self._generate_recommendations(
            combined
        )

        top_anomalies = self._extract_top_anomalies(
            combined
        )

        return {
            "generated_at": datetime.now().isoformat(),

            "summary": summary,
            "alerts": alerts,
            "recommendations": recommendations,

            "top_anomalies": top_anomalies,

            "anomaly_count": int(
                combined["is_anomaly"].sum()
            ),

            "anomaly_percentage": round(
                combined["is_anomaly"].mean() * 100,
                2
            ),

            "results": combined.to_dict(
                orient="records"
            )
        }

    # ========================================================
    # NUMERIC COLUMNS
    # ========================================================

    def _get_numeric_columns(
        self,
        df: pd.DataFrame
    ) -> List[str]:

        numeric_cols = df.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        ignored = [
            "customer_id",
            "id"
        ]

        numeric_cols = [
            c for c in numeric_cols
            if c not in ignored
        ]

        return numeric_cols

    # ========================================================
    # PREPARE DATA
    # ========================================================

    def _prepare_data(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str]
    ) -> pd.DataFrame:

        data = df[numeric_cols].copy()

        data = data.fillna(0)

        data = data.replace(
            [np.inf, -np.inf],
            0
        )

        scaled = self.scaler.fit_transform(data)

        scaled_df = pd.DataFrame(
            scaled,
            columns=numeric_cols
        )

        return scaled_df

    # ========================================================
    # ISOLATION FOREST
    # ========================================================

    def _run_isolation_forest(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:

        logger.info("Running Isolation Forest")

        self.model = IsolationForest(
            n_estimators=250,
            contamination=0.05,
            random_state=42,
            n_jobs=-1
        )

        predictions = self.model.fit_predict(data)

        scores = self.model.decision_function(data)

        result = pd.DataFrame({
            "iforest_prediction": predictions,
            "iforest_score": scores,
            "iforest_anomaly": (
                predictions == -1
            ).astype(int)
        })

        return result

    # ========================================================
    # Z-SCORE DETECTION
    # ========================================================

    def _run_zscore_detection(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:

        logger.info("Running Z-score detection")

        z_scores = np.abs(
            (data - data.mean()) /
            (data.std() + 1e-6)
        )

        anomaly_flags = (
            z_scores > 3
        ).sum(axis=1)

        return pd.DataFrame({
            "zscore_anomaly_count": anomaly_flags,
            "zscore_anomaly": (
                anomaly_flags > 0
            ).astype(int)
        })

    # ========================================================
    # BUSINESS RULES
    # ========================================================

    def _run_business_rule_detection(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        logger.info("Running business rule detection")

        anomaly_scores = []

        for _, row in df.iterrows():

            score = 0

            # Revenue anomalies
            if "monthly_revenue" in row:

                revenue = row["monthly_revenue"]

                if revenue < 0:
                    score += 5

                elif revenue > 500000:
                    score += 3

            # Login anomalies
            if "login_frequency" in row:

                logins = row["login_frequency"]

                if logins < 0:
                    score += 4

                elif logins > 1000:
                    score += 2

            # Ticket anomalies
            if "support_tickets" in row:

                tickets = row["support_tickets"]

                if tickets > 100:
                    score += 3

            # Payment anomalies
            if "payment_delays" in row:

                delays = row["payment_delays"]

                if delays > 15:
                    score += 4

            # Seat anomalies
            if (
                "active_seats" in row and
                "total_seats" in row
            ):

                if row["active_seats"] > row["total_seats"]:
                    score += 5

            # Usage anomalies
            if "feature_usage_score" in row:

                usage = row["feature_usage_score"]

                if usage < 0 or usage > 1:
                    score += 4

            anomaly_scores.append(score)

        scores = np.array(anomaly_scores)

        return pd.DataFrame({
            "business_rule_score": scores,
            "business_rule_anomaly": (
                scores >= 4
            ).astype(int)
        })

    # ========================================================
    # CLUSTER DETECTION
    # ========================================================

    def _run_cluster_detection(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:

        logger.info("Running cluster anomaly detection")

        pca = PCA(
            n_components=min(
                5,
                data.shape[1]
            )
        )

        reduced = pca.fit_transform(data)

        cluster_model = DBSCAN(
            eps=1.5,
            min_samples=5
        )

        clusters = cluster_model.fit_predict(
            reduced
        )

        return pd.DataFrame({
            "cluster_label": clusters,
            "cluster_anomaly": (
                clusters == -1
            ).astype(int)
        })

    # ========================================================
    # COMBINE RESULTS
    # ========================================================

    def _combine_results(
        self,
        original_df: pd.DataFrame,
        isolation_results: pd.DataFrame,
        zscore_results: pd.DataFrame,
        business_results: pd.DataFrame,
        cluster_results: pd.DataFrame
    ) -> pd.DataFrame:

        combined = original_df.copy()

        combined = pd.concat(
            [
                combined.reset_index(drop=True),
                isolation_results,
                zscore_results,
                business_results,
                cluster_results
            ],
            axis=1
        )

        combined["anomaly_score"] = (
            combined["iforest_anomaly"] * 4 +
            combined["zscore_anomaly"] * 2 +
            combined["business_rule_anomaly"] * 3 +
            combined["cluster_anomaly"] * 2
        )

        combined["severity"] = combined[
            "anomaly_score"
        ].apply(
            self._classify_severity
        )

        combined["is_anomaly"] = (
            combined["anomaly_score"] >= 4
        ).astype(int)

        combined["anomaly_reason"] = combined.apply(
            self._generate_reason,
            axis=1
        )

        return combined

    # ========================================================
    # SUMMARY
    # ========================================================

    def _generate_summary(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:

        total = len(df)

        anomalies = int(
            df["is_anomaly"].sum()
        )

        severe = int(
            (df["severity"] == "critical").sum()
        )

        return {
            "total_records": total,
            "total_anomalies": anomalies,
            "critical_anomalies": severe,
            "anomaly_rate_percent": round(
                anomalies / total * 100,
                2
            ) if total else 0,
            "business_health": self._classify_health(
                anomalies / total if total else 0
            )
        }

    # ========================================================
    # ALERTS
    # ========================================================

    def _generate_alerts(
        self,
        df: pd.DataFrame
    ) -> List[Dict]:

        alerts = []

        critical = df[
            df["severity"] == "critical"
        ]

        if len(critical) > 0:

            alerts.append({
                "type": "critical_anomaly_alert",
                "message": (
                    f"{len(critical)} critical anomalies detected"
                ),
                "severity": "critical"
            })

        high_revenue_risk = df[
            df["business_rule_score"] >= 5
        ]

        if len(high_revenue_risk) > 0:

            alerts.append({
                "type": "financial_risk",
                "message": (
                    "Potential financial anomalies detected"
                ),
                "severity": "high"
            })

        return alerts

    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    def _generate_recommendations(
        self,
        df: pd.DataFrame
    ) -> List[str]:

        recommendations = []

        anomaly_rate = df["is_anomaly"].mean()

        if anomaly_rate > 0.10:

            recommendations.append(
                "Investigate abnormal customer behavior immediately"
            )

        if (
            df["business_rule_anomaly"].sum() > 10
        ):

            recommendations.append(
                "Review revenue, seat, and payment validation rules"
            )

        if (
            df["cluster_anomaly"].sum() > 5
        ):

            recommendations.append(
                "Customer segments behaving unexpectedly"
            )

        if len(recommendations) == 0:

            recommendations.append(
                "Business operations appear stable"
            )

        return recommendations

    # ========================================================
    # TOP ANOMALIES
    # ========================================================

    def _extract_top_anomalies(
        self,
        df: pd.DataFrame
    ) -> List[Dict]:

        anomalies = df[
            df["is_anomaly"] == 1
        ]

        anomalies = anomalies.sort_values(
            "anomaly_score",
            ascending=False
        )

        return anomalies.head(20).to_dict(
            orient="records"
        )

    # ========================================================
    # REASON GENERATOR
    # ========================================================

    def _generate_reason(
        self,
        row
    ) -> str:

        reasons = []

        if row["iforest_anomaly"] == 1:
            reasons.append(
                "Unusual ML behavior pattern"
            )

        if row["zscore_anomaly"] == 1:
            reasons.append(
                "Extreme statistical deviation"
            )

        if row["business_rule_anomaly"] == 1:
            reasons.append(
                "Business rule violation"
            )

        if row["cluster_anomaly"] == 1:
            reasons.append(
                "Customer outside normal clusters"
            )

        if not reasons:

            return "Normal behavior"

        return " | ".join(reasons)

    # ========================================================
    # SEVERITY
    # ========================================================

    def _classify_severity(
        self,
        score
    ) -> str:

        if score >= 8:
            return "critical"

        elif score >= 6:
            return "high"

        elif score >= 4:
            return "medium"

        elif score >= 2:
            return "low"

        return "normal"

    # ========================================================
    # BUSINESS HEALTH
    # ========================================================

    def _classify_health(
        self,
        anomaly_rate
    ) -> str:

        if anomaly_rate > 0.20:
            return "critical"

        elif anomaly_rate > 0.10:
            return "warning"

        elif anomaly_rate > 0.05:
            return "monitoring_required"

        return "healthy"


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def detect_anomalies(
    df: pd.DataFrame
):

    detector = AnomalyDetector()

    return detector.detect(df)