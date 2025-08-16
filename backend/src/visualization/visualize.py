import json
import os
import warnings
from datetime import datetime
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

warnings.filterwarnings("ignore")

# Set style for plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class MultisimVisualizer:
    """
    Comprehensive visualization class for Multisim dataset analysis and model results.
    Provides both static (matplotlib/seaborn) and interactive (plotly) visualizations.
    """

    def __init__(
        self, data_path: Optional[str] = None, model_path: Optional[str] = None
    ):
        """
        Initialize the visualizer.

        Args:
            data_path (Optional[str]): Path to the dataset
            model_path (Optional[str]): Path to the trained model
        """
        self.data = None
        self.model = None
        self.data_path = data_path
        self.model_path = model_path
        self.predictions = None
        self.probabilities = None

        # Color schemes
        self.colors = {
            "primary": "#3498db",
            "secondary": "#e74c3c",
            "success": "#2ecc71",
            "warning": "#f39c12",
            "multisim": "#e74c3c",
            "non_multisim": "#3498db",
        }

    def load_data(self, data_path: Optional[str] = None) -> bool:
        """
        Load dataset for visualization.

        Args:
            data_path (Optional[str]): Path to dataset file

        Returns:
            bool: True if successful, False otherwise
        """
        if data_path:
            self.data_path = data_path

        if not self.data_path:
            print("❌ No data path provided")
            return False

        print(f"📊 Loading data from: {self.data_path}")

        try:
            if self.data_path.endswith(".parquet"):
                self.data = pd.read_parquet(self.data_path)
            elif self.data_path.endswith(".csv"):
                self.data = pd.read_csv(self.data_path)
            else:
                print("❌ Unsupported file format. Use CSV or Parquet.")
                return False

            print(f"✅ Data loaded successfully! Shape: {self.data.shape}")
            return True

        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False

    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load trained model for prediction-based visualizations.

        Args:
            model_path (Optional[str]): Path to model file

        Returns:
            bool: True if successful, False otherwise
        """
        if model_path:
            self.model_path = model_path

        if not self.model_path or not os.path.exists(self.model_path):
            print("⚠️ Model not found. Prediction-based visualizations will be skipped.")
            return False

        try:
            self.model = joblib.load(self.model_path)
            print("✅ Model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False

    def create_data_overview_dashboard(self) -> None:
        """Create comprehensive data overview dashboard."""
        if self.data is None:
            print("❌ No data loaded. Please load data first.")
            return

        print("📊 Creating data overview dashboard...")

        # Create subplot figure
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "Multisim Dataset Overview Dashboard", fontsize=16, fontweight="bold"
        )

        # 1. Target distribution
        if "target" in self.data.columns:
            target_counts = self.data["target"].value_counts()
            labels = ["Non-Multisim", "Multisim"]
            colors = [self.colors["non_multisim"], self.colors["multisim"]]

            axes[0, 0].pie(
                target_counts.values,
                labels=labels,
                autopct="%1.1f%%",
                colors=colors,
                startangle=90,
            )
            axes[0, 0].set_title("Target Distribution")

        # 2. Age distribution by target
        if "age" in self.data.columns and "target" in self.data.columns:
            try:
                age_clean = pd.to_numeric(self.data["age"], errors="coerce")
                data_clean = pd.DataFrame(
                    {"age": age_clean, "target": self.data["target"]}
                ).dropna()

                if len(data_clean) > 0:
                    sns.boxplot(data=data_clean, x="target", y="age", ax=axes[0, 1])
                    axes[0, 1].set_title("Age Distribution by Target")
                    axes[0, 1].set_xticklabels(["Non-Multisim", "Multisim"])
            except Exception as e:
                print(e)
                axes[0, 1].text(
                    0.5, 0.5, "Age data unavailable", ha="center", va="center"
                )
                axes[0, 1].set_title("Age Distribution - No Data")

        # 3. Device age distribution
        if "age_dev" in self.data.columns:
            try:
                age_dev_clean = pd.to_numeric(
                    self.data["age_dev"], errors="coerce"
                ).dropna()
                axes[0, 2].hist(
                    age_dev_clean, bins=20, alpha=0.7, color=self.colors["primary"]
                )
                axes[0, 2].set_title("Device Age Distribution")
                axes[0, 2].set_xlabel("Device Age (months)")
            except Exception as e:
                print(e)
                axes[0, 2].text(
                    0.5, 0.5, "Device age data unavailable", ha="center", va="center"
                )
                axes[0, 2].set_title("Device Age - No Data")

        # 4. Gender distribution
        if "gndr" in self.data.columns:
            gender_counts = self.data["gndr"].value_counts()
            axes[1, 0].bar(
                gender_counts.index,
                gender_counts.values,
                color=[self.colors["primary"], self.colors["secondary"]],
            )
            axes[1, 0].set_title("Gender Distribution")
            axes[1, 0].set_ylabel("Count")

        # 5. Device type distribution
        device_cols = ["is_dualsim", "is_featurephone", "is_smartphone"]
        available_device_cols = [col for col in device_cols if col in self.data.columns]

        if available_device_cols:
            device_counts = []
            device_labels = []
            for col in available_device_cols:
                count = self.data[col].sum() if col in self.data.columns else 0
                device_counts.append(count)
                device_labels.append(col.replace("is_", "").title())

            axes[1, 1].bar(device_labels, device_counts, color=self.colors["success"])
            axes[1, 1].set_title("Device Type Distribution")
            axes[1, 1].tick_params(axis="x", rotation=45)

        # 6. Missing data heatmap
        missing_data = self.data.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

        if len(missing_data) > 0:
            axes[1, 2].barh(range(len(missing_data)), missing_data.values)
            axes[1, 2].set_yticks(range(len(missing_data)))
            axes[1, 2].set_yticklabels(missing_data.index)
            axes[1, 2].set_title("Missing Data Count")
            axes[1, 2].set_xlabel("Missing Values")
        else:
            axes[1, 2].text(0.5, 0.5, "No missing data", ha="center", va="center")
            axes[1, 2].set_title("Missing Data - None")

        plt.tight_layout()
        plt.show()

        print("✅ Data overview dashboard created!")

    def create_feature_analysis_plots(self) -> None:
        """Create detailed feature analysis visualizations."""
        if self.data is None:
            print("❌ No data loaded. Please load data first.")
            return

        print("🔍 Creating feature analysis plots...")

        # 1. Correlation heatmap
        try:
            numerical_data = self.data.select_dtypes(include=[np.number])

            # Convert object columns that might be numeric
            for col in self.data.select_dtypes(include=["object"]).columns:
                if col not in [
                    "gndr",
                    "dev_man",
                    "device_os_name",
                    "simcard_type",
                    "region",
                ]:
                    temp_numeric = pd.to_numeric(self.data[col], errors="coerce")
                    if temp_numeric.notna().sum() > len(self.data) * 0.5:
                        numerical_data[col] = temp_numeric

            if len(numerical_data.columns) > 1:
                plt.figure(figsize=(14, 10))
                correlation_matrix = numerical_data.corr()

                # Create mask for upper triangle
                mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

                sns.heatmap(
                    correlation_matrix,
                    mask=mask,
                    annot=True,
                    cmap="RdYlBu_r",
                    center=0,
                    square=True,
                    linewidths=0.5,
                    fmt=".2f",
                    cbar_kws={"shrink": 0.8},
                )
                plt.title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print(f"⚠️ Could not create correlation matrix: {str(e)}")

        # 2. Feature distributions by target
        if "target" in self.data.columns:
            # Select top numerical features for visualization
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != "target"][:6]

            if len(numeric_cols) > 0:
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                fig.suptitle(
                    "Feature Distributions by Target Class",
                    fontsize=16,
                    fontweight="bold",
                )
                axes = axes.ravel()

                for i, col in enumerate(numeric_cols):
                    if i < 6:
                        try:
                            data_clean = self.data[[col, "target"]].dropna()

                            # Create violin plot
                            sns.violinplot(
                                data=data_clean, x="target", y=col, ax=axes[i]
                            )
                            axes[i].set_title(f"{col} by Target")
                            axes[i].set_xticklabels(["Non-Multisim", "Multisim"])

                        except Exception as e:
                            axes[i].text(
                                0.5,
                                0.5,
                                f"Error: {str(e)[:30]}...",
                                ha="center",
                                va="center",
                            )
                            axes[i].set_title(f"{col} - Error")

                # Hide unused subplots
                for j in range(len(numeric_cols), 6):
                    axes[j].set_visible(False)

                plt.tight_layout()
                plt.show()

        print("✅ Feature analysis plots created!")

    def create_interactive_dashboard(self) -> None:
        """Create interactive Plotly dashboard."""
        if self.data is None:
            print("❌ No data loaded. Please load data first.")
            return

        print("🌐 Creating interactive dashboard...")

        try:
            # Create subplots
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Target Distribution",
                    "Age vs Tenure by Target",
                    "Device Age Distribution",
                    "Feature Correlation",
                ),
                specs=[
                    [{"type": "pie"}, {"type": "scatter"}],
                    [{"type": "histogram"}, {"type": "heatmap"}],
                ],
            )

            # 1. Target distribution pie chart
            if "target" in self.data.columns:
                target_counts = self.data["target"].value_counts()
                fig.add_trace(
                    go.Pie(
                        labels=["Non-Multisim", "Multisim"],
                        values=target_counts.values,
                        marker_colors=[
                            self.colors["non_multisim"],
                            self.colors["multisim"],
                        ],
                    ),
                    row=1,
                    col=1,
                )

            # 2. Age vs Tenure scatter plot
            if all(col in self.data.columns for col in ["age", "tenure", "target"]):
                scatter_data = self.data[["age", "tenure", "target"]].dropna()

                if len(scatter_data) > 0:
                    for target_val in [0, 1]:
                        subset = scatter_data[scatter_data["target"] == target_val]
                        fig.add_trace(
                            go.Scatter(
                                x=subset["age"],
                                y=subset["tenure"],
                                mode="markers",
                                name=f'{"Multisim" if target_val == 1 else "Non-Multisim"}',
                                marker=dict(
                                    color=(
                                        self.colors["multisim"]
                                        if target_val == 1
                                        else self.colors["non_multisim"]
                                    ),
                                    opacity=0.6,
                                ),
                            ),
                            row=1,
                            col=2,
                        )

            # 3. Device age histogram
            if "age_dev" in self.data.columns:
                age_dev_clean = pd.to_numeric(
                    self.data["age_dev"], errors="coerce"
                ).dropna()
                fig.add_trace(
                    go.Histogram(
                        x=age_dev_clean,
                        name="Device Age",
                        marker_color=self.colors["primary"],
                    ),
                    row=2,
                    col=1,
                )

            # 4. Correlation heatmap (simplified)
            try:
                numerical_data = self.data.select_dtypes(include=[np.number]).iloc[
                    :, :10
                ]  # First 10 numeric columns
                if len(numerical_data.columns) > 1:
                    corr_matrix = numerical_data.corr()

                    fig.add_trace(
                        go.Heatmap(
                            z=corr_matrix.values,
                            x=corr_matrix.columns,
                            y=corr_matrix.columns,
                            colorscale="RdYlBu",
                            zmid=0,
                        ),
                        row=2,
                        col=2,
                    )
            except Exception as e:
                print(e)

            # Update layout
            fig.update_layout(
                title_text="Multisim Dataset Interactive Dashboard",
                title_x=0.5,
                height=800,
                showlegend=True,
            )

            fig.show()
            print("✅ Interactive dashboard created!")

        except Exception as e:
            print(f"❌ Error creating interactive dashboard: {str(e)}")

    def visualize_model_performance(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray
    ) -> None:
        """
        Create comprehensive model performance visualizations.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
        """
        print("📈 Creating model performance visualizations...")

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle("Model Performance Analysis", fontsize=16, fontweight="bold")

        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0])
        axes[0, 0].set_title("Confusion Matrix")
        axes[0, 0].set_xlabel("Predicted")
        axes[0, 0].set_ylabel("Actual")
        axes[0, 0].set_xticklabels(["Non-Multisim", "Multisim"])
        axes[0, 0].set_yticklabels(["Non-Multisim", "Multisim"])

        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        axes[0, 1].plot(
            fpr,
            tpr,
            color=self.colors["primary"],
            label=f"ROC curve (AUC = {roc_auc:.3f})",
            linewidth=2,
        )
        axes[0, 1].plot([0, 1], [0, 1], "k--", linewidth=1)
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel("False Positive Rate")
        axes[0, 1].set_ylabel("True Positive Rate")
        axes[0, 1].set_title("ROC Curve")
        axes[0, 1].legend(loc="lower right")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)

        axes[0, 2].plot(
            recall,
            precision,
            color=self.colors["secondary"],
            label=f"PR curve (AUC = {pr_auc:.3f})",
            linewidth=2,
        )
        axes[0, 2].set_xlabel("Recall")
        axes[0, 2].set_ylabel("Precision")
        axes[0, 2].set_title("Precision-Recall Curve")
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Prediction probability distribution
        axes[1, 0].hist(
            y_pred_proba[y_true == 0],
            bins=30,
            alpha=0.7,
            label="Non-Multisim",
            color=self.colors["non_multisim"],
        )
        axes[1, 0].hist(
            y_pred_proba[y_true == 1],
            bins=30,
            alpha=0.7,
            label="Multisim",
            color=self.colors["multisim"],
        )
        axes[1, 0].set_xlabel("Prediction Probability")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Prediction Probability Distribution")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Threshold analysis
        thresholds = np.linspace(0, 1, 100)
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)

            if len(np.unique(y_pred_thresh)) > 1:
                from sklearn.metrics import f1_score, precision_score, recall_score

                precision_scores.append(precision_score(y_true, y_pred_thresh))
                recall_scores.append(recall_score(y_true, y_pred_thresh))
                f1_scores.append(f1_score(y_true, y_pred_thresh))
            else:
                precision_scores.append(0)
                recall_scores.append(0)
                f1_scores.append(0)

        axes[1, 1].plot(
            thresholds,
            precision_scores,
            label="Precision",
            color=self.colors["primary"],
        )
        axes[1, 1].plot(
            thresholds, recall_scores, label="Recall", color=self.colors["secondary"]
        )
        axes[1, 1].plot(
            thresholds, f1_scores, label="F1-Score", color=self.colors["success"]
        )
        axes[1, 1].set_xlabel("Threshold")
        axes[1, 1].set_ylabel("Score")
        axes[1, 1].set_title("Metrics vs Threshold")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Feature importance (if model available)
        if self.model and hasattr(
            self.model.named_steps.get("model", None), "feature_importances_"
        ):
            importances = self.model.named_steps["model"].feature_importances_
            top_n = min(15, len(importances))
            top_indices = np.argsort(importances)[-top_n:]

            axes[1, 2].barh(
                range(top_n), importances[top_indices], color=self.colors["warning"]
            )
            axes[1, 2].set_yticks(range(top_n))
            axes[1, 2].set_yticklabels([f"Feature_{i}" for i in top_indices])
            axes[1, 2].set_xlabel("Importance")
            axes[1, 2].set_title(f"Top {top_n} Feature Importances")
        else:
            axes[1, 2].text(
                0.5, 0.5, "Feature importance\nnot available", ha="center", va="center"
            )
            axes[1, 2].set_title("Feature Importance - N/A")

        plt.tight_layout()
        plt.show()

        print("✅ Model performance visualizations created!")

    def create_interactive_model_analysis(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray
    ) -> None:
        """Create interactive model performance dashboard."""
        print("🌐 Creating interactive model analysis...")

        try:
            # Create subplots
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "ROC Curve",
                    "Precision-Recall Curve",
                    "Prediction Distribution",
                    "Confusion Matrix",
                ),
                specs=[
                    [{"type": "scatter"}, {"type": "scatter"}],
                    [{"type": "histogram"}, {"type": "heatmap"}],
                ],
            )

            # 1. Interactive ROC Curve
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"ROC Curve (AUC = {roc_auc:.3f})",
                    line=dict(color=self.colors["primary"], width=3),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Random Classifier",
                    line=dict(color="gray", dash="dash"),
                ),
                row=1,
                col=1,
            )

            # 2. Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            pr_auc = auc(recall, precision)

            fig.add_trace(
                go.Scatter(
                    x=recall,
                    y=precision,
                    mode="lines",
                    name=f"PR Curve (AUC = {pr_auc:.3f})",
                    line=dict(color=self.colors["secondary"], width=3),
                ),
                row=1,
                col=2,
            )

            # 3. Prediction probability distribution
            fig.add_trace(
                go.Histogram(
                    x=y_pred_proba[y_true == 0],
                    name="Non-Multisim",
                    marker_color=self.colors["non_multisim"],
                    opacity=0.7,
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Histogram(
                    x=y_pred_proba[y_true == 1],
                    name="Multisim",
                    marker_color=self.colors["multisim"],
                    opacity=0.7,
                ),
                row=2,
                col=1,
            )

            # 4. Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            fig.add_trace(
                go.Heatmap(
                    z=cm,
                    x=["Non-Multisim", "Multisim"],
                    y=["Non-Multisim", "Multisim"],
                    colorscale="Blues",
                    text=cm,
                    texttemplate="%{text}",
                    textfont={"size": 20},
                ),
                row=2,
                col=2,
            )

            # Update layout
            fig.update_layout(
                title_text="Interactive Model Performance Dashboard",
                title_x=0.5,
                height=800,
                showlegend=True,
            )

            fig.show()
            print("✅ Interactive model analysis created!")

        except Exception as e:
            print(f"❌ Error creating interactive analysis: {str(e)}")

    def create_customer_segmentation_viz(self) -> None:
        """Create customer segmentation visualizations."""
        if self.data is None:
            print("❌ No data loaded. Please load data first.")
            return

        print("👥 Creating customer segmentation visualizations...")

        try:
            # Prepare data for dimensionality reduction
            numerical_data = self.data.select_dtypes(include=[np.number])

            # Remove target if present
            if "target" in numerical_data.columns:
                target = numerical_data["target"]
                numerical_data = numerical_data.drop("target", axis=1)
            else:
                target = None

            # Fill missing values
            numerical_data_filled = numerical_data.fillna(numerical_data.median())

            if (
                len(numerical_data_filled.columns) >= 2
                and len(numerical_data_filled) > 50
            ):
                # PCA visualization
                pca = PCA(n_components=2, random_state=42)
                pca_result = pca.fit_transform(numerical_data_filled)

                plt.figure(figsize=(15, 6))

                # PCA plot
                plt.subplot(1, 2, 1)
                if target is not None:
                    scatter = plt.scatter(
                        pca_result[:, 0],
                        pca_result[:, 1],
                        c=target,
                        cmap="RdYlBu",
                        alpha=0.6,
                    )
                    plt.colorbar(scatter, label="Target (0: Non-Multisim, 1: Multisim)")
                else:
                    plt.scatter(
                        pca_result[:, 0],
                        pca_result[:, 1],
                        alpha=0.6,
                        color=self.colors["primary"],
                    )

                plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
                plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
                plt.title("Customer Segmentation - PCA")
                plt.grid(True, alpha=0.3)

                # t-SNE visualization (for smaller datasets)
                if len(numerical_data_filled) <= 5000:
                    plt.subplot(1, 2, 2)
                    tsne = TSNE(
                        n_components=2,
                        random_state=42,
                        perplexity=min(30, len(numerical_data_filled) - 1),
                    )
                    tsne_result = tsne.fit_transform(
                        numerical_data_filled.sample(
                            min(1000, len(numerical_data_filled))
                        )
                    )

                    if target is not None:
                        target_sample = target.sample(
                            min(1000, len(numerical_data_filled))
                        ).values
                        scatter = plt.scatter(
                            tsne_result[:, 0],
                            tsne_result[:, 1],
                            c=target_sample,
                            cmap="RdYlBu",
                            alpha=0.6,
                        )
                        plt.colorbar(
                            scatter, label="Target (0: Non-Multisim, 1: Multisim)"
                        )
                    else:
                        plt.scatter(
                            tsne_result[:, 0],
                            tsne_result[:, 1],
                            alpha=0.6,
                            color=self.colors["secondary"],
                        )

                    plt.xlabel("t-SNE 1")
                    plt.ylabel("t-SNE 2")
                    plt.title("Customer Segmentation - t-SNE")
                    plt.grid(True, alpha=0.3)
                else:
                    plt.subplot(1, 2, 2)
                    plt.text(
                        0.5,
                        0.5,
                        "Dataset too large for t-SNE\n(>5000 samples)",
                        ha="center",
                        va="center",
                        transform=plt.gca().transAxes,
                    )
                    plt.title("t-SNE - Skipped")

                plt.tight_layout()
                plt.show()

        except Exception as e:
            print(f"❌ Error creating segmentation visualizations: {str(e)}")

        print("✅ Customer segmentation visualizations created!")

    def create_business_insights_dashboard(self) -> None:
        """Create business-focused insights dashboard."""
        if self.data is None:
            print("❌ No data loaded. Please load data first.")
            return

        print("💼 Creating business insights dashboard...")

        try:
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle("Business Insights Dashboard", fontsize=16, fontweight="bold")

            # 1. Multisim adoption by age group
            if "age" in self.data.columns and "target" in self.data.columns:
                age_clean = pd.to_numeric(self.data["age"], errors="coerce")
                data_age = pd.DataFrame(
                    {"age": age_clean, "target": self.data["target"]}
                ).dropna()

                if len(data_age) > 0:
                    # Create age groups
                    data_age["age_group"] = pd.cut(
                        data_age["age"],
                        bins=[0, 25, 35, 45, 55, 100],
                        labels=["18-25", "26-35", "36-45", "46-55", "55+"],
                    )

                    adoption_by_age = (
                        data_age.groupby("age_group")["target"]
                        .agg(["count", "sum", "mean"])
                        .reset_index()
                    )
                    adoption_by_age["adoption_rate"] = adoption_by_age["mean"] * 100

                    bars = axes[0, 0].bar(
                        adoption_by_age["age_group"],
                        adoption_by_age["adoption_rate"],
                        color=self.colors["primary"],
                        alpha=0.7,
                    )
                    axes[0, 0].set_title("Multisim Adoption Rate by Age Group")
                    axes[0, 0].set_ylabel("Adoption Rate (%)")
                    axes[0, 0].tick_params(axis="x", rotation=45)

                    # Add value labels on bars
                    for bar, rate in zip(bars, adoption_by_age["adoption_rate"]):
                        axes[0, 0].text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.5,
                            f"{rate:.1f}%",
                            ha="center",
                            va="bottom",
                        )

            # 2. Device type analysis
            device_cols = ["is_dualsim", "is_featurephone", "is_smartphone"]
            available_device_cols = [
                col for col in device_cols if col in self.data.columns
            ]

            if available_device_cols and "target" in self.data.columns:
                device_adoption = {}
                for col in available_device_cols:
                    device_users = self.data[self.data[col] == 1]
                    if len(device_users) > 0:
                        adoption_rate = device_users["target"].mean() * 100
                        device_adoption[col.replace("is_", "").title()] = adoption_rate

                if device_adoption:
                    axes[0, 1].bar(
                        device_adoption.keys(),
                        device_adoption.values(),
                        color=self.colors["secondary"],
                        alpha=0.7,
                    )
                    axes[0, 1].set_title("Multisim Adoption by Device Type")
                    axes[0, 1].set_ylabel("Adoption Rate (%)")
                    axes[0, 1].tick_params(axis="x", rotation=45)

            # 3. Regional analysis
            if "region" in self.data.columns and "target" in self.data.columns:
                regional_data = (
                    self.data.groupby("region")["target"]
                    .agg(["count", "sum", "mean"])
                    .reset_index()
                )
                regional_data["adoption_rate"] = regional_data["mean"] * 100

                axes[0, 2].bar(
                    regional_data["region"],
                    regional_data["adoption_rate"],
                    color=self.colors["success"],
                    alpha=0.7,
                )
                axes[0, 2].set_title("Multisim Adoption by Region")
                axes[0, 2].set_ylabel("Adoption Rate (%)")
                axes[0, 2].tick_params(axis="x", rotation=45)

            # 4. Tenure vs Multisim adoption
            if "tenure" in self.data.columns and "target" in self.data.columns:
                tenure_clean = pd.to_numeric(self.data["tenure"], errors="coerce")
                data_tenure = pd.DataFrame(
                    {"tenure": tenure_clean, "target": self.data["target"]}
                ).dropna()

                if len(data_tenure) > 0:
                    # Create tenure groups
                    data_tenure["tenure_group"] = pd.cut(
                        data_tenure["tenure"],
                        bins=[0, 180, 365, 730, 1095, 10000],
                        labels=["0-6m", "6m-1y", "1-2y", "2-3y", "3y+"],
                    )

                    tenure_adoption = (
                        data_tenure.groupby("tenure_group")["target"].mean() * 100
                    )

                    axes[1, 0].plot(
                        tenure_adoption.index,
                        tenure_adoption.values,
                        marker="o",
                        linewidth=3,
                        markersize=8,
                        color=self.colors["warning"],
                    )
                    axes[1, 0].set_title("Multisim Adoption by Tenure")
                    axes[1, 0].set_ylabel("Adoption Rate (%)")
                    axes[1, 0].tick_params(axis="x", rotation=45)
                    axes[1, 0].grid(True, alpha=0.3)

            # 5. Device manufacturer analysis
            if "dev_man" in self.data.columns and "target" in self.data.columns:
                manufacturer_data = (
                    self.data.groupby("dev_man")["target"]
                    .agg(["count", "mean"])
                    .reset_index()
                )
                manufacturer_data = manufacturer_data[
                    manufacturer_data["count"] >= 10
                ]  # Filter small groups
                manufacturer_data["adoption_rate"] = manufacturer_data["mean"] * 100
                manufacturer_data = manufacturer_data.sort_values(
                    "adoption_rate", ascending=True
                ).tail(10)

                axes[1, 1].barh(
                    manufacturer_data["dev_man"],
                    manufacturer_data["adoption_rate"],
                    color=self.colors["primary"],
                    alpha=0.7,
                )
                axes[1, 1].set_title("Top 10 Manufacturers by Multisim Adoption")
                axes[1, 1].set_xlabel("Adoption Rate (%)")

            # 6. Customer value segmentation
            if all(col in self.data.columns for col in ["tenure", "age", "target"]):
                # Create a simple customer value score
                tenure_clean = pd.to_numeric(
                    self.data["tenure"], errors="coerce"
                ).fillna(0)
                age_clean = pd.to_numeric(self.data["age"], errors="coerce").fillna(0)

                # Normalize values
                tenure_norm = (tenure_clean - tenure_clean.min()) / (
                    tenure_clean.max() - tenure_clean.min() + 1e-8
                )
                age_norm = (age_clean - age_clean.min()) / (
                    age_clean.max() - age_clean.min() + 1e-8
                )

                value_score = (tenure_norm + age_norm) / 2

                # Create value segments
                value_segments = pd.cut(
                    value_score, bins=3, labels=["Low", "Medium", "High"]
                )
                segment_data = pd.DataFrame(
                    {"segment": value_segments, "target": self.data["target"]}
                )

                segment_adoption = (
                    segment_data.groupby("segment")["target"].mean() * 100
                )

                axes[1, 2].bar(
                    segment_adoption.index,
                    segment_adoption.values,
                    color=[
                        self.colors["warning"],
                        self.colors["primary"],
                        self.colors["success"],
                    ],
                    alpha=0.7,
                )
                axes[1, 2].set_title("Multisim Adoption by Customer Value")
                axes[1, 2].set_ylabel("Adoption Rate (%)")

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"❌ Error creating business dashboard: {str(e)}")

        print("✅ Business insights dashboard created!")

    def create_prediction_confidence_analysis(self, y_pred_proba: np.ndarray) -> None:
        """Analyze prediction confidence levels."""
        print("🎯 Creating prediction confidence analysis...")

        try:
            # Calculate confidence scores
            confidence_scores = (
                np.max(y_pred_proba, axis=1) if y_pred_proba.ndim > 1 else y_pred_proba
            )

            # Create confidence segments
            confidence_segments = pd.cut(
                confidence_scores,
                bins=[0, 0.6, 0.8, 0.9, 1.0],
                labels=[
                    "Low (0-60%)",
                    "Medium (60-80%)",
                    "High (80-90%)",
                    "Very High (90-100%)",
                ],
            )

            plt.figure(figsize=(15, 10))

            # 1. Confidence distribution
            plt.subplot(2, 3, 1)
            plt.hist(
                confidence_scores, bins=30, alpha=0.7, color=self.colors["primary"]
            )
            plt.axvline(
                confidence_scores.mean(),
                color="red",
                linestyle="--",
                label=f"Mean: {confidence_scores.mean():.3f}",
            )
            plt.xlabel("Confidence Score")
            plt.ylabel("Frequency")
            plt.title("Prediction Confidence Distribution")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 2. Confidence segments
            plt.subplot(2, 3, 2)
            segment_counts = confidence_segments.value_counts()
            plt.pie(
                segment_counts.values, labels=segment_counts.index, autopct="%1.1f%%"
            )
            plt.title("Confidence Level Segments")

            # 3. Probability distribution for each class
            plt.subplot(2, 3, 3)
            if y_pred_proba.ndim > 1:
                plt.hist(
                    y_pred_proba[:, 0],
                    bins=30,
                    alpha=0.7,
                    label="Non-Multisim Prob",
                    color=self.colors["non_multisim"],
                )
                plt.hist(
                    y_pred_proba[:, 1],
                    bins=30,
                    alpha=0.7,
                    label="Multisim Prob",
                    color=self.colors["multisim"],
                )
            else:
                plt.hist(y_pred_proba, bins=30, alpha=0.7, color=self.colors["primary"])
            plt.xlabel("Probability")
            plt.ylabel("Frequency")
            plt.title("Class Probability Distribution")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 4. Confidence vs Prediction scatter
            plt.subplot(2, 3, 4)
            multisim_probs = (
                y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba
            )
            plt.scatter(
                multisim_probs,
                confidence_scores,
                alpha=0.6,
                color=self.colors["primary"],
            )
            plt.xlabel("Multisim Probability")
            plt.ylabel("Confidence Score")
            plt.title("Confidence vs Multisim Probability")
            plt.grid(True, alpha=0.3)

            # 5. Confidence by prediction
            plt.subplot(2, 3, 5)
            predictions = (multisim_probs >= 0.5).astype(int)
            conf_by_pred = [
                confidence_scores[predictions == 0],
                confidence_scores[predictions == 1],
            ]

            bp = plt.boxplot(
                conf_by_pred, labels=["Non-Multisim", "Multisim"], patch_artist=True
            )
            bp["boxes"][0].set_facecolor(self.colors["non_multisim"])
            bp["boxes"][1].set_facecolor(self.colors["multisim"])
            plt.ylabel("Confidence Score")
            plt.title("Confidence by Prediction Class")
            plt.grid(True, alpha=0.3)

            # 6. High confidence predictions summary
            plt.subplot(2, 3, 6)
            high_conf_mask = confidence_scores > 0.8
            high_conf_preds = predictions[high_conf_mask]

            if len(high_conf_preds) > 0:
                high_conf_counts = pd.Series(high_conf_preds).value_counts()
                plt.bar(
                    ["Non-Multisim", "Multisim"],
                    [high_conf_counts.get(0, 0), high_conf_counts.get(1, 0)],
                    color=[self.colors["non_multisim"], self.colors["multisim"]],
                    alpha=0.7,
                )
                plt.title("High Confidence Predictions\n(>80% confidence)")
                plt.ylabel("Count")

                # Add percentage labels
                total_high_conf = len(high_conf_preds)
                for i, count in enumerate(
                    [high_conf_counts.get(0, 0), high_conf_counts.get(1, 0)]
                ):
                    if total_high_conf > 0:
                        pct = (count / total_high_conf) * 100
                        plt.text(
                            i,
                            count + max(high_conf_counts.values) * 0.01,
                            f"{pct:.1f}%",
                            ha="center",
                            va="bottom",
                        )
            else:
                plt.text(
                    0.5,
                    0.5,
                    "No high confidence\npredictions",
                    ha="center",
                    va="center",
                )
                plt.title("High Confidence Predictions - None")

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"❌ Error creating confidence analysis: {str(e)}")

        print("✅ Prediction confidence analysis created!")

    def save_visualization_report(
        self, output_dir: str = "visualization_reports"
    ) -> bool:
        """
        Save all visualizations and analysis to files.

        Args:
            output_dir (str): Directory to save reports

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            print(f"💾 Saving visualization reports to: {output_dir}")

            # Create summary report
            report = {
                "timestamp": timestamp,
                "data_path": self.data_path,
                "model_path": self.model_path,
                "data_summary": {},
                "visualization_summary": {},
            }

            if self.data is not None:
                report["data_summary"] = {
                    "total_samples": len(self.data),
                    "total_features": len(self.data.columns),
                    "missing_data_cols": self.data.isnull().sum().sum(),
                    "target_distribution": (
                        self.data["target"].value_counts().to_dict()
                        if "target" in self.data.columns
                        else None
                    ),
                }

            # Save report
            report_path = os.path.join(
                output_dir, f"visualization_report_{timestamp}.json"
            )
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)

            print("✅ Visualization report saved!")
            return True

        except Exception as e:
            print(f"❌ Error saving visualization report: {str(e)}")
            return False

    def run_complete_visualization_suite(
        self, data_path: str, model_path: Optional[str] = None
    ) -> bool:
        """
        Run complete visualization suite with all available plots.

        Args:
            data_path (str): Path to dataset
            model_path (Optional[str]): Path to trained model

        Returns:
            bool: True if successful, False otherwise
        """
        print("🎨 Starting Complete Visualization Suite")
        print("=" * 60)

        # Load data
        if not self.load_data(data_path):
            return False

        # Load model if provided
        if model_path:
            self.load_model(model_path)

        # 1. Data overview dashboard
        self.create_data_overview_dashboard()

        # 2. Feature analysis
        self.create_feature_analysis_plots()

        # 3. Interactive dashboard
        self.create_interactive_dashboard()

        # 4. Customer segmentation
        self.create_customer_segmentation_viz()

        # 5. Business insights
        self.create_business_insights_dashboard()

        # 6. Model-specific visualizations (if model available)
        if self.model is not None:
            try:
                # Make predictions for visualization
                X = (
                    self.data.drop("target", axis=1)
                    if "target" in self.data.columns
                    else self.data
                )
                y_true = self.data["target"] if "target" in self.data.columns else None

                # Preprocess data (basic preprocessing)
                X_processed = self._preprocess_for_prediction(X)

                y_pred = self.model.predict(X_processed)
                y_pred_proba = self.model.predict_proba(X_processed)

                # Store predictions for other methods
                self.predictions = y_pred
                self.probabilities = y_pred_proba

                if y_true is not None:
                    self.visualize_model_performance(y_true, y_pred, y_pred_proba[:, 1])
                    self.create_interactive_model_analysis(
                        y_true, y_pred, y_pred_proba[:, 1]
                    )

                self.create_prediction_confidence_analysis(y_pred_proba)

            except Exception as e:
                print(f"⚠️ Could not create model visualizations: {str(e)}")

        # 7. Save visualization report
        self.save_visualization_report()

        print("✅ Complete visualization suite finished!")
        return True

    def _preprocess_for_prediction(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Basic preprocessing for prediction visualization.

        Args:
            X (pd.DataFrame): Input features

        Returns:
            pd.DataFrame: Preprocessed features
        """
        X_processed = X.copy()

        # Convert specified columns to int type
        int_columns = [
            "age_dev",
            "dev_num",
            "is_dualsim",
            "is_featurephone",
            "is_smartphone",
        ]
        for col in int_columns:
            if col in X_processed.columns:
                try:
                    X_processed[col] = (
                        pd.to_numeric(X_processed[col], errors="coerce")
                        .fillna(0)
                        .astype(int)
                    )
                except Exception as e:
                    print(e)

        # Convert age to numeric
        if "age" in X_processed.columns:
            try:
                if X_processed["age"].dtype == "object":
                    X_processed["age"] = pd.to_numeric(
                        X_processed["age"], errors="coerce"
                    )
            except Exception as e:
                print(e)

        # Create age_gender_combined feature
        if "age" in X_processed.columns and "gndr" in X_processed.columns:
            try:
                age_filled = X_processed["age"].fillna("unknown")
                gndr_filled = X_processed["gndr"].fillna("U")
                X_processed["age_gender_combined"] = (
                    age_filled.astype(str) + "_" + gndr_filled.astype(str)
                )
            except Exception as e:
                print(e)

        # Drop unnecessary columns
        cols_to_drop = []
        val_cols = [col for col in X_processed.columns if col.startswith("val")]
        cols_to_drop.extend(val_cols)

        temp_cols = [
            col
            for col in X_processed.columns
            if col.startswith(("temp_", "tmp_", "test_"))
        ]
        cols_to_drop.extend(temp_cols)

        if "telephone_number" in X_processed.columns:
            cols_to_drop.append("telephone_number")

        existing_cols_to_drop = [
            col for col in cols_to_drop if col in X_processed.columns
        ]
        if existing_cols_to_drop:
            X_processed = X_processed.drop(existing_cols_to_drop, axis=1)

        return X_processed


def create_sample_visualizations():
    """Create sample visualizations with dummy data."""
    print("🎨 Creating sample visualizations...")

    # Generate sample data
    np.random.seed(42)
    n_samples = 1000

    sample_data = pd.DataFrame(
        {
            "age": np.random.normal(35, 10, n_samples),
            "tenure": np.random.exponential(500, n_samples),
            "age_dev": np.random.randint(1, 60, n_samples),
            "gndr": np.random.choice(["M", "F"], n_samples),
            "target": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        }
    )

    # Create simple visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Sample Multisim Visualizations", fontsize=16)

    # Target distribution
    target_counts = sample_data["target"].value_counts()
    axes[0, 0].pie(
        target_counts.values,
        labels=["Non-Multisim", "Multisim"],
        autopct="%1.1f%%",
        colors=["#3498db", "#e74c3c"],
    )
    axes[0, 0].set_title("Target Distribution")

    # Age distribution by target
    sns.boxplot(data=sample_data, x="target", y="age", ax=axes[0, 1])
    axes[0, 1].set_title("Age by Target")
    axes[0, 1].set_xticklabels(["Non-Multisim", "Multisim"])

    # Tenure distribution
    axes[1, 0].hist(sample_data["tenure"], bins=30, alpha=0.7, color="#2ecc71")
    axes[1, 0].set_title("Tenure Distribution")
    axes[1, 0].set_xlabel("Tenure (days)")

    # Device age vs target
    sns.boxplot(data=sample_data, x="target", y="age_dev", ax=axes[1, 1])
    axes[1, 1].set_title("Device Age by Target")
    axes[1, 1].set_xticklabels(["Non-Multisim", "Multisim"])

    plt.tight_layout()
    plt.show()

    print("✅ Sample visualizations created!")


def main():
    """Main function to demonstrate visualization capabilities."""
    print("🎨 Multisim Dataset Visualizer")
    print("=" * 50)

    # Initialize visualizer
    visualizer = MultisimVisualizer()

    # Check for data file
    data_files = [
        "multisim_dataset.parquet",
        "multisim_dataset.csv",
        "data.parquet",
        "data.csv",
    ]
    data_found = False

    for data_file in data_files:
        if os.path.exists(data_file):
            print(f"📊 Found data file: {data_file}")
            if visualizer.load_data(data_file):
                data_found = True
                break

    if not data_found:
        print("⚠️ No dataset found. Creating sample visualizations...")
        create_sample_visualizations()
        return

    # Check for model file
    model_files = ["best_multisim_model.pkl", "multisim_model.pkl", "model.pkl"]
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"🤖 Found model file: {model_file}")
            visualizer.load_model(model_file)
            break

    # Run complete visualization suite
    try:
        print("\n🚀 Running complete visualization suite...")

        # Data overview
        print("\n1️⃣ Creating data overview dashboard...")
        visualizer.create_data_overview_dashboard()

        # Feature analysis
        print("\n2️⃣ Creating feature analysis plots...")
        visualizer.create_feature_analysis_plots()

        # Interactive dashboard
        print("\n3️⃣ Creating interactive dashboard...")
        visualizer.create_interactive_dashboard()

        # Customer segmentation
        print("\n4️⃣ Creating customer segmentation visualizations...")
        visualizer.create_customer_segmentation_viz()

        # Business insights
        print("\n5️⃣ Creating business insights dashboard...")
        visualizer.create_business_insights_dashboard()

        # Model-specific visualizations (if model loaded)
        if visualizer.model is not None:
            print("\n6️⃣ Creating model performance visualizations...")

            # Prepare data for model evaluation
            X = (
                visualizer.data.drop("target", axis=1)
                if "target" in visualizer.data.columns
                else visualizer.data
            )
            y_true = (
                visualizer.data["target"]
                if "target" in visualizer.data.columns
                else None
            )

            if y_true is not None:
                try:
                    # Preprocess and predict
                    X_processed = visualizer._preprocess_for_prediction(X)
                    y_pred = visualizer.model.predict(X_processed)
                    y_pred_proba = visualizer.model.predict_proba(X_processed)

                    # Create model visualizations
                    visualizer.visualize_model_performance(
                        y_true, y_pred, y_pred_proba[:, 1]
                    )
                    visualizer.create_interactive_model_analysis(
                        y_true, y_pred, y_pred_proba[:, 1]
                    )
                    visualizer.create_prediction_confidence_analysis(y_pred_proba)

                except Exception as e:
                    print(f"⚠️ Could not create model visualizations: {str(e)}")

        # Save reports
        print("\n7️⃣ Saving visualization reports...")
        visualizer.save_visualization_report()

        print("\n🎉 Complete visualization suite finished successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Error in visualization suite: {str(e)}")


if __name__ == "__main__":
    main()
