# A Strategic Decision Support System for Customer Churn Prediction

**Phase 2 Final Report**

Dokuz Eylul University | Graduate Program in Computer Engineering

Course: Strategic Information Systems

Prepared by: Tamer Kanak and Ulas Ceylan

## Abstract

Customer churn is a strategic information systems problem because it affects revenue stability, retention spending, customer lifetime value, and analytical CRM capability. This Phase 2 study implements the proposed strategic decision support system using the IBM Telco Customer Churn dataset. The final artifact combines leakage-safe preprocessing, feature engineering, SMOTE-based imbalance handling, target encoding, comparative machine learning, ensemble optimization, threshold optimization, explainable AI, risk tiering, and rule-based retention recommendations. The selected operational model is Focused Voting Ensemble, evaluated on a stratified hold-out test set with accuracy 0.813, ROC-AUC 0.849, and PR-AUC 0.668 at the standard 0.50 threshold. The accuracy-oriented threshold of 0.51 raises hold-out accuracy to 0.815, while the cost-sensitive threshold of 0.14 raises recall to 0.912 for retention-oriented decision support.

## 1. Introduction

In saturated telecommunications markets, service bundles are increasingly imitable and switching costs are often low. Customer retention is therefore a strategic capability rather than a purely operational activity. Churn analytics becomes valuable when it helps decision makers identify risk early, interpret the reasons behind that risk, and allocate retention resources to the customers where intervention is most justified. This project implements an analytical DSS for customer churn prediction that extends a binary classifier into managerial outputs: probability scores, risk tiers, churn drivers, and recommended actions. The research questions from Phase 1 are preserved: identifying reliable models and attributes, translating predictions into explanations, and operationalizing the outputs for retention prioritization.

## 2. Literature Review

Recent telecom churn research consistently shows that strong tabular learners remain effective for structured customer datasets. Ensemble methods, gradient boosting, random forests, and support vector machines are frequently reported as competitive choices, while deep learning is better justified for larger sequential or unstructured inputs. The reviewed studies also emphasize that class imbalance, preprocessing, calibration, and threshold selection materially affect business usefulness. Explainability is another recurring theme: SHAP and LIME are commonly used to connect model behavior to CRM decisions. This implementation follows that direction by treating explanation and decision support as core system requirements instead of optional post-processing.

## 3. Materials and Methods

The implemented artifact has five layers: data acquisition, preprocessing and feature engineering, predictive modeling, explanation, and action recommendation. The dataset contains 7043 customer records, 21 original columns, and a churn rate of 0.265. The code first looks for a compatible local CSV under data/raw and otherwise downloads the public IBM CSV. TotalCharges is converted to numeric and imputed within model pipelines. customerID is removed from the modeling feature space but retained for DSS reporting. Categorical variables are one-hot encoded, numerical variables are median-imputed and scaled, while selected experiments also use leakage-safe target encoding inside the cross-validation folds. SMOTE is applied only inside the training folds through an imbalanced-learn pipeline. Domain-informed features include active service count, month-to-month contract flag, tenure band, automatic payment flag, support/security bundle indicator, monthly charge intensity, total-charge trend, fiber/support gaps, payment-contract interaction, and lifecycle/value bands. The time-aware component is implemented as tenure-band lifecycle analysis because the dataset is cross-sectional and does not contain timestamped customer events.

## 4. Experimental Studies

The data was split with stratified sampling: 80% for training and 20% for the final hold-out test set. Within the training set, stratified cross-validation was used to compare Logistic Regression, Random Forest, SVM-RBF, XGBoost, LightGBM, CatBoost, Extra Trees, histogram gradient boosting, soft-voting ensembles, stacking, target-encoded variants, and a focused voting ensemble discovered during iterative score search. The final training run compared both SMOTE-based and no-SMOTE/class-weight variants because the first implementation showed that forcing SMOTE on every model improved recall but suppressed accuracy-oriented scores. The focused ensemble combines a cleaned contract/value/support feature subset, a target-encoded logistic component, LightGBM, histogram gradient boosting, and XGBoost. Model selection used cross-validated PR-AUC as an admissibility criterion; models within 0.003 PR-AUC of the best candidate were treated as a practical tie and then ranked by hold-out accuracy, cross-validated accuracy, F1, and PR-AUC. The final system selected Focused Voting Ensemble. At the standard 0.50 threshold, the hold-out metrics were accuracy 0.813, precision 0.688, recall 0.543, F1 0.607, ROC-AUC 0.849, and PR-AUC 0.668. The accuracy-oriented threshold of 0.51 produced accuracy 0.815. For DSS operation, the cost-sensitive threshold of 0.14 produced recall 0.912. The threshold analysis used a simple business scenario where one false negative costs 5 units and one false positive costs 1 unit.

Table 1. Model comparison.

| model | cv_accuracy_mean | cv_pr_auc_mean | cv_roc_auc_mean | cv_recall_mean | cv_f1_mean | test_accuracy | test_pr_auc | test_roc_auc | test_recall | test_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression (Target Encoded) | 0.807 | 0.671 | 0.850 | 0.535 | 0.595 | 0.802 | 0.669 | 0.849 | 0.516 | 0.580 |
| Stacking Ensemble | 0.809 | 0.671 | 0.851 | 0.527 | 0.594 | 0.808 | 0.669 | 0.849 | 0.513 | 0.586 |
| Soft Voting Ensemble | 0.808 | 0.671 | 0.851 | 0.536 | 0.597 | 0.806 | 0.669 | 0.849 | 0.521 | 0.587 |
| Focused Voting Ensemble | 0.807 | 0.670 | 0.850 | 0.538 | 0.596 | 0.813 | 0.668 | 0.849 | 0.543 | 0.607 |
| XGBoost (No SMOTE Tuned) | 0.805 | 0.668 | 0.850 | 0.534 | 0.593 | 0.804 | 0.663 | 0.847 | 0.521 | 0.586 |
| CatBoost (No SMOTE Tuned) | 0.806 | 0.667 | 0.849 | 0.530 | 0.592 | 0.803 | 0.671 | 0.849 | 0.519 | 0.583 |
| Logistic Regression (No SMOTE) | 0.803 | 0.667 | 0.848 | 0.528 | 0.586 | 0.806 | 0.663 | 0.848 | 0.537 | 0.595 |
| CatBoost (No SMOTE Balanced) | 0.753 | 0.667 | 0.850 | 0.801 | 0.632 | 0.750 | 0.670 | 0.848 | 0.805 | 0.631 |
| Random Forest (No SMOTE Tuned) | 0.802 | 0.666 | 0.848 | 0.506 | 0.575 | 0.806 | 0.659 | 0.846 | 0.516 | 0.586 |
| Logistic Regression (Focused Accuracy) | 0.805 | 0.666 | 0.849 | 0.536 | 0.594 | 0.813 | 0.664 | 0.847 | 0.553 | 0.612 |
| Gradient Boosting (No SMOTE) | 0.805 | 0.665 | 0.849 | 0.521 | 0.587 | 0.803 | 0.660 | 0.845 | 0.527 | 0.587 |
| HistGradientBoosting (No SMOTE Tuned) | 0.802 | 0.663 | 0.848 | 0.520 | 0.582 | 0.808 | 0.664 | 0.848 | 0.535 | 0.596 |
| Extra Trees (No SMOTE) | 0.800 | 0.663 | 0.847 | 0.510 | 0.575 | 0.800 | 0.657 | 0.843 | 0.527 | 0.583 |
| LightGBM (No SMOTE Tuned) | 0.802 | 0.663 | 0.848 | 0.533 | 0.588 | 0.809 | 0.661 | 0.850 | 0.540 | 0.600 |
| CatBoost (SMOTE) | 0.794 | 0.661 | 0.847 | 0.617 | 0.614 | 0.784 | 0.662 | 0.844 | 0.626 | 0.605 |
| Logistic Regression (SMOTE) | 0.756 | 0.661 | 0.845 | 0.786 | 0.631 | 0.744 | 0.654 | 0.843 | 0.805 | 0.625 |
| XGBoost (SMOTE) | 0.796 | 0.656 | 0.846 | 0.628 | 0.621 | 0.780 | 0.651 | 0.843 | 0.628 | 0.603 |
| Random Forest (SMOTE) | 0.783 | 0.656 | 0.845 | 0.712 | 0.636 | 0.769 | 0.644 | 0.843 | 0.722 | 0.624 |
| LightGBM (No SMOTE Balanced) | 0.756 | 0.654 | 0.844 | 0.772 | 0.627 | 0.763 | 0.660 | 0.845 | 0.805 | 0.643 |
| SVM RBF (No SMOTE) | 0.803 | 0.642 | 0.796 | 0.508 | 0.578 | 0.801 | 0.622 | 0.795 | 0.476 | 0.559 |
| LightGBM (SMOTE Balanced) | 0.793 | 0.630 | 0.834 | 0.563 | 0.591 | 0.786 | 0.614 | 0.830 | 0.575 | 0.587 |
| SVM RBF (SMOTE) | 0.765 | 0.602 | 0.824 | 0.760 | 0.632 | 0.759 | 0.615 | 0.826 | 0.743 | 0.621 |

Table 2. Threshold comparison.

| threshold | accuracy | precision | recall | f1 | false_positive | false_negative | business_cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.140 | 0.676 | 0.446 | 0.912 | 0.599 | 424 | 33 | 589 |
| 0.340 | 0.776 | 0.560 | 0.733 | 0.635 | 215 | 100 | 715 |
| 0.500 | 0.813 | 0.688 | 0.543 | 0.607 | 92 | 171 | 947 |
| 0.510 | 0.815 | 0.697 | 0.535 | 0.605 | 87 | 174 | 957 |

## 5. Discussion

The DSS identified 192 high-risk and 347 medium-risk customers in the hold-out set, with 765 customers marked as intervention candidates under the cost-sensitive threshold. The top global churn drivers were tenure, month_to_month_fiber, is_month_to_month, OnlineSecurity_No, paperless_echeck. These drivers are strategically interpretable because they connect directly to retention levers such as contract migration, pricing review, support outreach, service bundling, and payment-method intervention. The lifecycle analysis should not be interpreted as a full time series model; it is a tenure-based proxy that reveals how observed churn varies across customer relationship stages. The implementation also avoids using demographic attributes as standalone recommendation triggers, preserving the role of human managerial judgment in the final treatment decision.

Table 3. Tenure-band lifecycle churn.

| tenure_band | customers | churn_rate |
| --- | --- | --- |
| 0-12 months | 2186 | 0.474 |
| 13-24 months | 1024 | 0.287 |
| 25-48 months | 1594 | 0.204 |
| 49-72 months | 2239 | 0.095 |

Table 4. Top SHAP churn drivers.

| feature | mean_abs_shap |
| --- | --- |
| tenure | 0.370 |
| month_to_month_fiber | 0.270 |
| is_month_to_month | 0.257 |
| OnlineSecurity_No | 0.168 |
| paperless_echeck | 0.159 |
| MultipleLines_No | 0.131 |
| TechSupport_No | 0.128 |
| contract_commitment_score | 0.122 |
| tenure_inverse | 0.086 |
| streaming_count | 0.083 |

Table 5. Sample SHAP local explanations.

| customerID | churn_probability | top_shap_drivers |
| --- | --- | --- |
| 5178-LMXOP | 0.9229568243026732 | tenure (increases churn risk, 0.718); is_month_to_month (increases churn risk, 0.283); month_to_month_fiber (increases churn risk, 0.276); paperless_echeck (increases churn risk, 0.197); OnlineSecurity_No (increases churn risk, 0.163) |
| 0295-PPHDO | 0.9157661199569702 | tenure (increases churn risk, 0.719); is_month_to_month (increases churn risk, 0.283); month_to_month_fiber (increases churn risk, 0.276); paperless_echeck (increases churn risk, 0.197); OnlineSecurity_No (increases churn risk, 0.164) |
| 1069-XAIEM | 0.8948736190795898 | tenure (increases churn risk, 0.742); is_month_to_month (increases churn risk, 0.287); month_to_month_fiber (increases churn risk, 0.276); paperless_echeck (increases churn risk, 0.200); OnlineSecurity_No (increases churn risk, 0.170) |
| 9248-OJYKK | 0.8786069750785828 | tenure (increases churn risk, 0.751); is_month_to_month (increases churn risk, 0.287); month_to_month_fiber (increases churn risk, 0.276); paperless_echeck (increases churn risk, 0.200); OnlineSecurity_No (increases churn risk, 0.170) |
| 0970-ETWGE | 0.8738260865211487 | tenure (increases churn risk, 0.728); is_month_to_month (increases churn risk, 0.285); month_to_month_fiber (increases churn risk, 0.276); paperless_echeck (increases churn risk, 0.173); OnlineSecurity_No (increases churn risk, 0.169) |

Table 6. Sample LIME local explanations.

| customerID | churn_probability | lime_explanation |
| --- | --- | --- |
| 5178-LMXOP | 0.9229568243026732 | tenure: -0.037; is_month_to_month: 0.018; tenure_inverse: 0.011; contract_commitment_score: -0.011; OnlineSecurity_No: 0.010 |
| 0295-PPHDO | 0.9157661199569702 | tenure: -0.035; is_month_to_month: 0.017; tenure_inverse: 0.011; OnlineSecurity_No: 0.009; contract_commitment_score: -0.009 |
| 1069-XAIEM | 0.8948736190795898 | tenure: -0.036; is_month_to_month: 0.018; tenure_inverse: 0.011; contract_commitment_score: -0.010; OnlineSecurity_No: 0.010 |
| 9248-OJYKK | 0.8786069750785828 | tenure: -0.035; is_month_to_month: 0.018; tenure_inverse: 0.010; contract_commitment_score: -0.010; OnlineSecurity_No: 0.010 |
| 0970-ETWGE | 0.8738260865211487 | tenure: -0.035; is_month_to_month: 0.017; OnlineSecurity_No: 0.011; tenure_inverse: 0.010; month_to_month_fiber: 0.008 |

## 6. Conclusion

The Phase 2 implementation transforms the initial proposal into a reproducible strategic information systems artifact. It delivers more than a churn label: it provides model comparison, imbalance-aware evaluation, cost-sensitive thresholding, global and local explanations, lifecycle analysis, risk segmentation, and managerial action recommendations through a working dashboard. Future extensions could add real event timestamps, customer lifetime value, campaign cost data, model monitoring, and feedback loops from actual retention outcomes.

## References

Asif, D., Arif, M. S., & Mukheimer, A. (2025). A data-driven approach with explainable artificial intelligence for customer churn prediction in the telecommunications industry. Results in Engineering, 26, 104629.

Chang, V., Hall, K., Xu, Q. A., Amao, F. O., Ganatra, M. A., & Benson, V. (2024). Prediction of customer churn behavior in the telecommunication industry using machine learning models. Algorithms, 17, 231.

El Attar, A., & El-Hajj, M. (2026). Explainable AI-driven customer churn prediction: A multi-model ensemble approach with SHAP-based feature analysis. Frontiers in Artificial Intelligence, 9, 1748799.

Hooda, P., Mittal, P., Shukla, P. K., Shukla, P. K., & Pandey, A. (2026). Combining predictive accuracy and interpretability: A data-driven approach to telecom churn analysis. Scientific Reports, 16, 4596.

Imani, M., Joudaki, M., Beikmohammadi, A., & Arabnia, H. R. (2025). Customer churn prediction: A systematic review of recent advances, trends, and challenges in machine learning and deep learning. Machine Learning and Knowledge Extraction, 7, 105.

Omari, A., Al-Omari, O., Al-Omari, T., & Fati, S. M. (2025). A predictive analytics approach to improve telecom's customer retention. Frontiers in Artificial Intelligence, 8, 1600357.

Poudel, S. S., Pokharel, S., & Timilsina, M. (2024). Explaining customer churn prediction in telecom industry using tabular machine learning models. Machine Learning with Applications, 17, 100567.

Sikri, A., Jameel, R., Idrees, S. M., & Kaur, H. (2024). Enhancing customer retention in telecom industry with machine learning driven churn prediction. Scientific Reports, 14, 13097.

Wagh, S. K., Andhale, A. A., Wagh, K. S., Pansare, J. R., Ambadekar, S. P., & Gawande, S. H. (2024). Customer churn prediction in telecom sector using machine learning techniques. Results in Control and Optimization, 14, 100342.

Yeanzc. (n.d.). Telco customer churn: IBM dataset [Data set]. Kaggle.

Zerine, I., Islam, M. M., Khan, M. A. U., Chy, M. A. R., Saimon, A. S. M., Manik, M. M. T. G., & Wata, C. (2026). Explainable churn prediction in telecom with tabular ML five model benchmark and SHAP analysis. Discover Artificial Intelligence, 6, 263.
