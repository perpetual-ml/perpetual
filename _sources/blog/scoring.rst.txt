Profit-Driven Lead Scoring
==========================

*Published: October 31, 2025 — Maximize marketing ROI with calibrated probabilities and profit-driven decision thresholds.*

Focus your sales efforts where they count. By pairing ``PerpetualBooster``'s calibrated probability predictions with **Optimum Business Decisioning**, marketing and sales teams can move beyond simple accuracy metrics to **directly optimize business profit** from marketing campaigns, demonstrated here through a lead scoring application.

Optimum Business Decisioning: Optimizing Profit, Not Just Accuracy
------------------------------------------------------------------

Traditional machine learning models prioritize statistical measures like accuracy, log-loss, or F1-score. However, in real-world marketing and sales pipelines, false positives and false negatives carry vastly different financial consequences. Incorporating specific **financial inputs** directly into the model's threshold selection function ensures that every lead outreach decision maximizes net expected profit.

Financial Payoff Parameters
---------------------------

Using the following per-lead financial parameters for our lead scoring evaluation:

.. list-table:: Lead Scoring Payoff Matrix
   :header-rows: 1
   :widths: 28 16 56

   * - Outcome
     - Input Value
     - Description
   * - **True Positive (TP)**
     - **+$1,000**
     - Financial gain from a successful conversion (Good Lead → Customer).
   * - **False Positive (FP)**
     - **-$100**
     - Financial cost of sales time wasted on a dead lead (Bad Lead → Sales Effort).
   * - **False Negative (FN)**
     - **-$500**
     - Opportunity cost of a missed good lead (Good Lead → No Sales Effort).

Verified Performance: The Financial Edge
----------------------------------------

Using the industry-standard **Bank Marketing dataset** from the `UCI Machine Learning Repository <https://archive.ics.uci.edu/dataset/222/bank+marketing>`_, we compare a calibrated ``LightGBM`` pipeline against ``PerpetualBooster`` using the exact same business objective:

* **LightGBM + CalibratedClassifierCV**: A tuned baseline using ``LightGBM`` combined with scikit-learn's ``CalibratedClassifierCV`` achieved a total campaign profit of **$733,200**.
* **PerpetualBooster + Optimum Business Decisioning**: Leveraging ``PerpetualBooster`` with profit-driven threshold optimization delivered a total campaign profit of **$740,600**.

.. list-table:: Bank Marketing Campaign Profit Comparison
   :header-rows: 1
   :widths: 50 30 20

   * - Pipeline
     - Total Campaign Profit
     - Relative Gain
   * - LightGBM + CalibratedClassifierCV
     - $733,200
     - Baseline
   * - **PerpetualBooster + Decisioning**
     - **$740,600**
     - **+1.0% (+$7,400)**

This validates that ``PerpetualBooster`` delivers **1.0% higher net profit** in a marketing campaign compared to a tuned ``LightGBM + CalibratedClassifierCV`` pipeline, even when both aim for the same maximum financial outcome. This increase in profitability is the direct result of superior probability calibration and integrating specific cost and benefit variables into the decision process.

