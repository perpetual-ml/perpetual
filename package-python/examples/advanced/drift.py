import json
import tempfile

import numpy as np
import pandas as pd
from logger import get_logger
from scipy.stats import chi2_contingency
from snowflake.ml.registry import Registry
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session
from snowflake.snowpark.functions import col

# TODO: try using leaf weights in model drift calculation


def predict_drift(
    function_name: str,
    model_name: str,
    model_version: str | None,
    source_ref: str,
    source_alias: str,
    job_name: str,
    timestamp_column: str,
    aggregation_window: str,
    warehouse: str | None,
) -> None:
    """
    Perform prediction drift monitoring using chi-squared statistical testing.

    Args:
        function_name: The function name for drift calculation mode
        model_name: Name of the model in the registry
        model_version: Specific model version or None for latest
        source_ref: Reference to the source data
        source_alias: Alias for the source data
        job_name: Name of the monitoring job (used for table naming)
        timestamp_column: Column name containing timestamps
        aggregation_window: Pandas frequency string for grouping (e.g., '1D', '1H')
        warehouse: Snowflake warehouse to use, or None for default
    """
    logger = get_logger(job_name)
    logger.info("Job started...")
    logger.info(f"Parameter: function_name={function_name}")
    logger.info(f"Parameter: model_name={model_name}")
    logger.info(f"Parameter: model_version={model_version}")
    logger.info(f"Parameter: source_ref={source_ref}")
    logger.info(f"Parameter: source_alias={source_alias}")
    logger.info(f"Parameter: job_name={job_name}")
    logger.info(f"Parameter: timestamp_column={timestamp_column}")
    logger.info(f"Parameter: aggregation_window={aggregation_window}")

    session = Session.builder.configs(SnowflakeLoginOptions()).create()
    try:
        if warehouse:
            session.use_warehouse(warehouse)

        reg = Registry(session)
        model = reg.get_model(model_name)
        if model_version:
            mv = model.version(model_version)
        else:
            mv = model.last()
            model_version = mv.version_name
        loaded_model = mv.load()

        features = [f.name for f in mv.show_functions()[0]["signature"].inputs]
        logger.info(f"features: {features}")

        logger.info(f"mv.functions: {mv.show_functions()}")

        qualified_name = mv.fully_qualified_model_name
        logger.info(f"mv.fully_qualified_model_name: {qualified_name}")

        model_db, model_schema, *_ = qualified_name.split(".")

        monitoring_table = job_name
        monitoring_table_name = f"{model_db}.{model_schema}.{monitoring_table}"

        # Note: Table/column names are constructed from trusted job configuration.
        # Ensure job_name and timestamp_column are validated at the API layer.
        sql_create_monitoring_table = f"""
            CREATE TABLE IF NOT EXISTS {monitoring_table_name} (
                {timestamp_column} TIMESTAMP_NTZ,
                combined_chi2 FLOAT,
                model_version VARCHAR(100)
            );
        """
        session.sql(sql_create_monitoring_table).collect()

        features_and_date = features + [timestamp_column]
        features_and_date_joined = ", ".join(features_and_date)

        latest_monitored_row_date = pd.to_datetime(
            session.sql(f"SELECT MAX({timestamp_column}) FROM {monitoring_table_name};")
            .to_pandas()
            .to_numpy()[0][0]
        )
        logger.info(f"latest_monitored_row_date: {latest_monitored_row_date}")

        sql_df = f"SELECT {features_and_date_joined} FROM reference('{source_ref}', '{source_alias}')"
        if pd.notnull(latest_monitored_row_date):
            # Format timestamp properly for SQL
            formatted_date = latest_monitored_row_date.strftime("'%Y-%m-%d %H:%M:%S'")
            sql_df = sql_df + f" WHERE {timestamp_column} > {formatted_date}"
        logger.info(f"sql_df: {sql_df}")
        df = session.sql(sql_df).to_pandas()

        # Early exit if no new data to process
        if df.empty:
            logger.info("No new data to process. Exiting.")
            return

        df = df.sort_values(by=timestamp_column)
        logger.info(f"df.len: {len(df)}")

        node_lists = loaded_model.get_node_lists()
        logger.info(f"model_calculated_number_of_trees: {len(node_lists)}")
        logger.info(
            f"model_calculated_number_of_leaves: {int(np.sum([(len(nl) + 1) / 2 for nl in node_lists]))}"
        )
        model_node_counts = {}
        model_node_parents = {}
        model_parent_to_nodes = {}
        model_node_is_leaf = {}
        for tree_number, nl in enumerate(node_lists):
            for n in nl:
                if n.is_leaf:
                    model_node_is_leaf[f"{tree_number}-{n.num}"] = True
                if n.node_type != "Root":
                    model_node_counts[f"{tree_number}-{n.num}"] = n.count
                    model_node_parents[f"{tree_number}-{n.num}"] = n.parent_node
                    ptl = model_parent_to_nodes.get(f"{tree_number}-{n.parent_node}")
                    if ptl:
                        if n.node_type == "Left":
                            ptl[0] = f"{tree_number}-{n.num}"
                        elif n.node_type == "Right":
                            ptl[1] = f"{tree_number}-{n.num}"
                    else:
                        if n.node_type == "Left":
                            ptl = [f"{tree_number}-{n.num}", None]
                        elif n.node_type == "Right":
                            ptl = [None, f"{tree_number}-{n.num}"]
                    model_parent_to_nodes[f"{tree_number}-{n.parent_node}"] = ptl

        logger.info(f"model_node_counts.len: {len(model_node_counts)}")
        logger.info(f"model_node_parents.len: {len(model_node_parents)}")
        logger.info(f"model_parent_to_nodes.len: {len(model_parent_to_nodes)}")

        json_path = f"{model_version}_pred_node_counts.json"
        session.sql(f"CREATE STAGE IF NOT EXISTS {monitoring_table_name}").collect()
        full_stage_path = f"@{monitoring_table_name}/{json_path}"
        result = session.read.json(full_stage_path).select(col("$1")).collect()
        if result and result[0][0]:
            pred_node_counts = json.loads(result[0][0])
        else:
            logger.warning(f"File not found or empty: {full_stage_path}")
            pred_node_counts = dict()

        grouper = pd.Grouper(
            key=timestamp_column, freq=aggregation_window, origin="epoch"
        )
        groups = df.groupby(grouper)
        drift_ts_list = []
        combined_chi2_list = []
        for name, gr in groups:
            logger.info(f"Monitoring group name: {name}")
            gr = gr.drop(columns=[timestamp_column])
            drift_ts_list.append(name)
            pred_nodes = loaded_model.predict_nodes(gr)
            nodes = []
            for tree_number, p_nodes in enumerate(pred_nodes):
                dn = [f"{tree_number}-{max(p)}" for p in p_nodes]
                nodes.append(dn)
            nodes = np.array(nodes).flatten()
            unique, counts = np.unique(nodes, return_counts=True)
            gr_pred_node_counts = dict(zip(unique.tolist(), counts.tolist()))

            for uniq, coun in gr_pred_node_counts.items():
                c = pred_node_counts.get(uniq, 0)
                pred_node_counts[uniq] = c + coun

            group_chi2_list = []
            for parent, (left_node, right_node) in model_parent_to_nodes.items():
                if (
                    model_node_is_leaf.get(left_node)
                    or model_node_is_leaf.get(right_node)
                    or function_name == "predict_nodes_data"  # multivariate data drift
                ):
                    left_pred_count = pred_node_counts.get(left_node, 0)
                    right_pred_count = pred_node_counts.get(right_node, 0)
                    if left_pred_count > 0 and right_pred_count > 0:
                        first_row = [model_node_counts[left_node], left_pred_count]
                        second_row = [model_node_counts[right_node], right_pred_count]
                        res = chi2_contingency([first_row, second_row])
                        group_chi2_list.append(res.statistic)

            # Handle empty chi2 list to avoid NaN
            combined_chi2 = np.mean(group_chi2_list) if group_chi2_list else 0.0
            combined_chi2_list.append(combined_chi2)

        # Save pred_node_counts to stage using tempfile for cross-platform compatibility
        json_data = json.dumps(pred_node_counts)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(json_data)
            temp_file_path = f.name
        session.sql(
            f"PUT file://{temp_file_path} @{monitoring_table_name}/{json_path} OVERWRITE=TRUE"
        ).collect()
        logger.info(
            f"Dictionary uploaded to stage: @{monitoring_table_name}/{json_path}"
        )

        model_version_list = [model_version] * len(combined_chi2_list)

        date_list = pd.Series(pd.to_datetime(drift_ts_list, utc=True))

        logger.info(f"drift_ts_list: {date_list}")
        logger.info(f"combined_chi2_list: {combined_chi2_list}")
        logger.info(f"model_version_list: {model_version_list}")

        drift_predictions = pd.DataFrame(
            {
                timestamp_column.upper(): date_list,
                "COMBINED_CHI2": combined_chi2_list,
                "MODEL_VERSION": model_version_list,
            }
        )

        # Use fully qualified table name for DELETE
        sql_delete_last_row = f"""DELETE FROM {monitoring_table_name}
                WHERE {timestamp_column} = (
                SELECT MAX({timestamp_column})
                FROM {monitoring_table_name}
            );
            """
        session.sql(sql_delete_last_row).collect()

        session.write_pandas(
            drift_predictions,
            monitoring_table,
            database=model_db,
            schema=model_schema,
            use_logical_type=True,
            quote_identifiers=False,
        )

        logger.info("Finished prediction!")

    finally:
        session.close()
