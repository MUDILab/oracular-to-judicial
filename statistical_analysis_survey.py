"""
statistical_analysis_survey.py

This script reads anonymized questionnaire data and calibration data,
merges and preprocesses them, then performs various statistical tests
(Wilcoxon, McNemar, bootstrap-based effect sizes) to analyze the impact
of AI/XAI in clinical settings.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.utils import resample
from scipy.stats import spearmanr, kendalltau

# ------------------------------------------------------------------------
# 1. Data Loading and Basic Preprocessing
# ------------------------------------------------------------------------

def create_profession_columns(df, id_mapping, text_mapping):
    """
    Adds two columns to df: 'type_profession' and 'expertise' based on a mapping.
    """
    df["type_profession"] = None
    df["expertise"] = None
    
    for index, row in df.iterrows():
        pid = row["id"]
        # Find the key in id_mapping whose value matches pid
        number = list(id_mapping.keys())[list(id_mapping.values()).index(pid)]
        
        # Get profession and expertise from text_mapping
        profession_info = text_mapping[number].split()
        
        df.at[index, "type_profession"] = profession_info[0]
        df.at[index, "expertise"]       = profession_info[1]
    
    return df

def create_expertise_dico(x):
    """
    Bins the expertise level into a binary variable:
    Returns 1 if expertise is '5-10' years, else 0.
    """
    return 1 if x == "5-10" else 0

def merge_multiple_dataframes(dataframes, merge_on):
    """
    Sequentially merges multiple dataframes in a dictionary on the specified keys.
    Returns a single merged dataframe.
    """
    # Start with the first dataframe as the base
    keys = list(dataframes.keys())
    merged_df = dataframes[keys[0]].rename({"Response": keys[0]}, axis=1).drop("Question", axis=1)
    
    # Merge each subsequent dataframe
    for k in keys[1:]:
        temp = dataframes[k].drop("Question", axis=1).rename({"Response": k}, axis=1)
        merged_df = pd.merge(
            merged_df,
            temp,
            on=merge_on,
            suffixes=("", "_right")
        )
        # Handle duplicates from right merges
        for col in temp.columns:
            if col not in merge_on and col in merged_df.columns and col + "_right" in merged_df.columns:
                if merged_df[col].equals(merged_df[col + "_right"]):
                    merged_df.drop(col + "_right", axis=1, inplace=True)
                else:
                    merged_df.rename(columns={col + "_right": col + "_from_right"}, inplace=True)
    
    return merged_df


def glass_delta(control, tested):
    """
    Computes Glass's Delta comparing a tested group to a control group:
    (mean(tested) - mean(control)) / std(control).
    """
    mean_control = np.mean(control)
    mean_tested  = np.mean(tested)
    std_dev_ctrl = np.std(control, ddof=1)
    
    return (mean_tested - mean_control) / std_dev_ctrl


def bootstrap_glass_delta_ci(
    control, tested, 
    name_save="glass_delta", 
    n_bootstraps=1000, 
    n_samples=100, 
    alpha=0.05, 
    margin_error=0.2
):
    """
    Bootstraps Glass's Delta to get a BCa confidence interval and checks non-inferiority
    relative to a specified margin of error.
    """
    from scipy.stats import norm
    
    # Original statistic
    initial_delta = glass_delta(control, tested)
    boot_deltas   = []
    
    # Create bootstrap replicates
    for _ in range(n_bootstraps):
        sample_control = resample(control, replace=True, n_samples=n_samples)
        sample_tested  = resample(tested,  replace=True, n_samples=n_samples)
        boot_deltas.append(glass_delta(sample_control, sample_tested))
    
    boot_deltas = np.array(boot_deltas)
    z0          = norm.ppf(np.sum(boot_deltas < initial_delta) / n_bootstraps)
    z_alpha     = norm.ppf(alpha / 2)
    
    lower_bca = np.percentile(boot_deltas, 100*norm.cdf(2*z0 + z_alpha))
    upper_bca = np.percentile(boot_deltas, 100*norm.cdf(2*z0 - z_alpha))
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.hist(boot_deltas, bins=50, color="skyblue", alpha=0.7, label="Bootstrapped Glass's Delta")
    plt.axvline(initial_delta, color="red", linestyle="--", label="Initial Glass's Delta")
    plt.axvline(lower_bca,    color="green", linestyle="--", label="95% CI Lower")
    plt.axvline(upper_bca,    color="green", linestyle="--", label="95% CI Upper")
    plt.axvline(margin_error,  color="black", linestyle="-", label="± Margin")
    plt.axvline(-margin_error, color="black", linestyle="-")
    plt.legend()
    plt.title(f"Bootstrap CI for Glass's Delta (Non-Inferiority Margin={margin_error})")
    plt.xlabel("Glass's Delta")
    plt.ylabel("Frequency")
    plt.savefig(f"non_inferiority_glass_delta_{name_save}.png")
    plt.close()
    
    # Non-inferiority check
    if lower_bca > -margin_error:
        print(f"[{name_save}] The data suggest non-inferiority.")
    else:
        print(f"[{name_save}] The data do NOT conclusively suggest non-inferiority.")
    
    print(f"Initial Glass's Delta: {initial_delta}")
    print(f"BCa 95% CI: ({lower_bca}, {upper_bca})")
    print(f"Margin of Error: {margin_error}")


def main():
    # -------------------------------
    # 1. Load Data
    # -------------------------------
    df = pd.read_csv("../data/result_survey_preprocessed_final.csv")
    calibration = pd.read_csv("../data/calibration_information.csv", sep=";")
    
    # Example ID mappings:
    id_mapping = {
        1: 44,  2: 48,  3: 49,  4: 50,  5: 51,  6: 52,  7: 53,  8: 54,
        9: 55, 10: 56, 11: 57, 12: 59, 13: 62, 14: 63, 15: 65, 16: 68,
    }
    
    text_mapping = {
        1:  "Orto 5-10",
        2:  "Orto 5-10",
        3:  "Orto 0-5",
        4:  "Orto 5-10",
        5:  "Orto 0-5",
        6:  "Orto 0-5",
        7:  "Orto 0-5",
        8:  "Radiologo 0-5",
        9:  "Radiologo 0-5",
        10: "Radiologo 0-5",
        11: "Orto 0-5",
        12: "Radio 5-10",
        13: "Radio 5-10",
        14: "Radio 5-10",
        15: "Orto 5-10",
        16: "Orto 5-10",
    }
    
    # -------------------------------
    # 2. Preprocess Data
    # -------------------------------
    df = create_profession_columns(df, id_mapping, text_mapping)
    df["expertise_dico"] = df["expertise"].apply(create_expertise_dico)
    
    # Replace commas in calibration
    calibration.replace(",", ".", regex=True, inplace=True)
    
    # Merge data
    merged_df = calibration.merge(df, on="id_Caso")
    
    # Prepare dictionary for multiple merges:
    data_dict = {
        "initial_confidence":   merged_df[merged_df.Question == "initial_confidence"],
        "final_confidence":     merged_df[merged_df.Question == "final_confidence"],
        "utility":              merged_df[merged_df.Question == "utility"],
        "ground_truth":         merged_df[merged_df.Question == "Ground_truth"],
        "reliance_pattern":     merged_df[merged_df.Question == "reliance_pattern"],
        "correct_human_first_decision": merged_df[merged_df.Question == "correct_human_first_decision"],
        "correct_human_final_decision": merged_df[merged_df.Question == "correct_human_final_decision"],
    }
    
    # Merge them into a single wide CSV:
    merged_complete = merge_multiple_dataframes(data_dict, ["id_Caso", "id"])
    # Some data might have "Unnamed: 7" if it existed in calibration; drop safely if present
    if "Unnamed: 7" in merged_complete.columns:
        merged_complete.drop(["Unnamed: 7"], axis=1, inplace=True)
    merged_complete.to_csv("colored_shadow_2_complete.csv", index=False)
    
    print("Merged dataset saved to colored_shadow_2_complete.csv.")
    
    # -------------------------------
    # 3. Example Subgroup Analyses
    # -------------------------------
    # We still have our original df for question-based filtering
    df_complex = df[df.Question == "complexity_perceived"]
    
    # Compute mean complexity by case, classify into 'less complex' (<3) or 'more complex' (>=3)
    mean_complexity = df_complex.groupby("id_Caso")["Response"].mean()
    less_complex_id_case = mean_complexity[mean_complexity < 3].index
    more_complex_id_case = mean_complexity[mean_complexity >= 3].index
    
    # Identify less-expert vs. more-expert participants
    less_expert_id = df[(df["expertise_dico"] == 0)]["id"].unique()
    more_expert_id = df[(df["expertise_dico"] == 1)]["id"].unique()
    
    # Extract the data for correctness: first vs. final
    human_first = df[df.Question == "correct_human_first_decision"].copy()
    human_final = df[df.Question == "correct_human_final_decision"].copy()
    
    # Example: filter for 'more_complex' cases and 'more_expert' participants
    hf_more_complex = human_first[
        human_first["id_Caso"].isin(more_complex_id_case) &
        human_first["id"].isin(more_expert_id)
    ]
    hf_final_more_complex = human_final[
        human_final["id_Caso"].isin(more_complex_id_case) &
        human_final["id"].isin(more_expert_id)
    ]
    
    # Merge to compare correct decision pre/post
    comp_merged = hf_more_complex.merge(
        hf_final_more_complex, 
        on=["id","id_Caso"], 
        suffixes=("_first","_final")
    )
    
    # McNemar's test
    a = sum(
        (comp_merged["Response_first"] == 1) & (comp_merged["Response_final"] == 1)
    )
    b = sum(
        (comp_merged["Response_first"] == 1) & (comp_merged["Response_final"] == 0)
    )
    c = sum(
        (comp_merged["Response_first"] == 0) & (comp_merged["Response_final"] == 1)
    )
    d = sum(
        (comp_merged["Response_first"] == 0) & (comp_merged["Response_final"] == 0)
    )
    
    contingency = [[a, b], [c, d]]
    result_mcnemar = mcnemar(contingency, exact=False)
    print("McNemar test (More Expert, More Complex):", result_mcnemar)
    
    # -------------------------------
    # 4. Effect Size (Glass's Delta) Example
    # -------------------------------
    # Suppose we compare the accuracy across participants pre vs. post
    # Summarize correctness by participant for 'first' vs. 'final'
    init_accuracy = hf_more_complex.groupby("id")["Response"].mean().values
    final_accuracy = hf_final_more_complex.groupby("id")["Response"].mean().values
    
    # Bootstrap to examine Glass's Delta
    bootstrap_glass_delta_ci(
        control=init_accuracy, 
        tested=final_accuracy,
        name_save="more_expert_more_complex",
        n_bootstraps=2000,
        alpha=0.05,
        margin_error=0.2
    )
    
    # -------------------------------
    # 5. Confidence Analysis Example
    # -------------------------------
    # Compare initial vs. final confidence for less complex cases, or any subgroup
    conf_init = df[(df["Question"] == "initial_confidence") &
                   (df["id_Caso"].isin(less_complex_id_case))]["Response"]
    conf_final = df[(df["Question"] == "final_confidence") &
                    (df["id_Caso"].isin(less_complex_id_case))]["Response"]
    
    stat, p_value = wilcoxon(conf_init, conf_final)
    print(f"Wilcoxon test on initial vs. final confidence (less complex group): statistic={stat}, p={p_value}")
    
    # Plot distribution
    plt.figure()
    plt.hist(conf_init, bins=10, alpha=0.5, label="Initial Confidence")
    plt.hist(conf_final, bins=10, alpha=0.5, label="Final Confidence")
    plt.legend()
    plt.title("Confidence Distribution (Less Complex Cases)")
    plt.xlabel("Confidence Rating")
    plt.ylabel("Frequency")
    plt.savefig("confidence_less_complex.png")
    plt.close()
    
    print("Analysis complete.")

if __name__ == "__main__":
    main()
