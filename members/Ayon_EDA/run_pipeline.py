from dataset_analysis import run_dataset_analysis
from class_distribution import save_class_distribution
from preprocessing import generate_degradation_dataset


def main():
    # 1. Load and analyse datasets
    results = run_dataset_analysis(
        output_dir="/content/drive/MyDrive/Cityscapes_EDA/results"
    )

    # 2. Analyse semantic class distribution
    mask_df, class_df = save_class_distribution(
        results["gt_zip"],
        results["acdc_sample"],
        results["output_dirs"]["csv"],
        results["output_dirs"]["graphs"]
    )

    # 3. Generate controlled degradation samples
    degradation_df, severity_summary, family_summary = generate_degradation_dataset(
        results["rgb_zip"],
        results["acdc_sample"],
        output_dir="/content/drive/MyDrive/Cityscapes_EDA/results/degradation_experiments",
        max_images=10
    )

    return {
        "results": results,
        "mask_df": mask_df,
        "class_df": class_df,
        "degradation_df": degradation_df,
        "severity_summary": severity_summary,
        "family_summary": family_summary,
    }


if __name__ == "__main__":
    main()
