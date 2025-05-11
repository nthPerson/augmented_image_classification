import json
import pandas as pd


def make_summary(input_json="summary.json", output_csv="summary.csv"):
    """
    Reads summary.json and writes summary.csv with columns:
    model, top1, top3, avg_inference_time
    """
    with open(input_json) as f:
        summary = json.load(f)

    rows = []
    for model, vals in summary.items():
        rows.append({
            "model": model,
            "top1": vals["top1"],
            "top3": vals.get("top3", vals.get("top5")),  # some JSONs might call it top5
            "avg_inference_time": vals["avg_inference_time"]
        })
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"→ Wrote summary to {output_csv}")


def make_per_class(
    json2d="model2d_metrics.json",
    json3d="model3d_metrics.json",
    output_csv="per_class_metrics.csv"
):
    """
    Reads the precision/recall/f1 lists from each JSON,
    turns them into rows of (model, class, precision, recall, f1),
    and concatenates into one CSV.
    """
    def load_json(json_file, model_name):
        j = json.load(open(json_file))
        n = len(j["precision"])
        df = pd.DataFrame({
            "class": list(range(n)),
            "precision": j["precision"],
            "recall":    j["recall"],
            "f1":        j["f1"]
        })
        df["model"] = model_name
        return df

    df2 = load_json(json2d, "model2d")
    df3 = load_json(json3d, "model3d")
    pd.concat([df2, df3], ignore_index=True)\
      .to_csv(output_csv, index=False)
    print(f"→ Wrote per-class metrics to {output_csv}")


def unpivot_confusion(
    input_csv, model_name, output_csv
):
    """
    Reads a 10×10 confusion matrix (no row index),
    pivots it to rows of (model, actual, predicted, count).
    """
    df = pd.read_csv(input_csv, header=0)
    df["actual"] = df.index
    long = df.melt(
        id_vars=["actual"],
        var_name="predicted",
        value_name="count"
    )
    long["model"] = model_name
    # convert predicted to int if it's string
    long["predicted"] = long["predicted"].astype(int)
    long.to_csv(output_csv, index=False)
    print(f"→ Wrote unpivoted confusion to {output_csv}")


def make_inference_times(
    csv2d="model2d_inference_times.csv",
    csv3d="model3d_inference_times.csv",
    output_csv="inference_times2d.csv"
):
    """
    Reads two single-column CSVs of inference times,
    tags them with model, and stacks into one file.
    """
    def load_times(fpath, model_name):
        df = pd.read_csv(fpath, header=None, names=["inference_time"])
        df["model"] = model_name
        return df

    t2 = load_times(csv2d, "model2d")
    t3 = load_times(csv3d, "model3d")
    pd.concat([t2, t3], ignore_index=True)\
      .to_csv(output_csv, index=False)
    print(f"→ Wrote all inference times to {output_csv}")


if __name__ == "__main__":
    make_summary("summary.json", "summary.csv")
    make_per_class(
        json2d="model2d_metrics.json",
        json3d="model3d_metrics.json",
        output_csv="per_class_metrics.csv"
    )
    unpivot_confusion("model2d_confusion.csv", "model2d", "confusion2d_flat.csv")
    unpivot_confusion("model3d_confusion.csv", "model3d", "confusion3d_flat.csv")
    make_inference_times(
        csv2d="model2d_inference_times.csv",
        csv3d="model3d_inference_times.csv",
        output_csv="inference_times2d.csv"
    )
    print("All flat files generated.")