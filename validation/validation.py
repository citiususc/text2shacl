from pathlib import Path
from pyshacl import validate
from rdflib import Graph, RDF, SH
import os
import pickle
import argparse

def evaluate_shacl_quality(data_file_nq, auto_shacl_file, gold_shacl_file):
    VALIDATION_DIR = Path(__file__).resolve().parent # root/validation

    gold_path = VALIDATION_DIR/'gold_standard_validation.pkl'

    print("Loading RDF data...")
    data_graph = Graph()
    data_graph.parse(data_file_nq, format='nquads')
    auto_graph = Graph().parse(auto_shacl_file, format='ttl')

    if os.path.exists(gold_path):
        print("Loading gold standard validation results from file...")
        with open(gold_path, 'rb') as f:
            results_graph_gold = pickle.load(f)
    else:
        print("Validating with SHACL gold standard...")
        gold_graph = Graph().parse(gold_shacl_file, format='ttl')
        _, results_graph_gold, _ = validate(data_graph=data_graph, shacl_graph=gold_graph, inference='rdfs')
        with open(gold_path, 'wb') as f:
            pickle.dump(results_graph_gold, f)

    print("Validating with automatic SHACL...")
    _, results_graph_auto, _ = validate(data_graph=data_graph, shacl_graph=auto_graph, inference='rdfs')

    def extract_violations(graph):
        violations = set()
        for result in graph.subjects(RDF.type, SH.ValidationResult):
            focus = graph.value(result, SH.focusNode)
            path = graph.value(result, SH.resultPath)
            if focus is None:
                continue
            focus_str = str(focus)
            path_str = str(path) if path is not None else ''
            violations.add((focus_str, path_str))
        return violations

    violations_auto = extract_violations(results_graph_auto)
    violations_gold = extract_violations(results_graph_gold)

    # relaxed evaluation
    tp = set()
    fp = violations_auto.copy()
    fn = violations_gold.copy()

    matched_gold = set()

    for auto_violation in violations_auto:
        focus_auto, path_auto = auto_violation
        for gold_violation in violations_gold:
            if gold_violation in matched_gold:
                continue
            focus_gold, path_gold = gold_violation

            same_focus = focus_auto == focus_gold
            path_match = path_auto == path_gold or not path_gold

            if same_focus and path_match:
                tp.add(auto_violation)
                fp.discard(auto_violation)
                fn.discard(gold_violation)
                matched_gold.add(gold_violation)
                break  # matched

    precision = len(tp) / (len(tp) + len(fp)) if (tp or fp) else None
    recall = len(tp) / (len(tp) + len(fn)) if (tp or fn) else None
    f1_score = (2 * precision * recall / (precision + recall)) if (precision and recall) else None

    # parse file name to get model, temperature, technique
    filename = os.path.basename(auto_shacl_file)
    # assuming: rinf-application-guide-v1-6-1_model_temperature_technique.ttl
    parts = filename.replace(".ttl", "").split("_")
    if len(parts) >= 4:
        model = parts[1]
        temperature = parts[2]
        technique = parts[3]
    else:
        model = "unknown"
        temperature = "unknown"
        technique = "unknown"

    # write to CSV
    csv_file = f"{VALIDATION_DIR}/validation_results.csv"
    file_exists = os.path.exists(csv_file)

    with open(csv_file, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("model;temperature;technique;TP;FP;FN;precision;recall;f1-score\n")
        f.write(
            f"{model};{temperature};{technique};"
            f"{len(tp)};{len(fp)};{len(fn)};"
            f"{precision if precision is not None else 'NA'};"
            f"{recall if recall is not None else 'NA'};"
            f"{f1_score if f1_score is not None else 'NA'}\n"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    
    args = parser.parse_args()

    VALIDATION_DIR = Path(__file__).resolve().parent # root/validation
    evaluate_shacl_quality(
        data_file_nq=f"{VALIDATION_DIR}/data/ES.zip_combined-new.nq",
        auto_shacl_file=args.file,
        gold_shacl_file=f"{VALIDATION_DIR}/../era-shapes.ttl"
    )
