"""
KGE experiments: KB size sensitivity, nearest neighbors, t-SNE, relation analysis,
rule-based vs embedding-based reasoning comparison.
"""

import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def kb_size_sensitivity(
    data_dir: str = "kge_datasets",
    sizes: list[int] | None = None,
    output_path: str = "reports/kb_size_sensitivity.csv",
) -> pd.DataFrame:
    """
    Train TransE on subsets of the KB (20k, 50k, full) and compare metrics.
    """
    from pykeen.triples import TriplesFactory
    from pykeen.pipeline import pipeline

    if sizes is None:
        sizes = [20000, 50000]

    train_path = Path(data_dir) / "train.txt"
    valid_path = Path(data_dir) / "valid.txt"
    test_path = Path(data_dir) / "test.txt"

    # Load full dataset
    full_training = TriplesFactory.from_path(train_path)

    results_rows = []

    # Add full size to experiments
    all_sizes = sizes + [full_training.num_triples]

    for size in all_sizes:
        label = f"{size // 1000}k" if size >= 1000 else str(size)
        if size == full_training.num_triples:
            label = f"full ({size})"

        logger.info(f"Training TransE on {label} triples")

        if size >= full_training.num_triples:
            training_subset = full_training
        else:
            # Subsample triples
            indices = list(range(full_training.num_triples))
            random.seed(42)
            random.shuffle(indices)
            subset_indices = sorted(indices[:size])
            training_subset = full_training.new_with_restriction(
                entities=None,
                relations=None,
            )
            # Use create_inverse_triples approach: take a slice
            import torch
            subset_triples = full_training.mapped_triples[subset_indices]
            training_subset = TriplesFactory(
                mapped_triples=subset_triples,
                entity_to_id=full_training.entity_to_id,
                relation_to_id=full_training.relation_to_id,
            )

        validation = TriplesFactory.from_path(
            valid_path,
            entity_to_id=full_training.entity_to_id,
            relation_to_id=full_training.relation_to_id,
        )
        testing = TriplesFactory.from_path(
            test_path,
            entity_to_id=full_training.entity_to_id,
            relation_to_id=full_training.relation_to_id,
        )

        try:
            result = pipeline(
                model="TransE",
                training=training_subset,
                validation=validation,
                model_kwargs={"embedding_dim": 200},
                training_kwargs={"num_epochs": 50, "batch_size": 256},
                optimizer_kwargs={"lr": 0.001},
                random_seed=42,
            )

            from kge.evaluate import evaluate_model
            metrics = evaluate_model(result, testing)
            metrics["KB_Size"] = label
            metrics["Num_Triples"] = size
            results_rows.append(metrics)
        except Exception as e:
            logger.warning(f"Training failed for size {label}: {e}")

    df = pd.DataFrame(results_rows)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info(f"Size sensitivity results saved to {out}")
    return df


def nearest_neighbors(
    result,
    training,
    entity_names: list[str] | None = None,
    k: int = 10,
    output_path: str = "reports/nearest_neighbors.json",
) -> dict:
    """Find k nearest neighbors in embedding space for selected entities."""
    import torch

    model = result.model
    entity_embeddings = model.entity_representations[0](indices=None).detach().cpu().numpy()

    id_to_entity = {v: k for k, v in training.entity_to_id.items()}

    # If no entities specified, pick some well-connected ones
    if entity_names is None:
        # Pick first 5 entities by ID
        entity_names = [id_to_entity[i] for i in range(min(5, len(id_to_entity)))]

    results = {}
    for entity_name in entity_names:
        if entity_name not in training.entity_to_id:
            # Try partial match
            matches = [e for e in training.entity_to_id if entity_name.lower() in e.lower()]
            if matches:
                entity_name = matches[0]
            else:
                logger.warning(f"Entity '{entity_name}' not found, skipping")
                continue

        entity_id = training.entity_to_id[entity_name]
        entity_vec = entity_embeddings[entity_id]

        # Compute cosine similarity
        norms = np.linalg.norm(entity_embeddings, axis=1)
        entity_norm = np.linalg.norm(entity_vec)
        if entity_norm == 0:
            continue

        similarities = entity_embeddings @ entity_vec / (norms * entity_norm + 1e-8)
        top_indices = np.argsort(-similarities)[1:k + 1]  # exclude self

        neighbors = []
        for idx in top_indices:
            neighbors.append({
                "entity": id_to_entity.get(idx, f"entity_{idx}"),
                "similarity": float(similarities[idx]),
            })

        results[entity_name] = neighbors
        logger.info(f"Nearest neighbors of '{entity_name}':")
        for n in neighbors[:5]:
            logger.info(f"  {n['entity']} (sim={n['similarity']:.4f})")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Nearest neighbors saved to {out}")
    return results


def tsne_visualization(
    result,
    training,
    output_path: str = "reports/tsne.png",
    max_entities: int = 2000,
) -> Path:
    """Run t-SNE on entity embeddings and plot colored by entity type."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    model = result.model
    embeddings = model.entity_representations[0](indices=None).detach().cpu().numpy()

    id_to_entity = {v: k for k, v in training.entity_to_id.items()}

    # Subsample if too many entities
    n_entities = embeddings.shape[0]
    if n_entities > max_entities:
        indices = random.sample(range(n_entities), max_entities)
        embeddings_subset = embeddings[indices]
        entities_subset = [id_to_entity.get(i, "") for i in indices]
    else:
        embeddings_subset = embeddings
        entities_subset = [id_to_entity.get(i, "") for i in range(n_entities)]

    # Assign colors by URI prefix / type
    def _get_type(uri: str) -> str:
        if "wikidata.org" in uri:
            return "Wikidata"
        elif "ai-news" in uri:
            return "AI-News (local)"
        else:
            return "Other"

    types = [_get_type(e) for e in entities_subset]
    unique_types = sorted(set(types))
    color_map = {t: i for i, t in enumerate(unique_types)}
    colors = [color_map[t] for t in types]

    logger.info(f"Running t-SNE on {len(embeddings_subset)} entity embeddings...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_subset) - 1))
    coords = tsne.fit_transform(embeddings_subset)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        coords[:, 0], coords[:, 1],
        c=colors, cmap="Set1", alpha=0.6, s=10,
    )
    plt.title("t-SNE Visualization of Entity Embeddings")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")

    # Legend
    handles = []
    for t in unique_types:
        h = plt.scatter([], [], c=[plt.cm.Set1(color_map[t] / max(len(unique_types), 1))], label=t, s=50)
        handles.append(h)
    plt.legend(handles=handles, title="Entity Source")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"t-SNE plot saved to {out}")
    return out


def relation_behavior_analysis(
    result,
    training,
    output_path: str = "reports/relation_analysis.csv",
) -> pd.DataFrame:
    """Analyze relation embeddings: symmetry, inverse, composition patterns."""
    import torch

    model = result.model
    relation_embeddings = model.relation_representations[0](indices=None).detach().cpu().numpy()

    id_to_relation = {v: k for k, v in training.relation_to_id.items()}

    rows = []
    n_relations = relation_embeddings.shape[0]

    for i in range(n_relations):
        rel_name = id_to_relation.get(i, f"rel_{i}")
        rel_vec = relation_embeddings[i]
        norm = float(np.linalg.norm(rel_vec))

        # Check symmetry: r ≈ -r (symmetric relation)
        neg_vec = -rel_vec
        sym_score = float(np.dot(rel_vec, neg_vec) / (norm ** 2 + 1e-8))

        # Find most similar relation (potential inverse)
        best_sim = -1
        best_inverse = ""
        for j in range(n_relations):
            if i == j:
                continue
            other_vec = relation_embeddings[j]
            # Check if r_j ≈ -r_i (inverse)
            sim = float(np.dot(other_vec, neg_vec) / (
                np.linalg.norm(other_vec) * norm + 1e-8
            ))
            if sim > best_sim:
                best_sim = sim
                best_inverse = id_to_relation.get(j, f"rel_{j}")

        rows.append({
            "relation": rel_name,
            "norm": round(norm, 4),
            "symmetry_score": round(sym_score, 4),
            "potential_inverse": best_inverse,
            "inverse_similarity": round(best_sim, 4),
        })

    df = pd.DataFrame(rows)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info(f"Relation analysis saved to {out}")
    return df


def rule_vs_embedding_comparison(
    result,
    training,
    output_path: str = "reports/rule_vs_embedding.csv",
) -> pd.DataFrame:
    """
    Compare SWRL-style rules with embedding vector arithmetic.
    Example: vector(develops) + vector(AIProduct_type) ≈ vector(AICompany_type)?
    """
    import torch

    model = result.model
    relation_embeddings = model.relation_representations[0](indices=None).detach().cpu().numpy()
    entity_embeddings = model.entity_representations[0](indices=None).detach().cpu().numpy()

    id_to_relation = {v: k for k, v in training.relation_to_id.items()}
    id_to_entity = {v: k for k, v in training.entity_to_id.items()}

    rows = []

    # Find relations that could correspond to rule components
    rel_names = {id_to_relation.get(i, ""): i for i in range(len(id_to_relation))}

    # Look for patterns like: r1 + r2 ≈ r3
    relation_list = list(range(min(len(id_to_relation), 50)))  # limit to first 50
    for i in relation_list:
        for j in relation_list:
            if i >= j:
                continue
            composed = relation_embeddings[i] + relation_embeddings[j]
            composed_norm = np.linalg.norm(composed)
            if composed_norm < 1e-8:
                continue

            # Find closest relation to the composition
            best_sim = -1
            best_k = -1
            for k in relation_list:
                sim = float(np.dot(relation_embeddings[k], composed) / (
                    np.linalg.norm(relation_embeddings[k]) * composed_norm + 1e-8
                ))
                if sim > best_sim:
                    best_sim = sim
                    best_k = k

            if best_sim > 0.7:  # only report strong matches
                rows.append({
                    "relation_1": id_to_relation.get(i, f"r_{i}"),
                    "relation_2": id_to_relation.get(j, f"r_{j}"),
                    "composed_closest": id_to_relation.get(best_k, f"r_{best_k}"),
                    "cosine_similarity": round(best_sim, 4),
                })

    df = pd.DataFrame(rows)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info(f"Rule vs embedding comparison saved to {out} ({len(rows)} compositions found)")
    return df


def run_experiments(
    trained_results: dict | None = None,
    data_dir: str = "kge_datasets",
    models_dir: str = "data/kge_models",
    output_dir: str = "reports",
) -> None:
    """Run all experiments."""
    from kge.train_models import load_pykeen_dataset

    training, validation, testing = load_pykeen_dataset(data_dir)

    # Use provided results or the first available model
    if trained_results is None:
        logger.warning("No trained results provided, loading from disk")
        from pykeen.pipeline import PipelineResult
        models_path = Path(models_dir)
        trained_results = {}
        for model_dir in models_path.iterdir():
            if model_dir.is_dir():
                try:
                    result = PipelineResult.from_directory(str(model_dir))
                    trained_results[model_dir.name] = result
                except Exception as e:
                    logger.warning(f"Could not load {model_dir.name}: {e}")

    # Pick best model for detailed analysis (use first available)
    if not trained_results:
        logger.error("No trained models available for experiments")
        return

    best_model_name = list(trained_results.keys())[0]
    best_result = trained_results[best_model_name]
    logger.info(f"Using {best_model_name} for detailed analysis")

    # Experiment 1: KB size sensitivity
    logger.info("=" * 60)
    logger.info("Experiment 1: KB Size Sensitivity")
    try:
        kb_size_sensitivity(data_dir, output_path=f"{output_dir}/kb_size_sensitivity.csv")
    except Exception as e:
        logger.warning(f"KB size sensitivity failed: {e}")

    # Experiment 2: Nearest neighbors
    logger.info("=" * 60)
    logger.info("Experiment 2: Nearest Neighbors")
    try:
        nearest_neighbors(
            best_result, training,
            output_path=f"{output_dir}/nearest_neighbors.json",
        )
    except Exception as e:
        logger.warning(f"Nearest neighbors failed: {e}")

    # Experiment 3: t-SNE visualization
    logger.info("=" * 60)
    logger.info("Experiment 3: t-SNE Visualization")
    for model_name, result in trained_results.items():
        try:
            tsne_visualization(
                result, training,
                output_path=f"{output_dir}/tsne_{model_name}.png",
            )
        except Exception as e:
            logger.warning(f"t-SNE for {model_name} failed: {e}")

    # Experiment 4: Relation behavior analysis
    logger.info("=" * 60)
    logger.info("Experiment 4: Relation Behavior Analysis")
    try:
        relation_behavior_analysis(
            best_result, training,
            output_path=f"{output_dir}/relation_analysis.csv",
        )
    except Exception as e:
        logger.warning(f"Relation analysis failed: {e}")

    # Experiment 5: Rule vs embedding comparison
    logger.info("=" * 60)
    logger.info("Experiment 5: Rule vs Embedding Comparison")
    try:
        rule_vs_embedding_comparison(
            best_result, training,
            output_path=f"{output_dir}/rule_vs_embedding.csv",
        )
    except Exception as e:
        logger.warning(f"Rule vs embedding comparison failed: {e}")

    logger.info("=" * 60)
    logger.info("All experiments complete!")


if __name__ == "__main__":
    run_experiments()
