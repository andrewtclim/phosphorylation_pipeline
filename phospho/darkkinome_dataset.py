"""Dark kinome weak-label dataset assembly helpers.

Planned contents:
- Build long-form PTM rows from bait/prey UniProt pairs.
- Use existing UniProt helpers to fetch entries and extract PTM texts.
- Apply role-to-label mapping (bait=1, prey=2, optional background=0).
"""

from __future__ import annotations

import pandas as pd

from phospho.uniprot import fetch_entry, get_ptm_texts


def build_darkkinome_ptm_rows(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Build long-form PTM rows for dark kinome weak-label training data.

    Planned behavior:
    - Consume canonical pair columns: experiment_id, bait_uniprot, prey_uniprot.
    - Fetch each unique UniProt entry (with per-run caching).
    - Extract PTM comment texts and emit one row per PTM text.
    - Apply deterministic role precedence (bait overrides prey).

    Planned minimum output columns:
    - uniprot_id
    - role
    - ptm_text
    - output_class

    Planned preferred extra columns:
    - experiment_id
    - bait_uniprot
    - prey_uniprot
    - source
    """

    # Validate canonical input schema expected from extract_pairs().
    required_cols = ["experiment_id", "bait_uniprot", "prey_uniprot"]
    missing = [c for c in required_cols if c not in pairs_df.columns]
    if missing:
        raise ValueError(
            f"pairs_df missing required column(s): {missing}. "
            f"Expected: {required_cols}"
        )

    # Determine role per protein with deterministic precedence: bait > prey.
    bait_ids = set(pairs_df["bait_uniprot"].dropna().astype(str))
    prey_ids = set(pairs_df["prey_uniprot"].dropna().astype(str))
    role_map: dict[str, str] = {}

    for uid in prey_ids:
        role_map[uid] = "prey"
    for uid in bait_ids:
        role_map[uid] = "bait"

    # Collect all experiments where each protein appears (for context columns).
    contexts_by_uid: dict[str, list[tuple[str, str, str]]] = {}
    for _, row in pairs_df.iterrows():
        exp_id = str(row["experiment_id"])
        bait_uid = str(row["bait_uniprot"])
        prey_uid = str(row["prey_uniprot"])

        contexts_by_uid.setdefault(bait_uid, []).append(
            (exp_id, bait_uid, prey_uid))
        contexts_by_uid.setdefault(prey_uid, []).append(
            (exp_id, bait_uid, prey_uid))

    # Cache UniProt PTM lookups so each UniProt accession is fetched once per run.
    ptm_cache: dict[str, list[str]] = {}
    output_rows: list[dict] = []

    for uid, role in role_map.items():
        if uid not in ptm_cache:
            entry = fetch_entry(uid)
            ptm_cache[uid] = get_ptm_texts(entry)

        ptm_texts = ptm_cache[uid]

        # Default behavior: skip proteins with zero PTM texts.
        if not ptm_texts:
            continue

        # Map role to weak label class.
        output_class = 1 if role == "bait" else 2

        # Emit one row per PTM text per experiment context where this protein appears.
        for exp_id, bait_uid, prey_uid in contexts_by_uid.get(uid, []):
            for ptm_text in ptm_texts:
                output_rows.append(
                    {
                        "uniprot_id": uid,
                        "role": role,
                        "ptm_text": ptm_text,
                        "output_class": output_class,
                        "experiment_id": exp_id,
                        "bait_uniprot": bait_uid,
                        "prey_uniprot": prey_uid,
                        "source": "dark_kinome_ppi",
                    }
                )

    # Build DataFrame with consistent column ordering.
    out_df = pd.DataFrame(output_rows)
    desired_cols = [
        "uniprot_id",
        "role",
        "ptm_text",
        "output_class",
        "experiment_id",
        "bait_uniprot",
        "prey_uniprot",
        "source",
    ]
    if out_df.empty:
        return pd.DataFrame(columns=desired_cols)

    return out_df[desired_cols]
