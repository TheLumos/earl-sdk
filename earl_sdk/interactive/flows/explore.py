"""Explore flow — browse dimensions, patients, and pipelines.

Read-only exploration of the EARL catalog: what evaluation criteria exist,
which simulated patients are available, and how your pipelines are configured.
"""

from __future__ import annotations

from rich.text import Text

from ...verifiers_catalog import parse_verifiers_list_payload

from ..ui import (
    ask_confirm,
    ask_int,
    ask_text,
    console,
    datatable,
    error,
    info_panel,
    muted,
    score_text,
    select_many,
    select_one,
    select_with_preview,
    success,
    warn,
)


def _normalize_verifier_ids(verifier_ids: list[str]) -> list[str]:
    """Convert legacy dimension IDs into Lumos verifier paths.

    Legacy IDs like ``turn-pacing`` are interpreted as scoring dimensions and
    mapped to ``scoring-dimensions/turn-pacing``.
    """
    normalized: list[str] = []
    for verifier_id in verifier_ids:
        if "/" in verifier_id:
            normalized.append(verifier_id)
        else:
            normalized.append(f"scoring-dimensions/{verifier_id}")
    return normalized


def flow_explore(client) -> None:
    """Top-level explore menu."""
    while True:
        action = select_one("Explore", [
            ("cases",     "Cases                  — pre-built clinical scenarios with case verifiers, gates, and scoring dims"),
            ("patients",  "Patients               — simulated patient profiles available for testing"),
            ("pipelines", "Pipelines              — evaluation configs that combine a case (or verifiers) + doctor"),
            ("dims",      "Verifiers              — all available hard gates, scoring dimensions, and evaluation criteria"),
        ])
        if action is None:
            return
        if action == "cases":
            _flow_cases(client)
        elif action == "dims":
            _flow_dimensions(client)
        elif action == "patients":
            _flow_patients(client)
        elif action == "pipelines":
            _flow_pipelines(client)


# ── Cases ─────────────────────────────────────────────────────────────────────


def _flow_cases(client) -> None:
    """Browse available evaluation cases."""
    console.print("\n  [dim]Loading cases...[/]")
    try:
        cases = client.cases.list()
    except Exception as e:
        error(f"Failed to load cases: {e}")
        return

    if not cases:
        warn("No cases available.")
        return

    while True:
        rows = []
        for c in cases:
            snap = c.get("case_snapshot") or {}
            name = c.get("name") or snap.get("name") or ""
            patient = (
                c.get("patient_id")
                or snap.get("patient_name")
                or snap.get("patient_id")
                or ""
            )
            verifiers = c.get("case_verifiers", snap.get("case_verifiers", 0))
            gates = c.get("hard_gates", snap.get("hard_gates", 0))
            dims = c.get("scoring_dimensions", snap.get("scoring_dimensions", 0))
            rows.append([
                c.get("case_id", ""),
                (name or "")[:40],
                str(verifiers),
                str(gates),
                str(dims),
                (patient or "")[:25],
            ])
        datatable(
            columns=[
                ("Case ID", "bold"),
                ("Name", ""),
                ("Verifiers", "cyan"),
                ("Gates", "green"),
                ("Dims", "magenta"),
                ("Patient", "dim"),
            ],
            rows=rows,
            title=f"Evaluation Cases ({len(cases)})",
        )

        case_choices = [(c["case_id"], f"{c.get('name', '') or c.get('case_snapshot', {}).get('name', '') or c['case_id']}  ({c.get('case_verifiers', 0)} verifiers)") for c in cases]
        case_choices.append(("back", "← Back"))
        picked = select_one("Inspect a case", case_choices, allow_back=False)
        if picked is None or picked == "back":
            return

        try:
            detail = client.cases.get(picked)
            totals = detail.get("totals", {})
            console.print()
            info_panel(f"Case: {detail.get('name', picked)}", [
                f"[bold]ID:[/]           {detail['case_id']}",
                f"[bold]Patient:[/]      {detail.get('patient_id', 'N/A')}",
                f"[bold]Specialty:[/]    {detail.get('medical_speciality', 'N/A')}",
                f"[bold]Encounter:[/]    {detail.get('encounter_type', 'N/A')}",
                "",
                f"[bold]Case Verifiers:[/]      {totals.get('case_verifiers', 0)}",
                f"[bold]Hard Gates:[/]          {totals.get('hard_gates', 0)}",
                f"[bold]Scoring Dimensions:[/]  {totals.get('scoring_dimensions', 0)}",
            ])

            if detail.get("description"):
                console.print(f"\n  [italic]{detail['description']}[/]")

            verifiers = detail.get("case_verifiers", [])
            if verifiers:
                console.print(f"\n  [bold]Case Verifiers ({len(verifiers)}):[/]")
                for v in verifiers:
                    pts = v.get("points", 0)
                    color = "green" if pts > 0 else "red"
                    console.print(f"    [{color}]{pts:+d}[/{color}]  {v.get('name', '?')}")

            gates = detail.get("default_hard_gates", [])
            if gates:
                console.print(f"\n  [bold]Default Hard Gates ({len(gates)}):[/]")
                for g in gates:
                    console.print(f"    • {g}")

            dims = detail.get("default_scoring_dimensions", [])
            if dims:
                console.print(f"\n  [bold]Default Scoring Dimensions ({len(dims)}):[/]")
                for d in dims:
                    console.print(f"    • {d}")

            console.print()
        except Exception as e:
            error(f"Failed to load case: {e}")


# ── Verifiers / Dimensions ────────────────────────────────────────────────────


def _fetch_lumos_verifiers(client) -> tuple[list[str], list[str]]:
    """Collect unique hard-gate and scoring-dimension IDs from all cases."""
    gates: set[str] = set()
    scoring: set[str] = set()
    try:
        cases = client.cases.list()
        for c in cases:
            detail = client.cases.get(c["case_id"])
            for g in detail.get("default_hard_gates", []):
                gates.add(g)
            for s in detail.get("default_scoring_dimensions", []):
                scoring.add(s)
    except Exception:
        pass
    return sorted(gates), sorted(scoring)


def _fetch_lumos_verifier_paths(client) -> tuple[list[str], list[str]]:
    """Load generic Lumos paths via ``GET /additional-verifiers``, else union of case defaults."""
    try:
        raw = client.verifiers.list()
        gates, scoring = parse_verifiers_list_payload(raw)
        if gates or scoring:
            return gates, scoring
    except Exception:
        pass
    return _fetch_lumos_verifiers(client)


def _build_additional_verifier_choices(
    client,
    case_detail: dict | None = None,
) -> list[tuple[str, str]]:
    """
    Selectable *extra* gates/scoring dims from the generic verifiers catalog.

    Case-local ``case_verifiers`` are never listed here — only platform catalog paths.
    When ``case_detail`` is set, default hard gates and scoring dimensions already
    included in that case are omitted so the menu focuses on true add-ons.
    """
    choices: list[tuple[str, str]] = []
    gates, scoring = _fetch_lumos_verifier_paths(client)
    skip: set[str] = set()
    if case_detail:
        skip.update(case_detail.get("default_hard_gates") or [])
        skip.update(case_detail.get("default_scoring_dimensions") or [])

    for g in gates:
        if g in skip:
            continue
        short = g.replace("hard-gates/", "")
        choices.append((g, f"🛡  {short}  [hard-gate]"))
    for s in scoring:
        if s in skip:
            continue
        short = s.replace("scoring-dimensions/", "")
        choices.append((s, f"📊 {short}  [scoring]"))

    choices.sort(key=lambda x: x[1].lower())
    if not choices and (gates or scoring) and skip:
        warn(
            "Every catalog verifier is already a default on this case — "
            "nothing extra to add."
        )
    return choices


def _build_verifier_choices(client) -> list[tuple[str, str]]:
    """Build a combined choices list: Lumos verifiers first, then legacy dims."""
    choices: list[tuple[str, str]] = []
    gates, scoring = _fetch_lumos_verifier_paths(client)
    if gates or scoring:
        for g in gates:
            short = g.replace("hard-gates/", "")
            choices.append((g, f"🛡  {short}  [hard-gate]"))
        for s in scoring:
            short = s.replace("scoring-dimensions/", "")
            choices.append((s, f"📊 {short}  [scoring]"))
    else:
        try:
            dims = client.dimensions.list()
            for d in dims:
                choices.append((d.id, f"{d.name}  [{d.category}]  — {d.description[:60]}..."))
        except Exception:
            pass
    return choices


def _flow_dimensions(client) -> None:
    console.print("\n  [dim]Loading verifiers...[/]")

    lumos_gates, lumos_scoring = _fetch_lumos_verifier_paths(client)
    has_lumos = bool(lumos_gates or lumos_scoring)

    if has_lumos:
        # Show Lumos verifiers as primary
        rows = []
        for g in lumos_gates:
            rows.append([g, Text("hard-gate", style="red bold"), "Fail/Pass — blocks the score if violated"])
        for s in lumos_scoring:
            rows.append([s, Text("scoring", style="cyan"), "1–4 scale dimension score"])
        datatable(
            columns=[
                ("Verifier ID", "bold"),
                ("Type", ""),
                ("Description", "dim"),
            ],
            rows=rows,
            title=f"Lumos Verifiers ({len(lumos_gates)} hard gates + {len(lumos_scoring)} scoring dims)",
        )
        muted(f"  Use these IDs in pipelines: verifier_ids=[\"scoring-dimensions/clinical-correctness\", ...]")
        console.print()

    # Also offer legacy dimensions
    try:
        dims = client.dimensions.list()
    except Exception as e:
        if not has_lumos:
            error(f"Failed to load dimensions: {e}")
        return

    if dims:
        label = "Legacy Dimensions" if has_lumos else "Evaluation Dimensions"
        if has_lumos:
            if not ask_confirm(f"Also show {len(dims)} legacy dimensions?", default=False):
                return

        while True:
            rows = []
            for d in dims:
                cat_style = "cyan" if d.category == "standard" else "magenta"
                rows.append([
                    d.id[:20],
                    d.name,
                    Text(d.category, style=cat_style),
                    f"{d.weight:.1f}",
                    "✓" if d.is_custom else "",
                ])
            datatable(
                columns=[
                    ("ID", "dim"),
                    ("Name", "bold"),
                    ("Category", ""),
                    ("Weight", ""),
                    ("Custom", ""),
                ],
                rows=rows,
                title=f"{label} ({len(dims)} total)",
            )

            choices = [(d.id, f"{d.name}  [{d.category}]") for d in dims]
            action = select_one("View dimension details", choices)
            if action is None:
                return
            _view_dimension(client, action, dims)
    elif not has_lumos:
        warn("No verifiers found.")


def _view_dimension(client, dim_id: str, dims: list) -> None:
    dim = next((d for d in dims if d.id == dim_id), None)
    if not dim:
        try:
            dim = client.dimensions.get(dim_id)
        except Exception as e:
            error(f"Failed to load dimension: {e}")
            return

    info_panel(f"Dimension: {dim.name}", [
        f"[bold]ID:[/]          {dim.id}",
        f"[bold]Category:[/]    {dim.category}",
        f"[bold]Weight:[/]      {dim.weight:.1f}",
        f"[bold]Custom:[/]      {'yes' if dim.is_custom else 'no (built-in)'}",
        f"",
        f"[bold]Description:[/]",
        f"  {dim.description or '(none)'}",
    ])


# ── Patients ──────────────────────────────────────────────────────────────────


def _format_patient_preview(p) -> str:
    """Format patient details as plain text for the prompt_toolkit preview panel."""
    import textwrap

    lines: list[str] = []
    lines.append(f"  Name:              {p.name}")
    if p.age:
        lines.append(f"  Age:               {p.age}")
    if p.gender:
        lines.append(f"  Gender:            {p.gender.capitalize()}")
    primary = p.condition or (p.conditions[0] if p.conditions else None)
    if primary:
        lines.append(f"  Primary Condition: {primary}")
    if getattr(p, "medical_speciality", None):
        lines.append(f"  Specialty:         {p.medical_speciality}")

    if getattr(p, "reason", None):
        lines.append("")
        lines.append("  Reason for Visit:")
        for ln in textwrap.wrap(p.reason, 65):
            lines.append(f"    {ln}")

    vignette = getattr(p, "clinical_vignette", None) or getattr(p, "vignette", None)
    if vignette:
        lines.append("")
        lines.append("  Clinical Vignette:")
        for ln in textwrap.wrap(vignette, 65):
            lines.append(f"    {ln}")

    if getattr(p, "patient_goal", None):
        lines.append("")
        lines.append("  Patient Goal:")
        for ln in textwrap.wrap(p.patient_goal, 65):
            lines.append(f"    {ln}")

    if getattr(p, "doctor_goal", None):
        lines.append("")
        lines.append("  Doctor Goal:")
        for ln in textwrap.wrap(p.doctor_goal, 65):
            lines.append(f"    {ln}")

    if not any(getattr(p, f, None) for f in ("reason", "clinical_vignette", "vignette", "patient_goal", "doctor_goal")):
        lines.append("")
        lines.append("  (No detailed encounter info available)")

    return "\n".join(lines)


def _flow_patients(client) -> None:
    console.print("\n  [dim]Loading patients...[/]")
    try:
        patients = client.patients.list(limit=200)
    except Exception as e:
        error(f"Failed to load patients: {e}")
        return

    if not patients:
        warn("No patients found.")
        return

    # Pre-fetch full details for preview panel
    console.print(f"  [dim]Loading patient details ({len(patients)} patients)...[/]")
    details_map: dict[str, object] = {}
    for p in patients:
        try:
            details_map[p.id] = client.patients.get(p.id)
        except Exception:
            details_map[p.id] = p

    # Build items and previews
    items: list[tuple[str, str]] = []
    previews: dict[str, str] = {}
    for p in patients:
        primary = p.condition or (p.conditions[0] if p.conditions else "")
        label = f"{p.name}"
        if p.age:
            label += f"  {p.age}y"
        if p.gender:
            label += f" {p.gender[0].upper()}"
        if primary:
            label += f"  — {primary}"
        items.append((p.id, label))
        previews[p.id] = _format_patient_preview(details_map.get(p.id, p))

    # Single-select with live preview — user can browse and press ESC to go back
    select_with_preview(
        f"Patients ({len(patients)} available)  — browse with ↑↓, press ESC to go back",
        items,
        previews,
        multi=False,
    )


def _view_patient(client, patient_id: str) -> None:
    """Fetch full patient details from the API and display them."""
    console.print("  [dim]Loading patient details...[/]")
    try:
        patient = client.patients.get(patient_id)
    except Exception as e:
        error(f"Failed to load patient: {e}")
        return

    # ── Header info ──
    lines = [
        f"[bold]ID:[/]                {patient.id}",
        f"[bold]Name:[/]              {patient.name}",
    ]
    if patient.age:
        lines.append(f"[bold]Age:[/]               {patient.age}")
    if patient.gender:
        lines.append(f"[bold]Gender:[/]            {patient.gender.capitalize()}")
    primary = patient.condition or (patient.conditions[0] if patient.conditions else None)
    if primary:
        lines.append(f"[bold]Primary Condition:[/] {primary}")
    if patient.medical_speciality:
        lines.append(f"[bold]Specialty:[/]         {patient.medical_speciality}")
    if patient.difficulty:
        lines.append(f"[bold]Difficulty:[/]        {patient.difficulty}")

    # ── Encounter details ──
    if patient.reason:
        lines.append("")
        lines.append(f"[bold]Reason for Visit:[/]")
        lines.append(f"  {patient.reason}")

    vignette = patient.clinical_vignette or patient.vignette
    if vignette:
        lines.append("")
        lines.append(f"[bold]Clinical Vignette:[/]")
        lines.append(f"  {vignette}")

    if patient.patient_goal:
        lines.append("")
        lines.append(f"[bold]Patient Goal:[/]")
        lines.append(f"  {patient.patient_goal}")

    if patient.doctor_goal:
        lines.append("")
        lines.append(f"[bold]Doctor Goal:[/]")
        lines.append(f"  {patient.doctor_goal}")

    # ── Extra context (if present) ──
    if patient.conditions and len(patient.conditions) > 1:
        lines.append("")
        lines.append(f"[bold]All Conditions:[/]    {', '.join(patient.conditions)}")
    if patient.chief_complaint and patient.chief_complaint != patient.reason:
        lines.append(f"[bold]Chief Complaint:[/]   {patient.chief_complaint}")
    if patient.tags:
        lines.append(f"[bold]Tags:[/]              {', '.join(patient.tags)}")

    info_panel(f"Patient: {patient.name}", lines)


# ── Pipelines ─────────────────────────────────────────────────────────────────


def _flow_pipelines(client) -> None:
    while True:
        console.print("\n  [dim]Loading pipelines...[/]")
        try:
            pipelines = client.pipelines.list()
        except Exception as e:
            error(f"Failed to load pipelines: {e}")
            return

        if not pipelines:
            warn("No pipelines found. Create one to get started.")

        rows = []
        for p in pipelines:
            doc_type = "Lumos's"
            if p.doctor_api:
                type_map = {"external": "Client's", "internal": "Lumos's", "client_driven": "Client-Driven"}
                doc_type = type_map.get(p.doctor_api.type, p.doctor_api.type)
            dims_str = str(len(p.dimension_ids)) if p.dimension_ids else "–"
            pats_str = str(len(p.patient_ids)) if p.patient_ids else "–"
            rows.append([
                p.name,
                dims_str,
                pats_str,
                doc_type,
                Text("✓", style="green") if p.is_active else Text("✗", style="dim"),
            ])
        if rows:
            datatable(
                columns=[
                    ("Name", "bold"),
                    ("Dims", ""),
                    ("Patients", ""),
                    ("Doctor", ""),
                    ("Active", ""),
                ],
                rows=rows,
                title=f"Pipelines ({len(pipelines)} total)",
            )

        choices = []
        for p in pipelines:
            parts = [p.name, " —"]
            if p.dimension_ids:
                parts.append(f" {len(p.dimension_ids)} dims,")
            if p.patient_ids:
                parts.append(f" {len(p.patient_ids)} patients")
            else:
                # Remove trailing comma from dims part if no patients
                if parts[-1].endswith(","):
                    parts[-1] = parts[-1][:-1]
            choices.append((p.name, "".join(parts)))
        choices.insert(0, ("create", "Create Pipeline        — build a new evaluation config from dimensions + patients + doctor"))
        action = select_one("View pipeline or create new", choices)
        if action is None:
            return
        if action == "create":
            created_name = _create_pipeline(client)
            if created_name:
                muted("Tip: use Run Simulation from the main menu to run this pipeline.")
        else:
            _view_pipeline(client, action, pipelines)


def _view_pipeline(client, name: str, pipelines: list) -> None:
    pipeline = next((p for p in pipelines if p.name == name), None)
    if not pipeline:
        try:
            pipeline = client.pipelines.get(name)
        except Exception as e:
            error(f"Failed to load pipeline: {e}")
            return

    doc_type = "Lumos's (built-in)"
    doc_url = ""
    if pipeline.doctor_api:
        type_map = {"external": "Client's", "internal": "Lumos's", "client_driven": "Client-Driven"}
        doc_type = type_map.get(pipeline.doctor_api.type, pipeline.doctor_api.type)
        if pipeline.doctor_api.api_url:
            doc_url = pipeline.doctor_api.api_url

    lines = [
        f"[bold]Name:[/]          {pipeline.name}",
        f"[bold]Active:[/]        {'yes' if pipeline.is_active else 'no'}",
        f"[bold]Dimensions:[/]    {len(pipeline.dimension_ids)} — {', '.join(pipeline.dimension_ids[:5])}{'...' if len(pipeline.dimension_ids) > 5 else ''}",
        f"[bold]Patients:[/]      {len(pipeline.patient_ids)}",
        f"[bold]Doctor:[/]        {doc_type}",
    ]
    if doc_url:
        lines.append(f"[bold]Doctor URL:[/]    {doc_url}")
    if pipeline.conversation:
        lines.append(f"[bold]Initiator:[/]     {pipeline.conversation.initiator}")
        lines.append(f"[bold]Max Turns:[/]     {pipeline.conversation.max_turns}")
    if pipeline.description:
        lines.append(f"")
        lines.append(f"[bold]Description:[/]   {pipeline.description}")

    info_panel(f"Pipeline: {pipeline.name}", lines)

    action = select_one("Pipeline Actions", [
        ("delete", "Delete Pipeline        — permanently remove this pipeline"),
    ])
    if action == "delete":
        if ask_confirm(f"Delete pipeline '{pipeline.name}'?", default=False):
            try:
                client.pipelines.delete(pipeline.name)
                success(f"Pipeline '{pipeline.name}' deleted")
            except Exception as e:
                error(f"Failed to delete: {e}")


def _create_pipeline(client) -> str | None:
    """Wizard to create a new pipeline.

    Returns the pipeline name on success, or ``None`` if cancelled/failed.
    """
    return create_pipeline_wizard(client)


def create_pipeline_wizard(client) -> str | None:
    """Create-pipeline wizard shared between Explore and Create-&-Run flows.

    Returns the pipeline name on success, or ``None`` if cancelled/failed.
    """
    console.print("\n[bold]Create Pipeline[/]")
    muted("A pipeline combines a clinical case (or custom verifiers) + patients + doctor config.\n")

    import time as _time
    default_name = f"eval-{int(_time.time()) % 100000}"
    name = ask_text("Pipeline name", default=default_name)
    if not name:
        return None

    description = ask_text("Description (optional)", default="") or ""

    # Case selection — offer pre-defined cases first
    selected_case_id = None
    case_detail: dict | None = None
    selected_dims = []
    selected_patients = []
    try:
        cases = client.cases.list()
    except Exception:
        cases = []

    if cases:
        case_choices = [("none", "Custom — pick verifiers and patients manually")]
        for c in cases:
            totals = f"{c.get('case_verifiers', 0)} verifiers, {c.get('hard_gates', 0)} gates, {c.get('scoring_dimensions', 0)} scoring dims"
            case_choices.append((c["case_id"], f"{c.get('name', '') or c.get('case_snapshot', {}).get('name', '') or c['case_id']}  ({totals})"))

        picked = select_one("Start from a clinical case?", case_choices, allow_back=False)
        if not picked:
            return None

        if picked != "none":
            selected_case_id = picked
            try:
                case_detail = client.cases.get(selected_case_id)
                totals = case_detail.get("totals", {})
                success(
                    f"Case: {case_detail.get('name', selected_case_id)} — "
                    f"{totals.get('case_verifiers', 0)} case verifiers, "
                    f"{totals.get('hard_gates', 0)} hard gates, "
                    f"{totals.get('scoring_dimensions', 0)} scoring dims"
                )
                muted(f"  {case_detail.get('description', '')[:120]}")
                muted(f"  Patient: {case_detail.get('patient_id', 'auto')}")
            except Exception as e:
                warn(f"Could not load case details: {e}")

    extra_verifiers = []
    if selected_case_id:
        if ask_confirm("Add extra verifiers on top of the case defaults?", default=False):
            console.print("\n  [dim]Loading generic verifiers catalog (API)...[/]")
            verifier_choices = _build_additional_verifier_choices(client, case_detail)
            if verifier_choices:
                extra = select_many("Select additional verifiers (space to toggle)", verifier_choices)
                if extra:
                    extra_verifiers = extra
                    success(f"Added {len(extra)} extra verifiers")
            else:
                warn("No generic verifiers available to add (catalog empty or unreachable).")
    else:
        # Manual verifier selection (no case)
        console.print("\n  [dim]Loading verifiers...[/]")
        verifier_choices = _build_verifier_choices(client)
        if not verifier_choices:
            error("No verifiers available.")
            return None

        selected_dims = select_many("Select evaluation verifiers (space to toggle)", verifier_choices)
        if not selected_dims:
            warn("No verifiers selected — cancelled.")
            return None

    if selected_case_id:
        muted("Patient is set by the case — skipping patient selection.")
    else:
        console.print("\n  [dim]Loading patients...[/]")
        try:
            patients = client.patients.list(limit=200)
        except Exception as e:
            error(f"Failed to load patients: {e}")
            return None
        if not patients:
            error("No patients available.")
            return None

        console.print(f"  [dim]Loading patient details ({len(patients)} patients)...[/]")
        details_map: dict[str, object] = {}
        for p in patients:
            try:
                details_map[p.id] = client.patients.get(p.id)
            except Exception:
                details_map[p.id] = p

        patient_items: list[tuple[str, str]] = []
        patient_previews: dict[str, str] = {}
        for p in patients:
            primary = p.condition or (p.conditions[0] if p.conditions else "")
            label = f"{p.name}"
            if p.age:
                label += f"  {p.age}y"
            if p.gender:
                label += f" {p.gender[0].upper()}"
            if primary:
                label += f"  — {primary}"
            patient_items.append((p.id, label))
            patient_previews[p.id] = _format_patient_preview(details_map.get(p.id, p))

        selected_patients = select_with_preview(
            "Select patients (SPACE to toggle, ENTER when done)",
            patient_items,
            patient_previews,
            multi=True,
        )
        if not selected_patients:
            warn("No patients selected — cancelled.")
            return None

        selected_names = []
        for pid in selected_patients:
            d = details_map.get(pid)
            selected_names.append(d.name if d else pid)
        muted(f"Selected {len(selected_patients)} patient{'s' if len(selected_patients) != 1 else ''}: {', '.join(selected_names)}")

    # Doctor config
    from earl_sdk import DoctorApiConfig
    doc_type = select_one("Doctor type", [
        ("internal",       "Lumos's Doctor        — EARL's built-in AI doctor, no setup needed"),
        ("external",       "Client's Doctor       — your own API that EARL will call with patient messages"),
        ("client_driven",  "Client-Driven         — you call your doctor yourself, submit responses via SDK"),
    ], allow_back=False)
    if not doc_type:
        return None

    doctor_config = None
    if doc_type == "internal":
        doctor_config = None  # None = internal
    elif doc_type == "external":
        api_url = ask_text("Doctor API URL (e.g. https://my-doctor.example.com/chat)")
        if not api_url:
            return None
        api_key = ask_text("API Key (optional)", secret=True) or None
        doctor_config = DoctorApiConfig.external(api_url=api_url, api_key=api_key)
    elif doc_type == "client_driven":
        doctor_config = DoctorApiConfig.client_driven()

    # Conversation settings
    max_turns = ask_int("Max conversation turns", default=10, min_val=1, max_val=50) or 10

    # Confirm
    console.print()
    preview_lines = [f"[bold]Name:[/]          {name}"]
    if selected_case_id:
        preview_lines.append(f"[bold]Case:[/]          {selected_case_id}")
        preview_lines.append(f"[bold]Verifiers:[/]     case defaults (verifiers + gates + dims)")
        if extra_verifiers:
            preview_lines.append(f"[bold]Extra:[/]         +{len(extra_verifiers)} additional verifiers")
        preview_lines.append(f"[bold]Patient:[/]       from case (auto)")
    else:
        preview_lines.append(f"[bold]Verifiers:[/]     {len(selected_dims)}")
        preview_lines.append(f"[bold]Patients:[/]      {len(selected_patients)}")
    preview_lines.append(f"[bold]Doctor:[/]        {doc_type}")
    preview_lines.append(f"[bold]Max Turns:[/]     {max_turns}")
    info_panel("Pipeline Preview", preview_lines)

    if not ask_confirm("Create this pipeline?"):
        muted("Cancelled.")
        return None

    try:
        normalized_extra_verifiers = _normalize_verifier_ids(extra_verifiers) if extra_verifiers else []
        normalized_selected_dims = _normalize_verifier_ids(selected_dims) if selected_dims else []

        create_kwargs = dict(
            name=name,
            description=description,
            use_internal_doctor=(doc_type == "internal"),
            max_turns=max_turns,
            verifiers="lumos",
        )
        if selected_case_id:
            create_kwargs["case_id"] = selected_case_id
            if normalized_extra_verifiers:
                create_kwargs["verifier_ids"] = normalized_extra_verifiers
        else:
            create_kwargs["verifier_ids"] = normalized_selected_dims
            create_kwargs["patient_ids"] = selected_patients
        if doctor_config is not None:
            create_kwargs["doctor_config"] = doctor_config

        pipeline = client.pipelines.create(**create_kwargs)
        if selected_case_id:
            success(f"Pipeline '{pipeline.name}' created with case '{selected_case_id}'")
        else:
            success(f"Pipeline '{pipeline.name}' created with {len(selected_dims)} verifiers and {len(selected_patients)} patients")
        return pipeline.name
    except Exception as e:
        error(f"Failed to create pipeline: {e}")
        return None
