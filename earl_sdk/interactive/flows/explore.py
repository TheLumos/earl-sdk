"""Explore flow — browse dimensions, patients, and pipelines.

Read-only exploration of the EARL catalog: what evaluation criteria exist,
which simulated patients are available, and how your pipelines are configured.
"""

from __future__ import annotations

from rich.text import Text

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
    success,
    warn,
)


def flow_explore(client) -> None:
    """Top-level explore menu."""
    while True:
        action = select_one("Explore", [
            ("dims",      "Dimensions             — evaluation criteria used to judge doctor performance (1-4 scale)"),
            ("patients",  "Patients               — simulated patient profiles available for testing"),
            ("pipelines", "Pipelines              — evaluation configs that combine dimensions + patients + doctor"),
        ])
        if action is None:
            return
        if action == "dims":
            _flow_dimensions(client)
        elif action == "patients":
            _flow_patients(client)
        elif action == "pipelines":
            _flow_pipelines(client)


# ── Dimensions ────────────────────────────────────────────────────────────────


def _flow_dimensions(client) -> None:
    while True:
        console.print("\n  [dim]Loading dimensions...[/]")
        try:
            dims = client.dimensions.list()
        except Exception as e:
            error(f"Failed to load dimensions: {e}")
            return

        if not dims:
            warn("No dimensions found.")
            return

        # Table
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
            title=f"Evaluation Dimensions ({len(dims)} total)",
        )

        choices = [(d.id, f"{d.name}  [{d.category}]") for d in dims]
        choices.insert(0, ("create", "Create Custom Dimension — define your own evaluation criterion"))
        action = select_one("View dimension details or create new", choices)
        if action is None:
            return
        if action == "create":
            _create_dimension(client)
        else:
            _view_dimension(client, action, dims)


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


def _create_dimension(client) -> None:
    console.print("\n[bold]Create Custom Dimension[/]")
    muted("Define a new evaluation criterion. The judge will score doctors 1-4 on this dimension.\n")

    name = ask_text("Dimension name (e.g. 'bedside_manner', 'referral_appropriateness')")
    if not name:
        return
    description = ask_text("Description (what the judge should evaluate)")
    if not description:
        return
    category = ask_text("Category", default="custom") or "custom"
    weight_val = ask_text("Weight (0.0-5.0)", default="1.0") or "1.0"
    try:
        weight = float(weight_val)
    except ValueError:
        weight = 1.0

    try:
        dim = client.dimensions.create(
            name=name,
            description=description,
            category=category,
            weight=weight,
        )
        success(f"Created dimension '{dim.name}' (id={dim.id})")
    except Exception as e:
        error(f"Failed to create dimension: {e}")


# ── Patients ──────────────────────────────────────────────────────────────────


def _flow_patients(client) -> None:
    while True:
        diff_filter = select_one("Filter patients by difficulty", [
            ("all",    "All patients           — show every available patient profile"),
            ("easy",   "Easy                   — straightforward cases, single clear condition"),
            ("medium", "Medium                 — moderate complexity, some ambiguity"),
            ("hard",   "Hard                   — complex multi-condition cases with tricky presentations"),
        ], allow_back=True)
        if diff_filter is None:
            return

        console.print("\n  [dim]Loading patients...[/]")
        try:
            kwargs = {} if diff_filter == "all" else {"difficulty": diff_filter}
            patients = client.patients.list(**kwargs, limit=100)
        except Exception as e:
            error(f"Failed to load patients: {e}")
            return

        if not patients:
            warn("No patients found for this filter.")
            continue

        rows = []
        for p in patients:
            diff_style = {"easy": "green", "medium": "yellow", "hard": "red"}.get(p.difficulty, "dim")
            rows.append([
                p.id[:16],
                p.name,
                Text(p.difficulty, style=diff_style),
                ", ".join(p.conditions[:2]) + ("..." if len(p.conditions) > 2 else ""),
                ", ".join(p.tags[:2]) if p.tags else "",
            ])
        datatable(
            columns=[
                ("ID", "dim"),
                ("Name", "bold"),
                ("Difficulty", ""),
                ("Conditions", ""),
                ("Tags", "dim"),
            ],
            rows=rows,
            title=f"Patients ({len(patients)} found)",
        )

        choices = [(p.id, f"{p.name}  [{p.difficulty}]  {', '.join(p.conditions[:2])}") for p in patients]
        pick = select_one("View patient details", choices)
        if pick is None:
            continue
        _view_patient(client, pick, patients)


def _view_patient(client, patient_id: str, patients: list) -> None:
    patient = next((p for p in patients if p.id == patient_id), None)
    if not patient:
        try:
            patient = client.patients.get(patient_id)
        except Exception as e:
            error(f"Failed to load patient: {e}")
            return

    lines = [
        f"[bold]ID:[/]              {patient.id}",
        f"[bold]Name:[/]            {patient.name}",
        f"[bold]Difficulty:[/]      {patient.difficulty}",
        f"[bold]Conditions:[/]      {', '.join(patient.conditions)}",
    ]
    if patient.tags:
        lines.append(f"[bold]Tags:[/]            {', '.join(patient.tags)}")
    if hasattr(patient, "age") and patient.age:
        lines.append(f"[bold]Age:[/]             {patient.age}")
    if hasattr(patient, "gender") and patient.gender:
        lines.append(f"[bold]Gender:[/]          {patient.gender}")
    if hasattr(patient, "chief_complaint") and patient.chief_complaint:
        lines.append(f"")
        lines.append(f"[bold]Chief Complaint:[/]")
        lines.append(f"  {patient.chief_complaint}")
    if patient.description:
        lines.append(f"")
        lines.append(f"[bold]Description:[/]")
        lines.append(f"  {patient.description[:300]}")

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
    muted("A pipeline combines evaluation dimensions + patients + doctor config.\n")

    name = ask_text("Pipeline name (unique, e.g. 'my-gpt4-eval')")
    if not name:
        return None

    description = ask_text("Description (optional)", default="") or ""

    # Select dimensions
    console.print("\n  [dim]Loading dimensions...[/]")
    try:
        dims = client.dimensions.list()
    except Exception as e:
        error(f"Failed to load dimensions: {e}")
        return None
    if not dims:
        error("No dimensions available.")
        return None

    dim_choices = [(d.id, f"{d.name}  [{d.category}]  — {d.description[:60]}...") for d in dims]
    selected_dims = select_many("Select evaluation dimensions (space to toggle)", dim_choices)
    if not selected_dims:
        warn("No dimensions selected — cancelled.")
        return None

    # Select patients
    console.print("\n  [dim]Loading patients...[/]")
    try:
        patients = client.patients.list(limit=100)
    except Exception as e:
        error(f"Failed to load patients: {e}")
        return None
    if not patients:
        error("No patients available.")
        return None

    patient_choices = [(p.id, f"{p.name}  [{p.difficulty}]  {', '.join(p.conditions[:2])}") for p in patients]
    selected_patients = select_many("Select patients (space to toggle)", patient_choices)
    if not selected_patients:
        warn("No patients selected — cancelled.")
        return None

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
    info_panel("Pipeline Preview", [
        f"[bold]Name:[/]          {name}",
        f"[bold]Dimensions:[/]    {len(selected_dims)}",
        f"[bold]Patients:[/]      {len(selected_patients)}",
        f"[bold]Doctor:[/]        {doc_type}",
        f"[bold]Max Turns:[/]     {max_turns}",
    ])

    if not ask_confirm("Create this pipeline?"):
        muted("Cancelled.")
        return None

    try:
        pipeline = client.pipelines.create(
            name=name,
            dimension_ids=selected_dims,
            patient_ids=selected_patients,
            doctor_config=doctor_config,
            description=description,
            use_internal_doctor=(doc_type == "internal"),
            max_turns=max_turns,
        )
        success(f"Pipeline '{pipeline.name}' created with {len(selected_dims)} dimensions and {len(selected_patients)} patients")
        return pipeline.name
    except Exception as e:
        error(f"Failed to create pipeline: {e}")
        return None
