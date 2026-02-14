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
    select_with_preview,
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
        action = select_one("View dimension details", choices)
        if action is None:
            return
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

    # Select patients with live preview panel
    console.print("\n  [dim]Loading patients...[/]")
    try:
        patients = client.patients.list(limit=200)
    except Exception as e:
        error(f"Failed to load patients: {e}")
        return None
    if not patients:
        error("No patients available.")
        return None

    # Pre-fetch full details for preview panel
    console.print(f"  [dim]Loading patient details ({len(patients)} patients)...[/]")
    details_map: dict[str, object] = {}
    for p in patients:
        try:
            details_map[p.id] = client.patients.get(p.id)
        except Exception:
            details_map[p.id] = p  # fallback to list-level data

    # Build items and preview text
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
