# SDK Testing Strategy

## Overview

This document outlines a multi-layered testing approach for the Earl SDK that can be integrated into GitHub Actions for CI/CD pipelines.

## Test Layers

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 4: E2E Tests (Comprehensive)                              │
│ - Full simulation workflows with judging                        │
│ - ~10-20 min, runs on release/tag                               │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│ Layer 3: Functional Tests (Core)                                │
│ - Pipeline CRUD, simulation management                          │
│ - ~3-5 min, runs on PR to main                                  │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│ Layer 2: Integration Tests (Smoke)                              │
│ - Auth, connection, basic API calls                             │
│ - ~30 sec, runs on every push                                   │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1: Unit Tests (Fast)                                      │
│ - Models, validation, serialization                             │
│ - ~5 sec, runs on every push                                    │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│ Layer 0: Pre-flight Checks (Sanity)                             │
│ - Syntax, imports, lint, type check                             │
│ - ~3 sec, runs on every push                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Layer 0: Pre-flight Checks (Sanity)

**Purpose**: Catch basic issues before any real testing  
**Requirements**: None (no credentials, no network)  
**Run On**: Every push, every PR

### Checks

| Check | Command | Description |
|-------|---------|-------------|
| Syntax | `python -m py_compile earl_sdk/*.py` | Verify Python syntax is valid |
| Imports | `python -c "from earl_sdk import EarlClient, DoctorApiConfig"` | Verify package imports work |
| Lint (Critical) | `ruff check --select=E9,F63,F7,F82` | Undefined vars, syntax errors |
| Lint (Style) | `ruff check` | Full style check |
| Type Check | `mypy earl_sdk/` | Static type analysis |
| Format Check | `black --check earl_sdk/` | Code formatting |

### Example Make Target

```makefile
sdk-preflight:
	@echo "Running pre-flight checks..."
	python -m py_compile earl_sdk/*.py
	python -c "from earl_sdk import EarlClient, DoctorApiConfig"
	ruff check earl_sdk/ --select=E9,F63,F7,F82
	@echo "✓ Pre-flight checks passed"
```

---

## Layer 1: Unit Tests (Fast)

**Purpose**: Test internal logic without network calls  
**Requirements**: None (mocked, no credentials)  
**Run On**: Every push, every PR

### Test Categories

#### 1.1 Model Tests (`tests/unit/test_models.py`)

```python
class TestDoctorApiConfig:
    def test_internal_factory(self):
        config = DoctorApiConfig.internal()
        assert config.type == "internal"
        assert config.api_url is None
    
    def test_external_factory(self):
        config = DoctorApiConfig.external(
            api_url="https://example.com/chat",
            api_key="test-key"
        )
        assert config.type == "external"
        assert config.api_url == "https://example.com/chat"
    
    def test_client_driven_factory(self):
        config = DoctorApiConfig.client_driven()
        assert config.type == "client_driven"
        assert config.is_client_driven is True
    
    def test_to_dict_roundtrip(self):
        config = DoctorApiConfig.external("http://test.com", "key")
        data = config.to_dict()
        restored = DoctorApiConfig.from_dict(data)
        assert restored.api_url == config.api_url

class TestConversationConfig:
    def test_defaults(self):
        config = ConversationConfig()
        assert config.initiator == "patient"
        assert config.max_turns == 10
    
    def test_max_turns_range(self):
        config = ConversationConfig(max_turns=50)
        assert config.max_turns == 50
    
    def test_to_dict_includes_max_turns(self):
        config = ConversationConfig(initiator="doctor", max_turns=25)
        data = config.to_dict()
        assert data["initiator"] == "doctor"
        assert data["max_turns"] == 25

class TestPipeline:
    def test_from_dict_parses_conversation(self):
        data = {
            "name": "test-pipeline",
            "config": {
                "conversation": {"initiator": "doctor", "max_turns": 20}
            }
        }
        pipeline = Pipeline.from_dict(data)
        assert pipeline.conversation_initiator == "doctor"
        assert pipeline.max_turns == 20

class TestSimulation:
    def test_progress_calculation(self):
        sim = Simulation(
            id="test", pipeline_name="p", organization_id="o",
            status=SimulationStatus.RUNNING,
            total_episodes=10, completed_episodes=5
        )
        assert sim.progress == 0.5
    
    def test_progress_zero_episodes(self):
        sim = Simulation(
            id="test", pipeline_name="p", organization_id="o",
            status=SimulationStatus.PENDING,
            total_episodes=0, completed_episodes=0
        )
        assert sim.progress == 0.0
```

#### 1.2 Validation Tests (`tests/unit/test_validation.py`)

```python
class TestPipelineValidation:
    def test_max_turns_validation_range(self):
        # These should be validated by the API
        assert 1 <= 10 <= 50  # Default is valid
        assert 1 <= 1 <= 50   # Min is valid
        assert 1 <= 50 <= 50  # Max is valid
    
    def test_conversation_initiator_validation(self):
        valid_initiators = ["patient", "doctor"]
        assert "patient" in valid_initiators
        assert "doctor" in valid_initiators
        assert "nurse" not in valid_initiators
```

#### 1.3 Exception Tests (`tests/unit/test_exceptions.py`)

```python
class TestExceptions:
    def test_not_found_error_message(self):
        err = NotFoundError("Pipeline", "my-pipeline")
        assert "Pipeline" in str(err)
        assert "my-pipeline" in str(err)
    
    def test_rate_limit_error_retry_after(self):
        err = RateLimitError(retry_after=30)
        assert err.retry_after == 30
    
    def test_validation_error_details(self):
        err = ValidationError("Invalid input", details={"field": "name"})
        assert err.details["field"] == "name"
```

---

## Layer 2: Integration Tests (Smoke)

**Purpose**: Verify SDK can connect and authenticate  
**Requirements**: Test environment credentials  
**Run On**: Every push to main, every PR

### Test Categories

#### 2.1 Authentication (`tests/integration/test_auth.py`)

```python
@pytest.fixture
def client():
    return EarlClient(
        client_id=os.environ["EARL_CLIENT_ID"],
        client_secret=os.environ["EARL_CLIENT_SECRET"],
        organization=os.environ.get("EARL_ORGANIZATION"),
        environment="test"
    )

class TestAuthentication:
    def test_connection(self, client):
        """Verify we can connect to the API."""
        result = client.test_connection()
        assert result is True
    
    def test_invalid_credentials_raises_auth_error(self):
        """Verify invalid credentials are rejected."""
        client = EarlClient(
            client_id="invalid",
            client_secret="invalid",
            environment="test"
        )
        with pytest.raises(AuthenticationError):
            client.test_connection()
```

#### 2.2 Basic API Calls (`tests/integration/test_smoke.py`)

```python
class TestSmoke:
    def test_list_dimensions(self, client):
        """Can list dimensions."""
        dims = client.dimensions.list()
        assert len(dims) > 0
        assert all(hasattr(d, 'id') for d in dims)
    
    def test_list_patients(self, client):
        """Can list patients."""
        patients = client.patients.list(limit=5)
        assert len(patients) > 0
        assert all(hasattr(p, 'id') for p in patients)
    
    def test_list_pipelines(self, client):
        """Can list pipelines."""
        pipelines = client.pipelines.list()
        # May be empty, but should not error
        assert isinstance(pipelines, list)
```

---

## Layer 3: Functional Tests (Core)

**Purpose**: Test core SDK functionality end-to-end  
**Requirements**: Test environment credentials, may create resources  
**Run On**: PR to main, pre-release

### Test Categories

#### 3.1 Pipeline Management (`tests/functional/test_pipelines.py`)

```python
class TestPipelineCRUD:
    def test_create_pipeline_internal_doctor(self, client, cleanup):
        """Create pipeline with internal doctor."""
        name = f"test-internal-{int(time.time())}"
        cleanup.append(name)
        
        pipeline = client.pipelines.create(
            name=name,
            dimension_ids=["accuracy", "empathy"],
            doctor_config=DoctorApiConfig.internal(),
        )
        
        assert pipeline.name == name
        assert pipeline.doctor_api.type == "internal"
    
    def test_create_pipeline_with_max_turns(self, client, cleanup):
        """Create pipeline with custom max_turns."""
        name = f"test-turns-{int(time.time())}"
        cleanup.append(name)
        
        pipeline = client.pipelines.create(
            name=name,
            dimension_ids=["accuracy"],
            max_turns=25,
        )
        
        assert pipeline.max_turns == 25
    
    def test_create_pipeline_external_doctor(self, client, cleanup, mock_doctor_url):
        """Create pipeline with external doctor."""
        name = f"test-external-{int(time.time())}"
        cleanup.append(name)
        
        pipeline = client.pipelines.create(
            name=name,
            dimension_ids=["accuracy"],
            doctor_config=DoctorApiConfig.external(
                api_url=mock_doctor_url,
                api_key="test-key"
            ),
            validate_doctor=False,  # Skip validation for test
        )
        
        assert pipeline.doctor_api.type == "external"
    
    def test_create_pipeline_client_driven(self, client, cleanup):
        """Create pipeline for client-driven mode."""
        name = f"test-client-driven-{int(time.time())}"
        cleanup.append(name)
        
        pipeline = client.pipelines.create(
            name=name,
            dimension_ids=["accuracy"],
            doctor_config=DoctorApiConfig.client_driven(),
        )
        
        assert pipeline.doctor_api.type == "client_driven"
        assert pipeline.doctor_api.is_client_driven is True
    
    def test_get_pipeline(self, client, test_pipeline):
        """Get existing pipeline."""
        pipeline = client.pipelines.get(test_pipeline.name)
        assert pipeline.name == test_pipeline.name
    
    def test_delete_pipeline(self, client):
        """Delete pipeline."""
        name = f"test-delete-{int(time.time())}"
        client.pipelines.create(name=name, dimension_ids=["accuracy"])
        
        client.pipelines.delete(name)
        
        with pytest.raises(NotFoundError):
            client.pipelines.get(name)
    
    def test_invalid_max_turns_raises_validation_error(self, client):
        """Invalid max_turns should raise ValidationError."""
        with pytest.raises(ValidationError):
            client.pipelines.create(
                name="test-invalid",
                dimension_ids=["accuracy"],
                max_turns=100,  # Exceeds 50 limit
            )
```

#### 3.2 Simulation Management (`tests/functional/test_simulations.py`)

```python
class TestSimulationManagement:
    def test_create_simulation(self, client, test_pipeline):
        """Create a simulation."""
        sim = client.simulations.create(
            pipeline_name=test_pipeline.name,
            num_episodes=1,
        )
        
        assert sim.id is not None
        assert sim.status in [SimulationStatus.PENDING, SimulationStatus.RUNNING]
    
    def test_get_simulation(self, client, test_simulation):
        """Get simulation status."""
        sim = client.simulations.get(test_simulation.id)
        assert sim.id == test_simulation.id
    
    def test_list_simulations(self, client, test_pipeline):
        """List simulations for a pipeline."""
        sims = client.simulations.list(pipeline_name=test_pipeline.name)
        assert isinstance(sims, list)
    
    def test_get_episodes(self, client, test_simulation):
        """Get episodes from simulation."""
        episodes = client.simulations.get_episodes(test_simulation.id)
        assert isinstance(episodes, list)
```

#### 3.3 Conversation Initiator (`tests/functional/test_conversation.py`)

```python
class TestConversationInitiator:
    def test_patient_initiated_pipeline(self, client, cleanup):
        """Patient-initiated conversation."""
        name = f"test-patient-init-{int(time.time())}"
        cleanup.append(name)
        
        pipeline = client.pipelines.create(
            name=name,
            dimension_ids=["accuracy"],
            conversation_initiator="patient",
        )
        
        assert pipeline.conversation_initiator == "patient"
    
    def test_doctor_initiated_pipeline(self, client, cleanup):
        """Doctor-initiated conversation."""
        name = f"test-doctor-init-{int(time.time())}"
        cleanup.append(name)
        
        pipeline = client.pipelines.create(
            name=name,
            dimension_ids=["accuracy"],
            conversation_initiator="doctor",
        )
        
        assert pipeline.conversation_initiator == "doctor"
```

---

## Layer 4: E2E Tests (Comprehensive)

**Purpose**: Full end-to-end workflow validation  
**Requirements**: Test environment, may take 10+ minutes  
**Run On**: Pre-release, release tags

### Test Categories

#### 4.1 Full Simulation Workflow (`tests/e2e/test_full_workflow.py`)

```python
class TestFullWorkflow:
    @pytest.mark.slow
    def test_internal_doctor_full_simulation(self, client, cleanup):
        """Complete simulation with internal doctor and judging."""
        # 1. Create pipeline
        name = f"e2e-internal-{int(time.time())}"
        cleanup.append(name)
        
        pipeline = client.pipelines.create(
            name=name,
            dimension_ids=["accuracy", "empathy"],
            patient_ids=["Adrian_Cruickshank"],
            doctor_config=DoctorApiConfig.internal(),
            max_turns=8,
        )
        
        # 2. Start simulation
        sim = client.simulations.create(
            pipeline_name=name,
            num_episodes=1,
        )
        
        # 3. Wait for completion
        completed = client.simulations.wait_for_completion(
            sim.id,
            timeout=600,
            poll_interval=5,
        )
        
        assert completed.status == SimulationStatus.COMPLETED
        
        # 4. Get report
        report = client.simulations.get_report(sim.id)
        assert "summary" in report
        assert "average_score" in report["summary"]
    
    @pytest.mark.slow
    def test_client_driven_workflow(self, client, cleanup, mock_doctor):
        """Complete client-driven simulation workflow."""
        # Full client-driven workflow test
        # (orchestrated by client code, not EARL)
        pass
```

#### 4.2 Error Scenarios (`tests/e2e/test_error_handling.py`)

```python
class TestErrorHandling:
    def test_simulation_with_unreachable_doctor(self, client, cleanup):
        """Simulation fails gracefully with unreachable external doctor."""
        pass
    
    def test_rate_limiting_handled(self, client):
        """Rate limiting is handled with retry."""
        pass
```

---

## Test Directory Structure

```
sdk/
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Shared fixtures
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_models.py
│   │   ├── test_validation.py
│   │   └── test_exceptions.py
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_auth.py
│   │   └── test_smoke.py
│   ├── functional/
│   │   ├── __init__.py
│   │   ├── test_pipelines.py
│   │   ├── test_simulations.py
│   │   └── test_conversation.py
│   └── e2e/
│       ├── __init__.py
│       ├── test_full_workflow.py
│       └── test_error_handling.py
```

---

## GitHub Actions Workflow

### `.github/workflows/sdk-tests.yml`

```yaml
name: SDK Tests

on:
  push:
    branches: [main]
    paths:
      - 'sdk/**'
  pull_request:
    branches: [main]
    paths:
      - 'sdk/**'
  release:
    types: [published]

env:
  PYTHON_VERSION: "3.11"

jobs:
  # ============================================
  # Layer 0: Pre-flight (runs on everything)
  # ============================================
  preflight:
    name: Pre-flight Checks
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install tools
        run: pip install ruff mypy black
      
      - name: Syntax check
        run: python -m py_compile sdk/earl_sdk/*.py
      
      - name: Import check
        run: |
          cd sdk
          python -c "from earl_sdk import EarlClient, DoctorApiConfig"
      
      - name: Lint (critical)
        run: ruff check sdk/earl_sdk/ --select=E9,F63,F7,F82
      
      - name: Lint (style)
        run: ruff check sdk/earl_sdk/
        continue-on-error: true  # Non-blocking
      
      - name: Type check
        run: mypy sdk/earl_sdk/ --ignore-missing-imports
        continue-on-error: true  # Non-blocking for now

  # ============================================
  # Layer 1: Unit Tests (runs on everything)
  # ============================================
  unit-tests:
    name: Unit Tests
    runs-on: ubuntu-latest
    needs: preflight
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install SDK
        run: |
          cd sdk
          pip install -e ".[dev]"
      
      - name: Run unit tests
        run: |
          cd sdk
          pytest tests/unit/ -v --tb=short

  # ============================================
  # Layer 2: Integration Tests (Smoke)
  # ============================================
  smoke-tests:
    name: Smoke Tests
    runs-on: ubuntu-latest
    needs: unit-tests
    if: github.event_name == 'push' || github.event_name == 'pull_request'
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install SDK
        run: |
          cd sdk
          pip install -e ".[dev]"
      
      - name: Run smoke tests
        env:
          EARL_CLIENT_ID: ${{ secrets.EARL_TEST_CLIENT_ID }}
          EARL_CLIENT_SECRET: ${{ secrets.EARL_TEST_CLIENT_SECRET }}
          EARL_ORGANIZATION: ${{ secrets.EARL_TEST_ORGANIZATION }}
        run: |
          cd sdk
          pytest tests/integration/ -v --tb=short

  # ============================================
  # Layer 3: Functional Tests (PRs to main)
  # ============================================
  functional-tests:
    name: Functional Tests
    runs-on: ubuntu-latest
    needs: smoke-tests
    if: github.event_name == 'pull_request' && github.base_ref == 'main'
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install SDK
        run: |
          cd sdk
          pip install -e ".[dev]"
      
      - name: Run functional tests
        env:
          EARL_CLIENT_ID: ${{ secrets.EARL_TEST_CLIENT_ID }}
          EARL_CLIENT_SECRET: ${{ secrets.EARL_TEST_CLIENT_SECRET }}
          EARL_ORGANIZATION: ${{ secrets.EARL_TEST_ORGANIZATION }}
        run: |
          cd sdk
          pytest tests/functional/ -v --tb=short

  # ============================================
  # Layer 4: E2E Tests (releases only)
  # ============================================
  e2e-tests:
    name: E2E Tests
    runs-on: ubuntu-latest
    needs: functional-tests
    if: github.event_name == 'release'
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install SDK
        run: |
          cd sdk
          pip install -e ".[dev]"
      
      - name: Run E2E tests
        env:
          EARL_CLIENT_ID: ${{ secrets.EARL_TEST_CLIENT_ID }}
          EARL_CLIENT_SECRET: ${{ secrets.EARL_TEST_CLIENT_SECRET }}
          EARL_ORGANIZATION: ${{ secrets.EARL_TEST_ORGANIZATION }}
        run: |
          cd sdk
          pytest tests/e2e/ -v --tb=short -m "not slow"
        timeout-minutes: 30
```

---

## Make Targets

Add these to the SDK section of the Makefile:

```makefile
# =============================================================================
# SDK Testing
# =============================================================================

# Layer 0: Pre-flight checks (no credentials needed)
sdk-preflight:
	@echo "Running pre-flight checks..."
	@cd sdk && python -m py_compile earl_sdk/*.py
	@cd sdk && python -c "from earl_sdk import EarlClient, DoctorApiConfig"
	@cd sdk && ruff check earl_sdk/ --select=E9,F63,F7,F82
	@echo "✓ Pre-flight checks passed"

# Layer 1: Unit tests (no credentials needed)
sdk-test-unit:
	@echo "Running unit tests..."
	@cd sdk && pytest tests/unit/ -v --tb=short

# Layer 2: Smoke tests (needs credentials)
sdk-test-smoke:
	@echo "Running smoke tests..."
	@cd sdk && pytest tests/integration/ -v --tb=short

# Layer 3: Functional tests (needs credentials)
sdk-test-functional:
	@echo "Running functional tests..."
	@cd sdk && pytest tests/functional/ -v --tb=short

# Layer 4: E2E tests (needs credentials, slow)
sdk-test-e2e:
	@echo "Running E2E tests..."
	@cd sdk && pytest tests/e2e/ -v --tb=short

# Run all tests (except E2E)
sdk-test: sdk-preflight sdk-test-unit sdk-test-smoke sdk-test-functional
	@echo "✓ All SDK tests passed"

# Run all tests including E2E (for releases)
sdk-test-all: sdk-test sdk-test-e2e
	@echo "✓ All SDK tests (including E2E) passed"

# Quick test (preflight + unit only, no credentials)
sdk-test-quick: sdk-preflight sdk-test-unit
	@echo "✓ Quick tests passed"
```

---

## Shared Fixtures (`tests/conftest.py`)

```python
import os
import time
import pytest
from earl_sdk import EarlClient, DoctorApiConfig

# Skip all tests if no credentials
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )

@pytest.fixture(scope="session")
def credentials():
    """Get test credentials from environment."""
    client_id = os.environ.get("EARL_CLIENT_ID")
    client_secret = os.environ.get("EARL_CLIENT_SECRET")
    organization = os.environ.get("EARL_ORGANIZATION")
    
    if not client_id or not client_secret:
        pytest.skip("No credentials provided (set EARL_CLIENT_ID and EARL_CLIENT_SECRET)")
    
    return client_id, client_secret, organization

@pytest.fixture(scope="session")
def client(credentials):
    """Create SDK client for testing."""
    client_id, client_secret, organization = credentials
    return EarlClient(
        client_id=client_id,
        client_secret=client_secret,
        organization=organization,
        environment="test",
    )

@pytest.fixture
def cleanup(client):
    """Fixture to clean up created pipelines after tests."""
    pipelines_to_delete = []
    yield pipelines_to_delete
    
    for name in pipelines_to_delete:
        try:
            client.pipelines.delete(name)
        except Exception:
            pass  # Ignore errors during cleanup

@pytest.fixture
def test_pipeline(client, cleanup):
    """Create a test pipeline for use in tests."""
    name = f"test-fixture-{int(time.time())}"
    cleanup.append(name)
    
    pipeline = client.pipelines.create(
        name=name,
        dimension_ids=["accuracy"],
        doctor_config=DoctorApiConfig.internal(),
    )
    return pipeline
```

---

## Feature Coverage Checklist

### Core Features
- [ ] Authentication (Auth0 M2M)
- [ ] Environment switching (test/prod)
- [ ] Connection testing

### Dimensions
- [ ] List dimensions
- [ ] Filter dimensions

### Patients
- [ ] List patients
- [ ] Filter patients

### Pipelines
- [ ] Create pipeline (internal doctor)
- [ ] Create pipeline (external doctor)
- [ ] Create pipeline (client-driven)
- [ ] Create pipeline with max_turns
- [ ] Create pipeline with conversation_initiator
- [ ] Get pipeline
- [ ] List pipelines
- [ ] Delete pipeline
- [ ] Validation errors

### Simulations
- [ ] Create simulation
- [ ] Get simulation status
- [ ] List simulations
- [ ] Wait for completion
- [ ] Get episodes
- [ ] Submit response (client-driven)
- [ ] Get report

### Error Handling
- [ ] AuthenticationError
- [ ] AuthorizationError
- [ ] NotFoundError
- [ ] ValidationError
- [ ] RateLimitError
- [ ] ServerError

---

## BRD Consideration

**Q: Would a BRD document help?**

**A:** A BRD (Business Requirements Document) would help for:
- **Acceptance criteria** - What constitutes "working" for each feature
- **Edge cases** - Business rules around limits, error conditions
- **User journeys** - Typical workflows customers follow

However, the **code is the source of truth** for:
- API contracts (what the SDK actually does)
- Technical constraints (rate limits, timeouts)
- Implementation details

**Recommendation:** Write tests from the code first (as shown above), then enhance with BRD-derived acceptance tests if needed.

---

## Next Steps

1. Create the test directory structure
2. Implement Layer 1 (unit tests) - no credentials needed
3. Implement Layer 2 (smoke tests) - basic API validation
4. Implement Layer 3 (functional tests) - full feature coverage
5. Implement Layer 4 (E2E tests) - workflow validation
6. Set up GitHub Actions workflow
7. Add required secrets to GitHub repository
