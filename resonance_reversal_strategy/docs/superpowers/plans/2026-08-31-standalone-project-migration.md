# Resonance Reversal Standalone Project Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `D:\test\resonance_reversal_strategy` as a standalone Git repository containing only the frozen resonance strategy baseline, its research tools, documentation, and dedicated tests.

**Architecture:** Export the immutable strategy/test content from Git commit `5e81c07`, copy the approved migration design from its final source commit, then add four project-boundary files. Verify file-set identity and SHA-256 equality before compiling, running only resonance tests, and creating one atomic baseline commit in the new repository.

**Tech Stack:** Git, PowerShell, Python 3, pytest, .NET `System.IO.Compression` and SHA-256 APIs.

**Spec:** `resonance_reversal_strategy/docs/superpowers/specs/2026-08-31-standalone-project-migration-design.md`

## Global Constraints

- Source repository: `D:\test\select_stocks`.
- Source worktree: `D:\test\select_stocks\.worktrees\resonance-no-atr-exit`.
- Strategy/test content baseline: Git commit `5e81c07`.
- Approved migration-design source commit: Git commit `f4053de`.
- Destination: `D:\test\resonance_reversal_strategy`.
- Destination must not already exist; never overwrite, merge, or delete it.
- Do not modify strategy logic, parameters, analysis logic, tests, or frozen existing documentation.
- Do not read, copy, run, or validate any non-resonance strategy or test.
- Do not copy logs, reports, attachments, caches, market data, or local artifacts.
- Preserve the T-1 signal boundary and all training/validation boundaries by preserving source bytes exactly.
- The source worktree must remain clean throughout migration.
- The standalone project receives one atomic baseline commit only after all checks pass.

---

### Task 1: Preflight and Export the Frozen Content Baseline

**Files:**
- Read: source Git metadata only.
- Create temporarily: `C:\Users\C1-CWadmin\AppData\Local\Temp\resonance-standalone-baseline-5e81c07.zip`
- Create: `D:\test\resonance_reversal_strategy\resonance_reversal_strategy\**`
- Create: `D:\test\resonance_reversal_strategy\tests\test_resonance*.py`
- Create: `D:\test\resonance_reversal_strategy\resonance_reversal_strategy\docs\superpowers\specs\2026-08-31-standalone-project-migration-design.md`

**Interfaces:**
- Consumes: Git tree at `5e81c07` and design document at source commit `f4053de`.
- Produces: an unmodified destination snapshot plus the retained baseline ZIP used by Task 3.

- [ ] **Step 1: Verify the exact source and destination preconditions**

Run in `D:\test\select_stocks\.worktrees\resonance-no-atr-exit`:

```powershell
$sourceRoot = 'D:\test\select_stocks\.worktrees\resonance-no-atr-exit'
$destinationRoot = 'D:\test\resonance_reversal_strategy'
$archivePath = 'C:\Users\C1-CWadmin\AppData\Local\Temp\resonance-standalone-baseline-5e81c07.zip'

if ((git branch --show-current) -ne 'codex/resonance-no-atr-exit') {
    throw 'unexpected source branch'
}
git diff --quiet
if ($LASTEXITCODE -ne 0) { throw 'source worktree has unstaged changes' }
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) { throw 'source worktree has staged changes' }
git merge-base --is-ancestor 5e81c07 HEAD
if ($LASTEXITCODE -ne 0) { throw 'content baseline is not an ancestor of HEAD' }
git cat-file -e 'f4053de^{commit}' 2>$null
if ($LASTEXITCODE -ne 0) { throw 'approved design commit is missing' }
if (Test-Path -LiteralPath $destinationRoot) {
    throw 'destination already exists'
}
if (Test-Path -LiteralPath $archivePath) {
    throw 'fixed migration archive path already exists'
}
'PREFLIGHT=PASS'
```

Expected: `PREFLIGHT=PASS`. Stop on any exception; do not repair, overwrite, or delete the conflicting path.

- [ ] **Step 2: Export only the frozen resonance file set**

Run in the source worktree:

```powershell
$archivePath = 'C:\Users\C1-CWadmin\AppData\Local\Temp\resonance-standalone-baseline-5e81c07.zip'
git archive --format=zip --output=$archivePath 5e81c07 -- `
    resonance_reversal_strategy `
    tests/test_resonance_reversal_strategy.py `
    tests/test_resonance_relative_turn_analysis.py `
    tests/test_resonance_trade_risk_analysis.py
if ($LASTEXITCODE -ne 0) { throw 'git archive failed' }
if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
    throw 'baseline archive was not created'
}
```

Expected: the ZIP exists and contains only paths selected from commit `5e81c07`.

- [ ] **Step 3: Extract the baseline into the previously absent destination**

```powershell
$archivePath = 'C:\Users\C1-CWadmin\AppData\Local\Temp\resonance-standalone-baseline-5e81c07.zip'
$destinationRoot = 'D:\test\resonance_reversal_strategy'
Expand-Archive -LiteralPath $archivePath -DestinationPath $destinationRoot
if (-not (Test-Path -LiteralPath $destinationRoot -PathType Container)) {
    throw 'destination was not created'
}
```

Expected: destination exists with one `resonance_reversal_strategy` directory and the three dedicated tests.

- [ ] **Step 4: Copy the approved migration design without modifying it**

```powershell
$sourceDesign = 'D:\test\select_stocks\.worktrees\resonance-no-atr-exit\resonance_reversal_strategy\docs\superpowers\specs\2026-08-31-standalone-project-migration-design.md'
$targetDesign = 'D:\test\resonance_reversal_strategy\resonance_reversal_strategy\docs\superpowers\specs\2026-08-31-standalone-project-migration-design.md'
Copy-Item -LiteralPath $sourceDesign -Destination $targetDesign
if (-not (Test-Path -LiteralPath $targetDesign -PathType Leaf)) {
    throw 'migration design was not copied'
}
```

Expected: target design exists. Do not copy this implementation plan; provenance commands are recorded later in `MIGRATION_SOURCE.md`.

---

### Task 2: Establish the Standalone Project Boundary

**Files:**
- Create: `D:\test\resonance_reversal_strategy\AGENTS.md`
- Create: `D:\test\resonance_reversal_strategy\pytest.ini`
- Create: `D:\test\resonance_reversal_strategy\.gitignore`

**Interfaces:**
- Consumes: destination snapshot from Task 1.
- Produces: repository-local operating rules, test discovery restrictions, and artifact exclusions used by Tasks 3-5.

- [ ] **Step 1: Create the dedicated `AGENTS.md`**

Use `apply_patch` to create exactly:

```markdown
# Resonance Reversal Project Rules

## Scope

- This repository contains only the independent `resonance_reversal_strategy`.
- Only the strategy, its `research` tools, documentation, and `test_resonance*.py` tests are in scope.
- Do not import, read, copy, modify, run, or validate any other strategy.

## Working Process

- Before every code or behavior change, analyze the relevant context, present a Chinese implementation plan and impact boundary, and wait for explicit user approval.
- Apply the minimum-change principle. Do not alter unrelated files, formatting, parameters, reports, or behavior.
- Use test-driven development for implementation changes and run only the dedicated resonance tests.
- Complete each verified milestone with a summary and a Git commit.

## Data and Research Boundaries

- Daily signals may use only T-1 and earlier data. T-day execution data must not revise a frozen signal, rank, size, or sell decision.
- The 2019-2021 training period may use pre-2019 data only as read-only warm-up for rolling indicators; warm-up data must not enter returns, tuning, or rule selection.
- Do not use validation-period or full-period results to tune parameters, thresholds, indicators, or rules.
- JoinQuant backtests remain authoritative for strategy performance. Local research is read-only diagnostic work unless a separately approved design states otherwise.

## Protected Behavior

- Preserve the current ATR observation-only policy unless a separately approved strategy change explicitly replaces it.
- Preserve training manifests, log identity checks, source immutability, friction comparison, and fail-closed evidence validation.
- Research reports must never silently become trading rules or candidates.
```

- [ ] **Step 2: Restrict default pytest discovery**

Use `apply_patch` to create `pytest.ini` exactly:

```ini
[pytest]
testpaths = tests
python_files = test_resonance*.py
```

- [ ] **Step 3: Exclude local artifacts from Git**

Use `apply_patch` to create `.gitignore` exactly:

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
venv/
artifacts/
reports/
logs/
*.tmp
*.log
.idea/
.vscode/
```

- [ ] **Step 4: Verify only the three intended project-boundary files were added**

```powershell
$destinationRoot = 'D:\test\resonance_reversal_strategy'
$boundaryFiles = @('.gitignore', 'AGENTS.md', 'pytest.ini')
$actualBoundaryFiles = @(
    Get-ChildItem -LiteralPath $destinationRoot -File |
        Select-Object -ExpandProperty Name |
        Sort-Object
)
if (Compare-Object $boundaryFiles $actualBoundaryFiles) {
    throw 'unexpected root-level file before provenance creation'
}
'BOUNDARY_FILES=PASS'
```

Expected: `BOUNDARY_FILES=PASS`.

---

### Task 3: Prove Byte Identity and Scope Isolation

**Files:**
- Read: baseline ZIP from Task 1.
- Read: all destination files created by Tasks 1-2.
- No persistent file changes.

**Interfaces:**
- Consumes: baseline archive and destination tree.
- Produces: `FILE_SET_ASSERTIONS=PASS`, `BASELINE_HASH_ASSERTIONS=PASS`, `DESIGN_HASH_ASSERTION=PASS`, and `PYTHON_SCOPE_ASSERTION=PASS` evidence required by Task 5.

- [ ] **Step 1: Compare the complete baseline file set with the destination**

Run from any PowerShell directory:

```powershell
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archivePath = 'C:\Users\C1-CWadmin\AppData\Local\Temp\resonance-standalone-baseline-5e81c07.zip'
$destinationRoot = 'D:\test\resonance_reversal_strategy'
$designRelative = 'resonance_reversal_strategy/docs/superpowers/specs/2026-08-31-standalone-project-migration-design.md'
$allowedNew = @('.gitignore', 'AGENTS.md', 'pytest.ini', $designRelative)
$zipFile = [System.IO.Compression.ZipFile]::OpenRead($archivePath)
try {
    $archiveFiles = @(
        $zipFile.Entries |
            Where-Object { $_.Name -ne '' } |
            ForEach-Object { $_.FullName.Replace('\', '/') } |
            Sort-Object
    )
} finally {
    $zipFile.Dispose()
}
$destinationBaselineFiles = @(
    Get-ChildItem -LiteralPath $destinationRoot -Recurse -File |
        ForEach-Object {
            $_.FullName.Substring($destinationRoot.Length + 1).Replace('\', '/')
        } |
        Where-Object { $allowedNew -notcontains $_ } |
        Sort-Object
)
$fileSetDifference = @(Compare-Object $archiveFiles $destinationBaselineFiles)
if ($fileSetDifference.Count -ne 0) {
    $fileSetDifference | Format-Table | Out-String | Write-Host
    throw 'baseline file sets differ'
}
'FILE_SET_ASSERTIONS=PASS'
```

- [ ] **Step 2: Compare SHA-256 for every archived file**

```powershell
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archivePath = 'C:\Users\C1-CWadmin\AppData\Local\Temp\resonance-standalone-baseline-5e81c07.zip'
$destinationRoot = 'D:\test\resonance_reversal_strategy'
$zipFile = [System.IO.Compression.ZipFile]::OpenRead($archivePath)
$sha256 = [System.Security.Cryptography.SHA256]::Create()
try {
    foreach ($entry in @($zipFile.Entries | Where-Object { $_.Name -ne '' })) {
        $entryStream = $entry.Open()
        try {
            $sourceHash = [BitConverter]::ToString(
                $sha256.ComputeHash($entryStream)
            ).Replace('-', '').ToLowerInvariant()
        } finally {
            $entryStream.Dispose()
        }
        $targetPath = Join-Path $destinationRoot $entry.FullName
        $targetHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $targetPath).Hash.ToLowerInvariant()
        if ($sourceHash -ne $targetHash) {
            throw "baseline hash mismatch: $($entry.FullName)"
        }
        $sha256.Initialize()
    }
} finally {
    $sha256.Dispose()
    $zipFile.Dispose()
}
'BASELINE_HASH_ASSERTIONS=PASS'
```

- [ ] **Step 3: Compare the separately copied design document hash**

```powershell
$sourceDesign = 'D:\test\select_stocks\.worktrees\resonance-no-atr-exit\resonance_reversal_strategy\docs\superpowers\specs\2026-08-31-standalone-project-migration-design.md'
$targetDesign = 'D:\test\resonance_reversal_strategy\resonance_reversal_strategy\docs\superpowers\specs\2026-08-31-standalone-project-migration-design.md'
$sourceDesignHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $sourceDesign).Hash
$targetDesignHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $targetDesign).Hash
if ($sourceDesignHash -ne $targetDesignHash) {
    throw 'migration design hash mismatch'
}
'DESIGN_HASH_ASSERTION=PASS'
```

- [ ] **Step 4: Prove that every Python file belongs to resonance**

```powershell
$destinationRoot = 'D:\test\resonance_reversal_strategy'
$expectedPythonFiles = @(
    'resonance_reversal_strategy/research/analyze_relative_turn_observations.py',
    'resonance_reversal_strategy/research/analyze_resonance_trade_risk.py',
    'resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py',
    'tests/test_resonance_relative_turn_analysis.py',
    'tests/test_resonance_reversal_strategy.py',
    'tests/test_resonance_trade_risk_analysis.py'
) | Sort-Object
$actualPythonFiles = @(
    Get-ChildItem -LiteralPath $destinationRoot -Recurse -File -Filter '*.py' |
        ForEach-Object {
            $_.FullName.Substring($destinationRoot.Length + 1).Replace('\', '/')
        } |
        Sort-Object
)
if (Compare-Object $expectedPythonFiles $actualPythonFiles) {
    throw 'unexpected Python file in standalone project'
}
'PYTHON_SCOPE_ASSERTION=PASS'
```

---

### Task 4: Compile and Run Only the Resonance Verification Suite

**Files:**
- Read/execute: three production Python files.
- Read/execute: three `test_resonance*.py` files.
- Create only ignored cache files.

**Interfaces:**
- Consumes: isolated destination from Tasks 1-3 and `pytest.ini` from Task 2.
- Produces: fresh compilation evidence and the fixed runtime baseline `502 passed, 3 skipped`.

- [ ] **Step 1: Compile the strategy and both research analyzers**

Run in `D:\test\resonance_reversal_strategy`:

```powershell
python -m py_compile `
    resonance_reversal_strategy/smart_trade_joinquant_resonance_reversal_etf.py `
    resonance_reversal_strategy/research/analyze_relative_turn_observations.py `
    resonance_reversal_strategy/research/analyze_resonance_trade_risk.py
if ($LASTEXITCODE -ne 0) { throw 'resonance compilation failed' }
'COMPILE=PASS'
```

- [ ] **Step 2: Verify pytest discovers only the dedicated test filenames**

```powershell
$collected = pytest --collect-only -q
if ($LASTEXITCODE -ne 0) { throw 'pytest collection failed' }
$allowedTests = @(
    'test_resonance_relative_turn_analysis.py',
    'test_resonance_reversal_strategy.py',
    'test_resonance_trade_risk_analysis.py'
)
$reportedTestFiles = @(
    $collected |
        Select-String -Pattern 'tests[\\/](test_[^:]+\.py)' -AllMatches |
        ForEach-Object { $_.Matches } |
        ForEach-Object { $_.Groups[1].Value } |
        Sort-Object -Unique
)
if (Compare-Object ($allowedTests | Sort-Object) $reportedTestFiles) {
    throw 'pytest collected a non-resonance test file'
}
'PYTEST_COLLECTION_SCOPE=PASS'
```

- [ ] **Step 3: Run the complete standalone test suite**

```powershell
pytest -q
if ($LASTEXITCODE -ne 0) { throw 'standalone resonance tests failed' }
```

Expected terminal summary: `502 passed, 3 skipped`. A different count or any failure stops migration before Git initialization.

- [ ] **Step 4: Reconfirm the source worktree remained unchanged**

Run in the source worktree:

```powershell
git diff --quiet
if ($LASTEXITCODE -ne 0) { throw 'source worktree changed during migration' }
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) { throw 'source index changed during migration' }
'SOURCE_UNCHANGED=PASS'
```

---

### Task 5: Record Provenance and Create the Atomic Git Baseline

**Files:**
- Create: `D:\test\resonance_reversal_strategy\MIGRATION_SOURCE.md`
- Create internally: `D:\test\resonance_reversal_strategy\.git\**`
- Remove after verification: `C:\Users\C1-CWadmin\AppData\Local\Temp\resonance-standalone-baseline-5e81c07.zip`

**Interfaces:**
- Consumes: all PASS evidence from Tasks 3-4.
- Produces: a clean standalone Git repository on branch `main` with one baseline commit.

- [ ] **Step 1: Create the immutable migration provenance record**

Use `apply_patch` to create `MIGRATION_SOURCE.md` exactly:

```markdown
# Migration Source

- Migration date: 2026-08-31
- Source repository: `D:\test\select_stocks`
- Source worktree: `D:\test\select_stocks\.worktrees\resonance-no-atr-exit`
- Source branch: `codex/resonance-no-atr-exit`
- Strategy/test content baseline: `5e81c07`
- Approved migration-design source commit: `f4053de`
- Migration method: Git archive of the frozen content baseline plus byte-identical copy of the approved migration design.

## Verification

- Baseline file-set comparison: PASS
- Baseline SHA-256 comparison: PASS
- Migration-design SHA-256 comparison: PASS
- Python scope assertion: PASS
- Python compilation: PASS
- Pytest discovery scope: PASS
- Dedicated resonance tests: `502 passed, 3 skipped`
- Source worktree unchanged: PASS

No JoinQuant backtest was run during migration. The migrated strategy and research content remains byte-identical to the recorded sources.
```

- [ ] **Step 2: Initialize the independent repository and inspect the complete baseline**

Run in `D:\test\resonance_reversal_strategy`:

```powershell
git init
if ($LASTEXITCODE -ne 0) { throw 'git init failed' }
git branch -M main
git add -- .
$stagedFiles = @(git diff --cached --name-only)
if ($stagedFiles.Count -eq 0) { throw 'standalone baseline is empty' }
if ($stagedFiles | Where-Object { $_ -match '(^|/)(artifacts|reports|logs)/' }) {
    throw 'local artifact was staged'
}
git status --short
```

Expected: only the standalone project files are staged; caches and local artifacts are ignored.

- [ ] **Step 3: Commit the standalone baseline**

```powershell
git commit -m "chore: initialize standalone resonance project"
if ($LASTEXITCODE -ne 0) { throw 'standalone baseline commit failed' }
git status --short
git log -1 --oneline --decorate
```

Expected: one commit on `main` and empty `git status --short` output.

- [ ] **Step 4: Remove only the verified temporary archive**

```powershell
$archivePath = [System.IO.Path]::GetFullPath(
    'C:\Users\C1-CWadmin\AppData\Local\Temp\resonance-standalone-baseline-5e81c07.zip'
)
$expectedArchive = 'C:\Users\C1-CWadmin\AppData\Local\Temp\resonance-standalone-baseline-5e81c07.zip'
if ($archivePath -ne $expectedArchive) {
    throw 'temporary archive path resolution changed'
}
if (Test-Path -LiteralPath $archivePath -PathType Leaf) {
    Remove-Item -LiteralPath $archivePath
}
if (Test-Path -LiteralPath $archivePath) {
    throw 'temporary archive cleanup failed'
}
'TEMP_ARCHIVE_CLEANUP=PASS'
```

- [ ] **Step 5: Perform final source and destination checks**

```powershell
git -C 'D:\test\resonance_reversal_strategy' status --short
git -C 'D:\test\resonance_reversal_strategy' log -1 --oneline --decorate
git -C 'D:\test\select_stocks\.worktrees\resonance-no-atr-exit' status --short
```

Expected: both status outputs are empty. Record the new repository's baseline commit hash in the completion report.
