"""Optional OpenMolcas bridge for CASSCF/CASPT2/RASSI cluster workflows.

OpenMolcas/Molcas is kept outside Atomi's dependency set.  This bridge writes
reviewable inputs and JUSTUS-style run wrappers for small molecules or embedded
clusters, and collects compact summaries from OpenMolcas output.  It is a
workflow bridge, not an automatic active-space selector.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


DEFAULT_OPENMOLCAS_MODULE = "chem/openmolcas/24.02"
SCHEMA_PROJECT = "atomi.openmolcas_bridge_project.v1"
SCHEMA_SUMMARY = "atomi.openmolcas_summary.v1"


@dataclass(frozen=True)
class GasSpace:
    """One GASSCF space and its cumulative electron bounds."""

    label: str
    orbitals: str
    min_electrons: int
    max_electrons: int


@dataclass(frozen=True)
class RasscfStateBlock:
    """One explicitly ordered RAS/RASSCF state for a CASPT2/RASSI handoff."""

    title: str
    symmetry: int
    spin: int
    nactel: str
    inactive: str
    ras1: str = ""
    ras2: str = ""
    ras3: str = ""
    ciroots: str = "1 1 1"
    iterations: str = "200 100"
    levs: str = "2.0"
    cionly: bool = False
    extra_lines: tuple[str, ...] = ()
    include_in_rassi: bool = True
    run_caspt2: bool = True


@dataclass(frozen=True)
class OpenMolcasPrepareOptions:
    title: str
    xyz_name: str
    charge: int
    spin: int = 1
    symmetry: int = 1
    basis: str = "ANO-RCC-VDZP"
    group: str = "XYZ"
    recipe: str = "casscf-caspt2-rassi"
    nactel: str = "2 1 1"
    inactive: str = "26 25"
    ras1: str = "1 0"
    ras2: str = ""
    ras3: str = "0 3"
    ciroots: str = "1 1 1"
    iterations: str = "200 100"
    levs: str = "2.0"
    frozen: str = ""
    ipea: str = "0"
    imag: str = "5.0"
    threshold: str = "1.0E-09 1.0E-07"
    use_caspt2: bool = True
    xmultistate: bool = True
    include_orbital_prep: bool = True
    include_partner: bool = True
    partner_spin: int = 3
    partner_ciroots: str = "3 3 1"
    sonorb: str = ""
    include_bssh: bool = True
    include_amfi: bool = True
    core_hole_note: str = ""
    extra_rasscf_lines: tuple[str, ...] = ()
    gas_spaces: tuple[GasSpace, ...] = ()
    gas_selector_dimension: int = 10
    gas_selector_root: int | None = None
    gas_prwf: str = "1.0E-03"
    state_blocks: tuple[RasscfStateBlock, ...] = ()


def _json_dump(payload: dict[str, Any], path: Path | None = None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(text, end="")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _shell_quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def default_molcas_executable() -> str:
    return os.environ.get("ATOMI_MOLCAS_EXE") or shutil.which("pymolcas") or "pymolcas"


def default_molcas_module() -> str:
    return os.environ.get("ATOMI_MOLCAS_MODULE") or DEFAULT_OPENMOLCAS_MODULE


def _resolve_executable(executable: str) -> str | None:
    if not executable:
        return None
    expanded = Path(executable).expanduser()
    if expanded.is_file():
        return str(expanded)
    found = shutil.which(executable)
    return found or None


def _probe_executable_via_module(module_name: str, executable: str) -> dict[str, Any]:
    if not module_name:
        return {}
    shell = (
        "source /etc/profile.d/modules.sh 2>/dev/null || true; "
        f"module load {shlex.quote(module_name)} >/dev/null 2>&1; "
        "module_status=$?; "
        f"resolved=$(command -v {shlex.quote(executable)} 2>/dev/null || true); "
        'printf "module_status=%s\\nresolved_executable=%s\\nMOLCAS_VERSION=%s\\nMOLCAS=%s\\n" '
        '"$module_status" "$resolved" "${MOLCAS_VERSION:-}" "${MOLCAS:-}"'
    )
    try:
        proc = subprocess.run(["bash", "-lc", shell], capture_output=True, text=True, timeout=30)
    except Exception as exc:  # pragma: no cover - shell/module specific
        return {"module_probe_error": str(exc)}
    parsed: dict[str, Any] = {"module_probe_returncode": proc.returncode}
    if proc.stderr:
        parsed["module_probe_stderr_head"] = "\n".join(proc.stderr.splitlines()[:8])
    for line in proc.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            parsed[f"module_probe_{key}"] = value
    return parsed


def probe_molcas(executable: str | None = None, module: str | None = None) -> dict[str, Any]:
    exe = executable or default_molcas_executable()
    module_name = module if module is not None else os.environ.get("ATOMI_MOLCAS_MODULE", "")
    resolved = _resolve_executable(exe)
    module_probe = _probe_executable_via_module(module_name, exe) if not resolved and module_name else {}
    module_resolved = str(module_probe.get("module_probe_resolved_executable") or "")
    if not resolved and module_resolved:
        resolved = module_resolved
    payload: dict[str, Any] = {
        "executable": exe,
        "resolved_executable": resolved or "",
        "available": bool(resolved),
        "module": module_name,
        "molcas": os.environ.get("MOLCAS") or str(module_probe.get("module_probe_MOLCAS") or ""),
        "molcas_version": os.environ.get("MOLCAS_VERSION") or str(module_probe.get("module_probe_MOLCAS_VERSION") or ""),
        "molcas_workdir": os.environ.get("MOLCAS_WORKDIR", ""),
        **module_probe,
    }
    if resolved and not module_resolved:
        for flag in ("--version", "-h"):
            try:
                proc = subprocess.run([resolved, flag], capture_output=True, text=True, timeout=15)
            except Exception as exc:  # pragma: no cover - executable-specific behavior
                payload[f"probe_{flag}_error"] = str(exc)
                continue
            payload[f"probe_{flag}_returncode"] = proc.returncode
            head = "\n".join((proc.stdout or proc.stderr).splitlines()[:8])
            if head:
                payload[f"probe_{flag}_head"] = head
            if proc.returncode == 0:
                break
    return payload


def status_main(args: argparse.Namespace) -> dict[str, Any]:
    report = {
        "schema": "atomi.openmolcas_status.v1",
        "openmolcas": probe_molcas(args.executable, args.module),
        "environment": {
            "ATOMI_MOLCAS_EXE": os.environ.get("ATOMI_MOLCAS_EXE", ""),
            "ATOMI_MOLCAS_MODULE": os.environ.get("ATOMI_MOLCAS_MODULE", ""),
            "ATOMI_MOLCAS_ROOT": os.environ.get("ATOMI_MOLCAS_ROOT", ""),
            "MOLCAS": os.environ.get("MOLCAS", ""),
            "MOLCAS_VERSION": os.environ.get("MOLCAS_VERSION", ""),
            "MOLCAS_WORKDIR": os.environ.get("MOLCAS_WORKDIR", ""),
        },
    }
    if args.json:
        _json_dump(report)
    else:
        molcas = report["openmolcas"]
        print("Atomi OpenMolcas bridge status")
        print(f"  executable : {molcas.get('resolved_executable') or molcas.get('executable')}")
        print(f"  available  : {'yes' if molcas.get('available') else 'no'}")
        print(f"  module     : {molcas.get('module') or '(not set)'}")
        print(f"  MOLCAS     : {molcas.get('molcas') or '(not set)'}")
        print(f"  version    : {molcas.get('molcas_version') or '(not set)'}")
    return report


def install_plan_main(args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "schema": "atomi.openmolcas_install_plan.v1",
        "recommendation": "Keep OpenMolcas as an external HPC module/runtime and let Atomi write/review/run bridge workspaces.",
        "why": [
            "CASSCF/RASSCF active-space selection is scientific input, not something Atomi should infer silently.",
            "CASPT2 and RASSI depend on JOBIPH/JOBMIX handoffs from previous RASSCF/CASPT2 stages.",
            "Actinide M4,5-edge cluster spectra require separate full-core and core-hole states, spin-orbit RASSI, and careful basis/core treatment.",
        ],
        "bridge_roles": {
            "OpenMolcas": "Embedded clusters, explicit CASSCF/RASSCF/CASPT2/RASSI, spin-orbit state interaction, and core-hole scaffolds.",
            "OCEAN": "Periodic-solid XANES/BSE route from VASP/DFT+U ground states.",
            "FEFF/Larch": "Local-cluster EXAFS/XAFS ensemble comparisons from MD/DFT structures.",
        },
        "hpc_pattern": [
            "Use the JUSTUS2 OpenMolcas module, e.g. chem/openmolcas/24.02 or a validated newer module.",
            "Keep Atomi in m_lammps_env; do not install OpenMolcas inside m_lammps_env.",
            "Record module/executable choices in private KIT JSON under profiles.molcas.",
            "Apply with: eval \"$(confighpc --config ~/atomi_hpc/atomi_hpc_config.kit.local.json --shell)\"",
            "Check with: molcas-status",
        ],
        "example_profile": {
            "profiles": {
                "molcas": {
                    "module": DEFAULT_OPENMOLCAS_MODULE,
                    "executable": "pymolcas",
                    "scratch_root": "$SCRATCH",
                    "environment": {
                        "ATOMI_MOLCAS_MODULE": DEFAULT_OPENMOLCAS_MODULE,
                        "ATOMI_MOLCAS_EXE": "pymolcas",
                        "ATOMI_MOLCAS_ROOT": "",
                    },
                }
            }
        },
    }
    if args.json:
        _json_dump(payload)
    else:
        print("OpenMolcas / Atomi HPC install plan")
        for item in payload["why"]:
            print(f"  - {item}")
        print("  Recommended: external OpenMolcas module/runtime, configured through private KIT JSON.")
    return payload


def _value_lines(keyword: str, value: str) -> list[str]:
    if not value:
        return []
    return [keyword, f"  {value}"]


def _root_count(ciroots: str) -> int:
    try:
        return max(1, int(ciroots.split()[0]))
    except Exception:
        return 1


def _state_line(count: int) -> str:
    return " ".join(str(i) for i in range(1, count + 1))


def render_rasscf_block(
    *,
    title: str,
    spin: int,
    nactel: str,
    inactive: str,
    ras1: str,
    ras2: str,
    ras3: str,
    ciroots: str,
    iterations: str,
    levs: str,
    symmetry: int = 1,
    extra_lines: Sequence[str] = (),
) -> str:
    lines = [
        "&RASSCF",
        "Title",
        title,
        "Symmetry",
        f" {symmetry}",
        "Spin",
        f" {spin}",
        *_value_lines("nActEl", nactel),
        *_value_lines("Inactive", inactive),
        *_value_lines("Ras1", ras1),
        *_value_lines("Ras2", ras2),
        *_value_lines("Ras3", ras3),
        "Iterations",
        f" {iterations}",
        "ciroots",
        f" {ciroots}",
        "ORBL",
        " ALL",
        "ORBA",
        "COMP",
        "levs",
        f" {levs}",
        *extra_lines,
        "End of input",
    ]
    return "\n".join(lines) + "\n"


def render_caspt2_block(options: OpenMolcasPrepareOptions) -> str:
    xmul = "XMUL" if options.xmultistate else "MULT"
    lines = [
        "&CASPT2",
        xmul,
        "ALL",
        f"IPEA={options.ipea}",
        "Maxiter",
        " 100",
        "IMAG",
        f" {options.imag}",
        "THRE",
        f" {options.threshold}",
        *_value_lines("Frozen", options.frozen),
        "End of input",
    ]
    return "\n".join(lines) + "\n"


def render_rassi_block(job_count: int, state_counts: Sequence[int], *, use_jobmix: bool, sonorb: str = "") -> str:
    job_kind = "JobMix" if use_jobmix else "JobIph"
    lines: list[str] = []
    for idx in range(1, job_count + 1):
        lines.append(f">>> COPY $Project.{job_kind}_{idx} JOB{idx:03d}")
    lines.extend(["", "&RASSI", "SpinOrbit", "Ejob", "Omega"])
    if sonorb:
        lines.extend(["SONORB", *[f" {chunk}" for chunk in sonorb.replace(",", " ").split()]])
    lines.extend(["Nr of JobIph files:", f"{job_count}   " + "   ".join(str(max(1, count)) for count in state_counts)])
    for count in state_counts:
        lines.append(_state_line(max(1, count)))
    lines.append("end of input")
    return "\n".join(lines) + "\n"


def render_gasscf_selector_scout(options: OpenMolcasPrepareOptions) -> str:
    """Render a RASSCF-only true-GAS selector scout.

    The current validated OpenMolcas route cannot safely pass a true-GAS
    wavefunction through JobIph/JobMix into CASPT2/RASSI.  This renderer is
    intentionally terminal: it generates no production handoff regardless of
    the ordinary bridge defaults supplied in ``options``.
    """

    if not options.gas_spaces:
        raise ValueError("A GASSCF selector scout requires at least one --gas-space.")
    if options.gas_selector_root is None:
        raise ValueError("A GASSCF selector scout requires --gas-selector-root.")
    if not 1 <= options.gas_selector_root <= options.gas_selector_dimension:
        raise ValueError("gas selector root must be within gas selector dimension")
    if options.gas_selector_dimension < 1:
        raise ValueError("gas selector dimension must be positive")
    for space in options.gas_spaces:
        orbital_counts = space.orbitals.split()
        if not orbital_counts:
            raise ValueError(f"GAS space {space.label!r} requires one or more orbital counts")
        try:
            if any(int(value) < 0 for value in orbital_counts):
                raise ValueError
        except ValueError as exc:
            raise ValueError(f"GAS space {space.label!r} orbital counts must be non-negative integers") from exc
        if space.min_electrons < 0 or space.max_electrons < space.min_electrons:
            raise ValueError(f"Invalid cumulative occupancy bounds for GAS space {space.label!r}")

    gateway = [
        "* OpenMolcas true-GAS selector audit generated by Atomi.",
        "* Guard: no JobIph, CASPT2, JobMix, or RASSI is emitted from true GAS.",
        "&GATEWAY",
        f"COORD=$STARTDIR/{options.xyz_name}",
        "BASIS",
        options.basis,
        f"Group= {options.group}",
        "NOMOVE",
        "ANGM",
        "0.0 0.0 0.0",
    ]
    if options.include_amfi:
        gateway.append("AMFI")
    if options.include_bssh:
        gateway.append("BSSH")
    blocks = [
        "\n".join(gateway) + "\n",
        "&SEWARD\nCHOL\nEnd of input\n",
        f"&SCF\nCHARGE={options.charge}\nEnd of input\n",
        f"&SCF\nCHARGE={options.charge}\nPROR\n2 1.0 2\nEnd of input\n",
    ]
    blocks.append(
        "* Generic GAS guard: no guessed zero-electron RASSCF orbital-prep block is emitted.\n"
        "* Start from a reviewed SCF/RASSCF orbital set and supply only an element-specific, scout-validated Alter table.\n"
    )
    safe_extra = tuple(
        line
        for line in options.extra_rasscf_lines
        if not any(token in line.lower() for token in ("jobiph", "jobmix", "caspt2", "rassi"))
    )
    lines = [
        "&RASSCF",
        "Title",
        f" {options.title}_true_gas_selector",
        "Symmetry",
        f" {options.symmetry}",
        "Spin",
        f" {options.spin}",
        *_value_lines("nActEl", options.nactel),
        *_value_lines("Inactive", options.inactive),
        "GASSCF",
        f" {len(options.gas_spaces)}",
    ]
    for space in options.gas_spaces:
        lines.extend([f" {space.orbitals}", f" {space.min_electrons} {space.max_electrons}"])
    lines.extend(
        [
            "CIROOTS",
            f" 1 {options.gas_selector_dimension}",
            f" {options.gas_selector_root}",
            "CIONLY",
            *safe_extra,
            "OUTOrbitals",
            " AVERage",
            "ORBL",
            " NOTHING",
            "ORBA",
            " COMP",
            "PRWF",
            f" {options.gas_prwf}",
            "End of input",
            "* End of guarded true-GAS selector scout.",
            "* Build a separate common-orbital RAS continuation before CASPT2/RASSI.",
        ]
    )
    blocks.append("\n".join(lines) + "\n")
    return "\n".join(block.strip("\n") for block in blocks if block) + "\n"


def render_multiblock_ras_production(options: OpenMolcasPrepareOptions) -> str:
    """Render ordered RAS -> CASPT2 -> RASSI production blocks.

    ``options.state_blocks`` is the explicit source of both block order and
    RASSI state ordering.  It deliberately does not infer active spaces,
    roots, orbital rotations, or CIONLY policy from an element label.
    """

    if not options.state_blocks:
        raise ValueError("Multiblock RAS production requires state_blocks")
    gateway = [
        "* Ordered RAS/CASPT2/RASSI production input generated by Atomi.",
        "* The state-block JSON order is preserved in JobIph/JobMix and RASSI.",
        "&GATEWAY",
        f"COORD=$STARTDIR/{options.xyz_name}",
        "BASIS",
        options.basis,
        f"Group= {options.group}",
        "NOMOVE",
        "ANGM",
        "0.0 0.0 0.0",
    ]
    if options.include_amfi:
        gateway.append("AMFI")
    if options.include_bssh:
        gateway.append("BSSH")
    blocks = [
        "\n".join(gateway) + "\n",
        "&SEWARD\nCHOL\nEnd of input\n",
        f"&SCF\nCHARGE={options.charge}\nEnd of input\n",
        f"&SCF\nCHARGE={options.charge}\nPROR\n2 1.0 2\nEnd of input\n",
    ]
    if options.include_orbital_prep:
        blocks.append(
            render_rasscf_block(
                title=f"{options.title}_orbital_prep",
                symmetry=options.symmetry,
                spin=options.spin,
                nactel="0 0 0",
                inactive=options.inactive,
                ras1="",
                ras2=options.ras2,
                ras3="",
                ciroots="1 1 1",
                iterations=options.iterations,
                levs=options.levs,
            )
        )

    state_counts: list[int] = []
    caspt2_for_all_rassi_states = True
    rassi_index = 0
    for block in options.state_blocks:
        extra_lines = list(block.extra_lines)
        if block.cionly and not any(line.strip().upper() == "CIONLY" for line in extra_lines):
            extra_lines.append("CIONLY")
        blocks.append(
            render_rasscf_block(
                title=block.title,
                symmetry=block.symmetry,
                spin=block.spin,
                nactel=block.nactel,
                inactive=block.inactive,
                ras1=block.ras1,
                ras2=block.ras2,
                ras3=block.ras3,
                ciroots=block.ciroots,
                iterations=block.iterations,
                levs=block.levs,
                extra_lines=tuple(extra_lines),
            )
        )
        if not block.include_in_rassi:
            continue
        rassi_index += 1
        blocks.append(f">>> COPY $Project.JobIph $Project.JobIph_{rassi_index}\n")
        state_counts.append(_root_count(block.ciroots))
        if options.use_caspt2 and block.run_caspt2:
            blocks.append(render_caspt2_block(options))
            blocks.append(f">>> COPY $Project.JobMix $Project.JobMix_{rassi_index}\n")
        else:
            caspt2_for_all_rassi_states = False
    if not state_counts:
        raise ValueError("At least one state block must set include_in_rassi=true")
    blocks.append(render_rassi_block(len(state_counts), state_counts, use_jobmix=False, sonorb=options.sonorb))
    if options.use_caspt2 and caspt2_for_all_rassi_states:
        blocks.append(render_rassi_block(len(state_counts), state_counts, use_jobmix=True, sonorb=options.sonorb))
    return "\n".join(block.strip("\n") for block in blocks if block) + "\n"


def render_openmolcas_input(options: OpenMolcasPrepareOptions) -> str:
    if options.gas_spaces:
        return render_gasscf_selector_scout(options)
    if options.state_blocks:
        return render_multiblock_ras_production(options)
    gateway = [
        "* OpenMolcas input generated by Atomi. Review active spaces before production.",
        "&GATEWAY",
        f"COORD=$STARTDIR/{options.xyz_name}",
        "BASIS",
        options.basis,
        f"Group= {options.group}",
        "NOMOVE",
        "ANGM",
        "0.0 0.0 0.0",
    ]
    if options.include_amfi:
        gateway.append("AMFI")
    if options.include_bssh:
        gateway.append("BSSH")
    blocks = [
        "\n".join(gateway) + "\n",
        "&SEWARD\nCHOL\nEnd of input\n",
        f"&SCF\nCHARGE={options.charge}\nEnd of input\n",
        f"&SCF\nCHARGE={options.charge}\nPROR\n2 1.0 2\nEnd of input\n",
    ]
    if options.recipe == "actinide-m45-xanes":
        blocks.append(
            "\n".join(
                [
                    "* Actinide M4,5-edge scaffold:",
                    "* 1. Build and inspect full-core U-O cluster orbitals.",
                    "* 2. Build separate core-hole states; RASSCF manual recommends separate core-hole calculations.",
                    "* 3. Put U 3d-like orbitals in RAS1 and U 5f/ligand covalent orbitals in RAS2/RAS3 as justified.",
                    "* 4. Couple full-core and core-hole JOBIPH/JOBMIX files through spin-orbit RASSI.",
                    "* 5. Do not use an ECP/basis that removes U 3d core states for M4,5 spectroscopy.",
                    f"* User note: {options.core_hole_note}" if options.core_hole_note else "* User note: add oxidation/core-hole/site-specific details here.",
                    "",
                ]
            )
        )
    if options.include_orbital_prep:
        blocks.append(
            render_rasscf_block(
                title=f"{options.title}_orbital_prep",
                spin=options.spin,
                nactel="0 0 0",
                inactive=options.inactive,
                ras1="",
                ras2="0 0",
                ras3="",
                ciroots="1 1 1",
                iterations=options.iterations,
                levs=options.levs,
                symmetry=options.symmetry,
            )
        )
    blocks.append(
        render_rasscf_block(
            title=f"{options.title}_state_1",
            spin=options.spin,
            nactel=options.nactel,
            inactive=options.inactive,
            ras1=options.ras1,
            ras2=options.ras2,
            ras3=options.ras3,
            ciroots=options.ciroots,
            iterations=options.iterations,
            levs=options.levs,
            symmetry=options.symmetry,
            extra_lines=options.extra_rasscf_lines,
        )
    )
    blocks.append(">>> COPY $Project.JobIph $Project.JobIph_1\n")
    if options.use_caspt2:
        blocks.append(render_caspt2_block(options))
        blocks.append(">>> COPY $Project.JobMix $Project.JobMix_1\n")
    state_counts = [_root_count(options.ciroots)]
    if options.include_partner:
        blocks.append(
            render_rasscf_block(
                title=f"{options.title}_state_2",
                spin=options.partner_spin,
                nactel=options.nactel,
                inactive=options.inactive,
                ras1=options.ras1,
                ras2=options.ras2,
                ras3=options.ras3,
                ciroots=options.partner_ciroots,
                iterations=options.iterations,
                levs=options.levs,
                symmetry=options.symmetry,
                extra_lines=options.extra_rasscf_lines,
            )
        )
        blocks.append(">>> COPY $Project.JobIph $Project.JobIph_2\n")
        if options.use_caspt2:
            blocks.append(render_caspt2_block(options))
            blocks.append(">>> COPY $Project.JobMix $Project.JobMix_2\n")
        state_counts.append(_root_count(options.partner_ciroots))
    if state_counts:
        blocks.append(render_rassi_block(len(state_counts), state_counts, use_jobmix=False, sonorb=options.sonorb))
        if options.use_caspt2:
            blocks.append(render_rassi_block(len(state_counts), state_counts, use_jobmix=True, sonorb=options.sonorb))
    return "\n".join(block.strip("\n") for block in blocks if block) + "\n"


def write_run_scripts(args: argparse.Namespace, outdir: Path, input_name: str) -> dict[str, str]:
    executable = args.executable or default_molcas_executable()
    module_name = args.module if args.module is not None else default_molcas_module()
    label = args.label
    run_script = outdir / "run_openmolcas.sh"
    module_block = ""
    if module_name:
        module_block = (
            f"export ATOMI_MOLCAS_MODULE={_shell_quote(module_name)}\n"
            "if ! type module >/dev/null 2>&1; then\n"
            "  source /etc/profile.d/modules.sh 2>/dev/null || true\n"
            "fi\n"
            "if type module >/dev/null 2>&1; then\n"
            "  module load \"${ATOMI_MOLCAS_MODULE}\"\n"
            "else\n"
            "  echo \"WARNING: module command is unavailable; expecting OpenMolcas already on PATH.\" >&2\n"
            "fi\n"
        )
    run_script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "unset LANG; export LC_ALL=C\n"
        "export MKL_NUM_THREADS=1\n"
        "export OMP_NUM_THREADS=1\n"
        "ulimit -s 200000 || true\n"
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'cd "${SCRIPT_DIR}"\n'
        f"{module_block}"
        f"export ATOMI_MOLCAS_EXE={_shell_quote(executable)}\n"
        f"export MOLCAS_NPROCS=${{MOLCAS_NPROCS:-{args.ntasks}}}\n"
        f"export MOLCAS_MEM=${{MOLCAS_MEM:-{args.mem_per_cpu_mb}}}\n"
        'export STARTDIR="${SCRIPT_DIR}"\n'
        f'export MOLCAS_WORKDIR="${{MOLCAS_WORKDIR:-${{TMPDIR:-/tmp/${{USER}}}}/{label}.molcas_scratch}}"\n'
        'mkdir -p "${MOLCAS_WORKDIR}"\n'
        'echo "STARTDIR=${STARTDIR}"\n'
        'echo "MOLCAS_WORKDIR=${MOLCAS_WORKDIR}"\n'
        'echo "MOLCAS_NPROCS=${MOLCAS_NPROCS}"\n'
        f'"${{ATOMI_MOLCAS_EXE}}" -np "${{MOLCAS_NPROCS}}" "{input_name}" > "{label}.out"\n'
        "status=$?\n"
        f'tar -czf "{label}.molcas_work.tgz" -C "${{MOLCAS_WORKDIR}}" . || true\n'
        'rm -rf "${MOLCAS_WORKDIR}" || true\n'
        "exit ${status}\n",
        encoding="utf-8",
    )
    run_script.chmod(0o755)
    sbatch = outdir / "submit_openmolcas.sbatch"
    sbatch.write_text(
        "#!/bin/bash\n"
        f"#SBATCH --job-name={args.job_name or label}\n"
        "#SBATCH --nodes=1\n"
        f"#SBATCH --ntasks={args.ntasks}\n"
        f"#SBATCH --mem-per-cpu={args.mem_per_cpu_mb}M\n"
        f"#SBATCH --time={args.time}\n"
        f"#SBATCH --gres=scratch:{args.scratch_gb}\n"
        "#SBATCH --output=%x.%j.out\n"
        "#SBATCH --error=%x.%j.out\n"
        "\n"
        "set -euo pipefail\n"
        f"export MOLCAS_NPROCS={args.ntasks}\n"
        f"export MOLCAS_MEM={args.mem_per_cpu_mb}\n"
        "bash run_openmolcas.sh\n",
        encoding="utf-8",
    )
    sbatch.chmod(0o755)
    return {"run_script": str(run_script.resolve()), "sbatch_script": str(sbatch.resolve())}


def _parse_gas_space_specs(specs: Sequence[str]) -> tuple[GasSpace, ...]:
    """Parse ``label|n1,n2,...,nN|min|max`` GAS CLI values."""

    spaces: list[GasSpace] = []
    for raw in specs:
        fields = [item.strip() for item in raw.split("|")]
        if len(fields) != 4 or not all(fields):
            raise ValueError("--gas-space must be label|n1,n2,...,nN|min|max")
        label, vector_csv, lower, upper = fields
        vector_parts = [item.strip() for item in vector_csv.split(",")]
        if not vector_parts or any(not item for item in vector_parts):
            raise ValueError("--gas-space needs one or more comma-separated orbital counts")
        try:
            vector = [int(item) for item in vector_parts]
            minimum = int(lower)
            maximum = int(upper)
        except ValueError as exc:
            raise ValueError("--gas-space vectors and bounds must be integers") from exc
        if any(item < 0 for item in vector) or minimum < 0 or maximum < minimum:
            raise ValueError("--gas-space counts/bounds must be non-negative and ordered")
        spaces.append(GasSpace(label=label, orbitals=" ".join(str(item) for item in vector), min_electrons=minimum, max_electrons=maximum))
    return tuple(spaces)


def _load_state_blocks(path: Path | None) -> tuple[RasscfStateBlock, ...]:
    """Load explicit RAS/RASSCF block order from a JSON list or ``blocks`` map."""

    if path is None:
        return ()
    raw = json.loads(path.expanduser().read_text(encoding="utf-8"))
    records = raw.get("blocks") if isinstance(raw, dict) else raw
    if not isinstance(records, list) or not records:
        raise ValueError("--state-blocks-json must contain a non-empty JSON list or an object with a blocks list")
    blocks: list[RasscfStateBlock] = []
    required = {"title", "symmetry", "spin", "nactel", "inactive"}
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"state block {index} is not a JSON object")
        missing = sorted(required.difference(record))
        if missing:
            raise ValueError(f"state block {index} is missing required fields: {', '.join(missing)}")
        extra_lines = record.get("extra_lines", [])
        if not isinstance(extra_lines, list) or not all(isinstance(line, str) for line in extra_lines):
            raise ValueError(f"state block {index} extra_lines must be a list of strings")
        if any("gasscf" in line.lower() for line in extra_lines):
            raise ValueError("state-blocks-json is for the RAS continuation; true GAS belongs in the guarded selector-scout path")
        blocks.append(
            RasscfStateBlock(
                title=str(record["title"]),
                symmetry=int(record["symmetry"]),
                spin=int(record["spin"]),
                nactel=str(record["nactel"]),
                inactive=str(record["inactive"]),
                ras1=str(record.get("ras1", "")),
                ras2=str(record.get("ras2", "")),
                ras3=str(record.get("ras3", "")),
                ciroots=str(record.get("ciroots", "1 1 1")),
                iterations=str(record.get("iterations", "200 100")),
                levs=str(record.get("levs", "2.0")),
                cionly=bool(record.get("cionly", False)),
                extra_lines=tuple(extra_lines),
                include_in_rassi=bool(record.get("include_in_rassi", True)),
                run_caspt2=bool(record.get("run_caspt2", True)),
            )
        )
    return tuple(blocks)


def prepare_main(args: argparse.Namespace) -> dict[str, Any]:
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    xyz_source = args.xyz.expanduser().resolve()
    xyz_name = args.xyz_name or xyz_source.name
    xyz_target = outdir / xyz_name
    if args.copy_xyz and xyz_source != xyz_target:
        shutil.copy2(xyz_source, xyz_target)
    gas_spaces = _parse_gas_space_specs(tuple(getattr(args, "gas_space", ()) or ()))
    state_blocks = _load_state_blocks(getattr(args, "state_blocks_json", None))
    if gas_spaces and state_blocks:
        raise ValueError("Use either --gas-space for a guarded true-GAS scout or --state-blocks-json for a RAS production continuation, not both")
    if args.recipe == "gasscf-selector-scout" and not gas_spaces:
        raise ValueError("gasscf-selector-scout requires one or more --gas-space values")
    if args.recipe == "raspt2-rassi-blocks" and not state_blocks:
        raise ValueError("raspt2-rassi-blocks requires --state-blocks-json")
    gas_selector_root = getattr(args, "gas_selector_root", None)
    gas_selector_dimension = int(getattr(args, "gas_selector_dimension", 10))
    if gas_spaces and gas_selector_root is None:
        raise ValueError("--gas-selector-root is required whenever --gas-space is used")
    if gas_spaces and not 1 <= int(gas_selector_root) <= gas_selector_dimension:
        raise ValueError("--gas-selector-root must be within --gas-selector-dimension")
    requested_caspt2 = not args.no_caspt2
    requested_partner = not args.no_partner
    effective_recipe = "gasscf-selector-scout" if gas_spaces else "raspt2-rassi-blocks" if state_blocks else args.recipe
    options = OpenMolcasPrepareOptions(
        title=args.label,
        xyz_name=xyz_name,
        charge=args.charge,
        symmetry=int(getattr(args, "symmetry", 1)),
        spin=args.spin,
        basis=args.basis,
        group=args.group,
        recipe=effective_recipe,
        nactel=args.nactel,
        inactive=args.inactive,
        ras1=args.ras1,
        ras2=args.ras2,
        ras3=args.ras3,
        ciroots=args.ciroots,
        iterations=args.iterations,
        levs=args.levs,
        frozen=args.frozen,
        ipea=args.ipea,
        imag=args.imag,
        threshold=args.threshold,
        use_caspt2=False if gas_spaces else requested_caspt2,
        xmultistate=not args.multistate,
        include_orbital_prep=False if gas_spaces or state_blocks else not args.no_orbital_prep,
        include_partner=False if gas_spaces or state_blocks else requested_partner,
        partner_spin=args.partner_spin,
        partner_ciroots=args.partner_ciroots,
        sonorb=args.sonorb,
        include_bssh=not args.no_bssh,
        include_amfi=not args.no_amfi,
        core_hole_note=args.core_hole_note,
        extra_rasscf_lines=tuple(args.extra_rasscf_line or ()),
        gas_spaces=gas_spaces,
        gas_selector_dimension=gas_selector_dimension,
        gas_selector_root=gas_selector_root,
        gas_prwf=str(getattr(args, "gas_prwf", "1.0E-03")),
        state_blocks=state_blocks,
    )
    input_name = f"{args.label}.inp"
    molcas_input = outdir / input_name
    molcas_input.write_text(render_openmolcas_input(options), encoding="utf-8")
    scripts = write_run_scripts(args, outdir, input_name)
    metadata = {
        "schema": SCHEMA_PROJECT,
        "mode": "prepare",
        "role": "OpenMolcas CASSCF/CASPT2/RASSI bridge for cluster spectroscopy",
        "recipe": effective_recipe,
        "xyz_source": str(xyz_source),
        "xyz": str(xyz_target if xyz_target.exists() else xyz_source),
        "input": str(molcas_input.resolve()),
        "scripts": scripts,
        "options": asdict(options),
        "module": args.module if args.module is not None else default_molcas_module(),
        "recommendations": [
            "Inspect SCF/RASSCF orbitals before trusting the active space.",
            "For M4,5 edges, keep full-core and core-hole RASSCF states separate before RASSI.",
            "For U-O covalency, consider adding selected O 2p / U 5f bonding-antibonding orbitals to the active space.",
            "Benchmark a simpler oxide/carbide cluster before interpreting U4O9 cluster-family trends.",
        ],
    }
    if gas_spaces:
        metadata["true_gas_guard"] = {
            "active": True,
            "suppressed": ["JobIph", "JobMix", "CASPT2", "RASSI"],
            "requested_caspt2": requested_caspt2,
            "requested_partner": requested_partner,
            "requested_orbital_prep": not args.no_orbital_prep,
            "reason": "The validated true-GAS path is a selector-root diagnostic only; create a separate common-orbital RAS continuation after root/sector audit.",
        }
    if state_blocks:
        metadata["ras_production_order"] = {
            "state_block_count": len(state_blocks),
            "jobiph_jobmix_rassi_order": [block.title for block in state_blocks if block.include_in_rassi],
            "rule": "The JSON list is the exact RASSCF, JobIph/JobMix, and RASSI ordering. Review common-orbital/CIONLY design before submission.",
        }
    project = outdir / "openmolcas_bridge_project.json"
    _json_dump(metadata, project)
    print(f"Wrote OpenMolcas workspace: {outdir}")
    print(f"Wrote input: {molcas_input}")
    return metadata


def parse_molcas_output(path: Path) -> dict[str, Any]:
    text = path.expanduser().read_text(encoding="utf-8", errors="replace")
    module_stops = [
        {"module": match.group(1).lower(), "return_code": match.group(2)}
        for match in re.finditer(r"--- Stop Module:\s+([A-Za-z0-9_]+).*?/rc=([^ ]+)", text)
    ]
    caspt2_roots = [
        {"kind": match.group(1), "root": int(match.group(2)), "energy_hartree": float(match.group(3).replace("D", "E"))}
        for match in re.finditer(
            r"::\s+((?:XMS-|RMS-|MS-)?CASPT2) Root\s+(\d+)\s+Total energy:\s+([-+0-9.EDed]+)",
            text,
        )
    ]
    return {
        "schema": SCHEMA_SUMMARY,
        "output": str(path.expanduser().resolve()),
        "line_count": len(text.splitlines()),
        "all_well_count": text.count("_RC_ALL_IS_WELL_"),
        "module_stop_count": len(module_stops),
        "module_stops": module_stops,
        "rasscf_module_count": len(re.findall(r"Start Module:\s+rasscf", text, flags=re.IGNORECASE)),
        "caspt2_module_count": len(re.findall(r"Start Module:\s+caspt2", text, flags=re.IGNORECASE)),
        "rassi_module_count": len(re.findall(r"Start Module:\s+rassi", text, flags=re.IGNORECASE)),
        "caspt2_roots": caspt2_roots,
        "has_error_marker": "ERROR" in text.upper() or "_RC_INPUT_ERROR_" in text or "_RC_INTERNAL_ERROR_" in text,
    }


def collect_main(args: argparse.Namespace) -> dict[str, Any]:
    summary = parse_molcas_output(args.output)
    if args.write:
        _json_dump(summary, args.write)
    else:
        _json_dump(summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "doctor"):
        p = sub.add_parser(name, help="Check configured OpenMolcas executable/runtime.")
        p.add_argument("--executable")
        p.add_argument("--module")
        p.add_argument("--json", action="store_true")
        p.set_defaults(func=status_main)
    p = sub.add_parser("install-plan", help="Print recommended OpenMolcas HPC/KIT setup pattern.")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=install_plan_main)
    p = sub.add_parser("prepare", help="Prepare a reviewable OpenMolcas CASSCF/CASPT2/RASSI workspace.")
    p.add_argument("--xyz", type=Path, required=True, help="Cluster XYZ used by GATEWAY.")
    p.add_argument("--outdir", type=Path, default=Path("openmolcas_bridge"))
    p.add_argument("--label", default="molcas_cluster")
    p.add_argument("--xyz-name", default="", help="Name to use inside the workspace; defaults to source basename.")
    p.add_argument("--copy-xyz", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--charge", type=int, required=True)
    p.add_argument("--symmetry", type=int, default=1, help="OpenMolcas symmetry number for the requested RASSCF/GASSCF block.")
    p.add_argument("--spin", type=int, default=1)
    p.add_argument("--basis", default="ANO-RCC-VDZP")
    p.add_argument("--group", default="XYZ")
    p.add_argument("--recipe", choices=("casscf-caspt2-rassi", "ga-cluster-xanes", "actinide-m45-xanes", "gasscf-selector-scout", "raspt2-rassi-blocks"), default="casscf-caspt2-rassi")
    p.add_argument("--nactel", default="2 1 1")
    p.add_argument("--inactive", default="26 25")
    p.add_argument("--ras1", default="1 0")
    p.add_argument("--ras2", default="")
    p.add_argument("--ras3", default="0 3")
    p.add_argument("--ciroots", default="1 1 1")
    p.add_argument("--iterations", default="200 100")
    p.add_argument("--levs", default="2.0")
    p.add_argument("--frozen", default="")
    p.add_argument("--ipea", default="0")
    p.add_argument("--imag", default="5.0")
    p.add_argument("--threshold", default="1.0E-09 1.0E-07")
    p.add_argument("--no-caspt2", action="store_true")
    p.add_argument("--multistate", action="store_true", help="Use MULT instead of XMUL in CASPT2 blocks.")
    p.add_argument("--no-orbital-prep", action="store_true")
    p.add_argument("--no-partner", action="store_true")
    p.add_argument("--partner-spin", type=int, default=3)
    p.add_argument("--partner-ciroots", default="3 3 1")
    p.add_argument("--sonorb", default="", help="Optional SONORB orbital list, comma/space separated.")
    p.add_argument("--no-bssh", action="store_true")
    p.add_argument("--no-amfi", action="store_true")
    p.add_argument("--core-hole-note", default="")
    p.add_argument("--extra-rasscf-line", action="append", default=[])
    p.add_argument(
        "--gas-space",
        action="append",
        default=[],
        help="Repeatable true-GAS space: label|n1,n2,...,nN|min|max. The vector must follow the chosen OpenMolcas group order. Enables the no-JobIph selector-scout guard.",
    )
    p.add_argument("--gas-selector-dimension", type=int, default=10, help="Davidson selector dimension for true-GAS scout.")
    p.add_argument("--gas-selector-root", type=int, default=None, help="One selected root for true-GAS scout; required with --gas-space.")
    p.add_argument("--gas-prwf", default="1.0E-03", help="PRWF threshold for true-GAS selector output.")
    p.add_argument("--state-blocks-json", type=Path, help="Explicit ordered RAS state blocks for CASPT2/RASSI continuation; see RasscfStateBlock fields.")
    p.add_argument("--executable", default="")
    p.add_argument("--module", default=None, help="Environment module to load; empty string disables module loading.")
    p.add_argument("--job-name", default="")
    p.add_argument("--ntasks", type=int, default=8)
    p.add_argument("--mem-per-cpu-mb", type=int, default=8000)
    p.add_argument("--time", default="24:00:00")
    p.add_argument("--scratch-gb", default="200")
    p.set_defaults(func=prepare_main)
    p = sub.add_parser("collect", help="Collect a compact JSON summary from an OpenMolcas output file.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--write", type=Path)
    p.set_defaults(func=collect_main)
    return parser


def main(argv: list[str] | None = None) -> Any:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


def status_cli(argv: list[str] | None = None) -> Any:
    extra = sys.argv[1:] if argv is None else argv
    return main(["status", *extra])


def install_plan_cli(argv: list[str] | None = None) -> Any:
    extra = sys.argv[1:] if argv is None else argv
    return main(["install-plan", *extra])


if __name__ == "__main__":
    main()
