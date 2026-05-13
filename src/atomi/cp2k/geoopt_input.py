import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from atomi.cp2k.acid_box import KIND_DEFAULTS, render_kinds


CP2K_DATA_DIR = os.environ.get("ATOMI_CP2K_DATA_DIR") or os.environ.get("CP2K_DATA_DIR") or "."


@dataclass(frozen=True)
class StageSettings:
    cutoff: int
    rel_cutoff: int
    eps_scf: str
    max_scf: int
    outer_max_scf: int
    outer_eps_scf: str
    optimizer: str
    max_iter: int
    max_force: str
    rms_force: str
    max_dr: str
    rms_dr: str | None
    use_extrapolation: bool


STAGE_SETTINGS = {
    "cheap": StageSettings(
        cutoff=300,
        rel_cutoff=40,
        eps_scf="1.0E-4",
        max_scf=100,
        outer_max_scf=10,
        outer_eps_scf="1.0E-4",
        optimizer="LBFGS",
        max_iter=150,
        max_force="5.0E-3",
        rms_force="2.0E-3",
        max_dr="5.0E-3",
        rms_dr=None,
        use_extrapolation=False,
    ),
    "refine": StageSettings(
        cutoff=350,
        rel_cutoff=50,
        eps_scf="5.0E-5",
        max_scf=150,
        outer_max_scf=15,
        outer_eps_scf="5.0E-5",
        optimizer="LBFGS",
        max_iter=200,
        max_force="2.0E-3",
        rms_force="8.0E-4",
        max_dr="2.0E-3",
        rms_dr=None,
        use_extrapolation=True,
    ),
    "final": StageSettings(
        cutoff=400,
        rel_cutoff=50,
        eps_scf="1.0E-5",
        max_scf=200,
        outer_max_scf=20,
        outer_eps_scf="1.0E-5",
        optimizer="LBFGS",
        max_iter=250,
        max_force="1.0E-3",
        rms_force="7.0E-4",
        max_dr="2.0E-3",
        rms_dr=None,
        use_extrapolation=True,
    ),
}


def read_xyz_symbols(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"not enough lines for XYZ file: {path}")
    try:
        natoms = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"first XYZ line must be atom count: {path}") from exc
    symbols = []
    for line in lines[2 : 2 + natoms]:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"malformed XYZ atom line in {path}: {line!r}")
        symbols.append(parts[0])
    if len(symbols) != natoms:
        raise ValueError(f"XYZ atom count mismatch in {path}")
    return symbols


def estimate_box_from_xyz(path: Path, padding: float = 10.0, min_box: float = 18.0) -> float:
    coords = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 4:
            coords.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if not coords:
        raise ValueError(f"no coordinates found in {path}")
    mins = [min(row[i] for row in coords) for i in range(3)]
    maxs = [max(row[i] for row in coords) for i in range(3)]
    span = max(maxs[i] - mins[i] for i in range(3))
    return round(max(min_box, span + padding), 2)


def extract_step_start_val(restart_file: Path) -> int:
    if not restart_file.is_file():
        return 0
    pattern = re.compile(r"^\s*STEP_START_VAL\s+(\d+)\b")
    value = 0
    with restart_file.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = pattern.match(line)
            if match:
                value = int(match.group(1))
    return value


def resolve_max_iter(mode: str, requested_steps: int, restart_file: Path) -> tuple[int, int]:
    step_start = extract_step_start_val(restart_file) if mode == "restart" else 0
    max_iter = step_start + requested_steps if mode == "restart" else requested_steps
    return max_iter, step_start


def normalize_optimizer(value: str) -> str:
    opt = value.upper()
    if opt not in {"LBFGS", "BFGS"}:
        raise ValueError("--optimizer must be lbfgs or bfgs")
    return opt


def normalize_scf_guess(value: str) -> str:
    guess = value.upper()
    if guess not in {"ATOMIC", "RESTART"}:
        raise ValueError("--scf-guess must be atomic or restart")
    return guess


def read_optional_snippet(path: Path | None) -> str:
    if path is None:
        return ""
    text = path.read_text(encoding="utf-8").strip()
    return text + "\n\n" if text else ""


def render_geoopt_input(args: argparse.Namespace) -> str:
    settings = STAGE_SETTINGS[args.stage]
    optimizer = normalize_optimizer(args.optimizer or settings.optimizer)
    scf_guess = normalize_scf_guess(args.scf_guess)
    extrapolation = ""
    if settings.use_extrapolation:
        extrapolation = "\n      EXTRAPOLATION ASPC\n      EXTRAPOLATION_ORDER 3"
    ext_restart = ""
    if args.mode == "restart":
        ext_restart = f"""&EXT_RESTART
  RESTART_FILE_NAME {args.restart_file}
&END EXT_RESTART

"""
    rms_dr = f"\n    RMS_DR    {settings.rms_dr}" if settings.rms_dr else ""
    colvars = read_optional_snippet(args.colvar_file)
    constraints = read_optional_snippet(args.constraint_file)
    return f"""&GLOBAL
  PROJECT {args.project}
  RUN_TYPE GEO_OPT
  PRINT_LEVEL MEDIUM
&END GLOBAL

{ext_restart}&FORCE_EVAL
  METHOD QS

  &DFT
    BASIS_SET_FILE_NAME {args.basis_file}
    POTENTIAL_FILE_NAME {args.potential_file}
    CHARGE {args.charge}
    MULTIPLICITY {args.multiplicity}

    &MGRID
      CUTOFF {settings.cutoff}
      REL_CUTOFF {settings.rel_cutoff}
      NGRIDS 4
    &END MGRID

    &QS
      METHOD GPW
      EPS_DEFAULT 1.0E-12{extrapolation}
    &END QS

    &SCF
      MAX_SCF {settings.max_scf}
      EPS_SCF {settings.eps_scf}
      SCF_GUESS {scf_guess}

      &OT
        MINIMIZER CG
        PRECONDITIONER FULL_ALL
        ENERGY_GAP 0.20
      &END OT

      &OUTER_SCF
        MAX_SCF {settings.outer_max_scf}
        EPS_SCF {settings.outer_eps_scf}
      &END OUTER_SCF

      &PRINT
        &RESTART
          BACKUP_COPIES 1
        &END RESTART
      &END PRINT
    &END SCF

    &XC
      &XC_FUNCTIONAL PBE
      &END XC_FUNCTIONAL

      &VDW_POTENTIAL
        POTENTIAL_TYPE PAIR_POTENTIAL
        &PAIR_POTENTIAL
          TYPE DFTD3
          PARAMETER_FILE_NAME {args.d3_file}
          REFERENCE_FUNCTIONAL PBE
        &END PAIR_POTENTIAL
      &END VDW_POTENTIAL
    &END XC
  &END DFT

  &SUBSYS
    &CELL
      ABC {args.box:.3f} {args.box:.3f} {args.box:.3f}
      PERIODIC XYZ
    &END CELL

    &TOPOLOGY
      COORD_FILE_FORMAT XYZ
      COORD_FILE_NAME {args.xyz.name if args.xyz_name_only else args.xyz}
    &END TOPOLOGY

{colvars}{render_kinds(args.symbols)}
  &END SUBSYS
&END FORCE_EVAL

&MOTION
  &GEO_OPT
    TYPE MINIMIZATION
    OPTIMIZER {optimizer}
    MAX_ITER {args.max_iter}
    MAX_FORCE {settings.max_force}
    RMS_FORCE {settings.rms_force}
    MAX_DR {settings.max_dr}{rms_dr}
  &END GEO_OPT

{constraints}  &PRINT
    &TRAJECTORY
      FORMAT XYZ
      &EACH
        GEO_OPT 1
      &END EACH
    &END TRAJECTORY

    &RESTART
      BACKUP_COPIES 2
    &END RESTART

    &RESTART_HISTORY
      &EACH
        GEO_OPT 1
      &END EACH
    &END RESTART_HISTORY
  &END PRINT
&END MOTION
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cp2k-geoopt-input",
        description="Write staged CP2K GEO_OPT inputs with restart-aware MAX_ITER.",
    )
    parser.add_argument("--xyz", type=Path, required=True, help="XYZ coordinate file.")
    parser.add_argument("--stage", choices=sorted(STAGE_SETTINGS), required=True)
    parser.add_argument("--mode", choices=("start", "restart"), required=True)
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--multiplicity", type=int, default=1)
    parser.add_argument("--project", default=None)
    parser.add_argument("--box", type=float, default=None)
    parser.add_argument("--restart-file", type=Path, default=None)
    parser.add_argument("--scf-guess", choices=("atomic", "restart"), default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--optimizer", choices=("lbfgs", "bfgs"), default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--basis-file", default=f"{CP2K_DATA_DIR}/BASIS_MOLOPT")
    parser.add_argument("--potential-file", default=f"{CP2K_DATA_DIR}/GTH_POTENTIALS")
    parser.add_argument("--d3-file", default=f"{CP2K_DATA_DIR}/dftd3.dat")
    parser.add_argument("--colvar-file", type=Path, default=None)
    parser.add_argument("--constraint-file", type=Path, default=None)
    parser.add_argument(
        "--xyz-name-only",
        action="store_true",
        help="Write only the XYZ basename in COORD_FILE_NAME.",
    )
    return parser


def infer_project(xyz: Path, project: str | None) -> str:
    return project or xyz.stem


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.xyz.is_file():
        parser.error(f"XYZ file not found: {args.xyz}")

    args.project = infer_project(args.xyz, args.project)
    args.out = args.out or Path(f"{args.project}.inp")
    args.restart_file = args.restart_file or Path(f"{args.project}-1.restart")
    args.scf_guess = args.scf_guess or ("atomic" if args.mode == "start" else "restart")
    args.symbols = read_xyz_symbols(args.xyz)
    for symbol in sorted(set(args.symbols)):
        if symbol not in KIND_DEFAULTS:
            print(
                f"WARNING: no curated KIND default for {symbol}; using generic GTH-PBE.",
                file=sys.stderr,
            )
    if args.box is None:
        args.box = estimate_box_from_xyz(args.xyz)
    requested_steps = args.max_iter or STAGE_SETTINGS[args.stage].max_iter
    args.max_iter, step_start = resolve_max_iter(args.mode, requested_steps, args.restart_file)

    args.out.write_text(render_geoopt_input(args), encoding="utf-8")

    print(f"Wrote {args.out}")
    print(f"  xyz        = {args.xyz}")
    print(f"  stage      = {args.stage}")
    print(f"  mode       = {args.mode}")
    print(f"  charge     = {args.charge}")
    print(f"  project    = {args.project}")
    print(f"  box        = {args.box:.3f} Angstrom")
    print(f"  scf_guess  = {normalize_scf_guess(args.scf_guess)}")
    optimizer = normalize_optimizer(args.optimizer or STAGE_SETTINGS[args.stage].optimizer)
    print(f"  optimizer  = {optimizer}")
    if args.mode == "restart":
        print(f"  restart    = {args.restart_file}")
        print(f"  step_start = {step_start}")
        print(f"  add_steps  = {requested_steps}")
        print(f"  max_iter   = {args.max_iter}  (= {step_start} + {requested_steps})")
    else:
        print(f"  max_iter   = {args.max_iter}")
    print("NOTE: XYZ does not store box size; CP2K uses the fixed &CELL from this input.")


if __name__ == "__main__":
    main(sys.argv[1:])
