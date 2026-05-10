import argparse
import csv
import gzip
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


STRESS_KEYS = ("xx", "yy", "zz", "yz", "xz", "xy")
PHONOPY_ARTIFACTS = ("FORCE_SETS", "thermal_properties.yaml", "total_dos.dat", "phonopy.yaml")


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("rt", encoding="utf-8", errors="replace")


def infer_scale_from_name(name: str) -> float:
    stripped = name.strip()
    match = re.match(r"V([0-9]+(?:\.[0-9]+)?)$", stripped)
    if match:
        return float(match.group(1))
    match = re.search(r"([0-9]+\.[0-9]+)", stripped)
    return float(match.group(1)) if match else math.nan


def det3(matrix: list[list[float]]) -> float:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def vector_norm(row: list[float]) -> float:
    return math.sqrt(sum(value * value for value in row))


def parse_vasprun(path: Path) -> dict:
    with open_text(path) as handle:
        root = ET.parse(handle).getroot()
    natoms = math.nan
    atominfo = root.find(".//atominfo/atoms")
    if atominfo is not None and atominfo.text:
        natoms = int(atominfo.text.strip())

    volume = math.nan
    structs = root.findall(".//structure")
    if structs:
        basis = structs[-1].find(".//crystal/varray[@name='basis']")
        if basis is not None:
            vecs = [[float(x) for x in v.text.split()] for v in basis.findall("v") if v.text]
            if len(vecs) == 3:
                volume = abs(det3(vecs))

    energy = math.nan
    energies = root.findall(".//calculation/energy/i[@name='e_fr_energy']")
    if energies and energies[-1].text:
        energy = float(energies[-1].text.strip())

    force_rms = math.nan
    force_max = math.nan
    force_blocks = root.findall(".//calculation/varray[@name='forces']")
    if force_blocks:
        forces = [
            [float(x) for x in v.text.split()]
            for v in force_blocks[-1].findall("v")
            if v.text
        ]
        if forces:
            norms = [vector_norm(row) for row in forces]
            force_rms = math.sqrt(sum(x * x for x in norms) / len(norms))
            force_max = max(norms)

    stress = {key: math.nan for key in STRESS_KEYS}
    stress_blocks = root.findall(".//calculation/varray[@name='stress']")
    if stress_blocks:
        rows = [
            [float(x) for x in v.text.split()]
            for v in stress_blocks[-1].findall("v")
            if v.text
        ]
        if len(rows) == 3:
            # VASP stress in vasprun.xml is kBar; convert to GPa.
            stress["xx"] = rows[0][0] * 0.1
            stress["yy"] = rows[1][1] * 0.1
            stress["zz"] = rows[2][2] * 0.1
            stress["yz"] = rows[1][2] * 0.1
            stress["xz"] = rows[0][2] * 0.1
            stress["xy"] = rows[0][1] * 0.1

    return calc_record(path, "vasprun_xml", energy, volume, natoms, force_rms, force_max, stress)


def parse_outcar(path: Path) -> dict:
    energy = math.nan
    volume = math.nan
    natoms = math.nan
    force_rms = math.nan
    force_max = math.nan
    stress = {key: math.nan for key in STRESS_KEYS}
    last_force_block = []
    in_force = False

    with open_text(path) as handle:
        for line in handle:
            if "NIONS =" in line:
                match = re.search(r"NIONS\s*=\s*([0-9]+)", line)
                if match:
                    natoms = int(match.group(1))
            if "free  energy   TOTEN" in line:
                match = re.search(r"TOTEN\s*=\s*([\-0-9\.Ee+]+)", line)
                if match:
                    energy = float(match.group(1))
            if "volume of cell" in line:
                match = re.search(r"volume of cell\s*:\s*([\-0-9\.Ee+]+)", line)
                if match:
                    volume = float(match.group(1))
            if "in kB" in line and "external pressure" in line:
                nums = re.findall(r"[-+]?\d+\.\d+|[-+]?\d+", line)
                if len(nums) >= 6:
                    vals = [float(x) * 0.1 for x in nums[-6:]]
                    stress["xx"], stress["yy"], stress["zz"] = vals[:3]
                    stress["xy"], stress["yz"], stress["xz"] = vals[3:6]
            if "TOTAL-FORCE (eV/Angst)" in line:
                in_force = True
                last_force_block = []
                continue
            if in_force:
                if "----" in line or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        last_force_block.append([float(parts[3]), float(parts[4]), float(parts[5])])
                        continue
                    except ValueError:
                        pass
                in_force = False

    if last_force_block:
        norms = [vector_norm(row) for row in last_force_block]
        force_rms = math.sqrt(sum(x * x for x in norms) / len(norms))
        force_max = max(norms)

    return calc_record(path, "outcar_text", energy, volume, natoms, force_rms, force_max, stress)


def calc_record(
    path: Path,
    parser: str,
    energy: float,
    volume: float,
    natoms: float,
    force_rms: float,
    force_max: float,
    stress: dict,
) -> dict:
    return {
        "energy_eV": energy,
        "volume_A3": volume,
        "natoms": natoms,
        "force_rms_eVA": force_rms,
        "force_max_eVA": force_max,
        "stress_xx_GPa": stress["xx"],
        "stress_yy_GPa": stress["yy"],
        "stress_zz_GPa": stress["zz"],
        "stress_yz_GPa": stress["yz"],
        "stress_xz_GPa": stress["xz"],
        "stress_xy_GPa": stress["xy"],
        "parser_used": parser,
        "source_file": str(path),
    }


def empty_calc_record() -> dict:
    stress = {key: math.nan for key in STRESS_KEYS}
    return calc_record(Path(""), "none", math.nan, math.nan, math.nan, math.nan, math.nan, stress)


def parse_calc_folder(folder: Path) -> dict:
    candidates = [
        folder / "vasprun.xml",
        folder / "vasprun.xml.gz",
        folder / "OUTCAR",
        folder / "OUTCAR.gz",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            if "vasprun.xml" in path.name:
                return parse_vasprun(path)
            return parse_outcar(path)
        except Exception:
            continue
    return empty_calc_record()


def find_parent_folder(volume_dir: Path, parent_pattern: str) -> Path:
    parent_dirs = sorted(path for path in volume_dir.glob(parent_pattern) if path.is_dir())
    return parent_dirs[0] if parent_dirs else volume_dir


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024) if path.exists() else math.nan


def count_disp_runs(volume_dir: Path, disp_pattern: str) -> dict:
    disp_dirs = sorted(path for path in volume_dir.glob(disp_pattern) if path.is_dir())
    done = 0
    vasprun = 0
    vasprun_gz = 0
    outcar = 0
    outcar_gz = 0
    for directory in disp_dirs:
        has_result = False
        if (directory / "vasprun.xml").exists():
            vasprun += 1
            has_result = True
        if (directory / "vasprun.xml.gz").exists():
            vasprun_gz += 1
            has_result = True
        if (directory / "OUTCAR").exists():
            outcar += 1
            has_result = True
        if (directory / "OUTCAR.gz").exists():
            outcar_gz += 1
            has_result = True
        if has_result:
            done += 1
    return {
        "n_disp_dirs": len(disp_dirs),
        "n_disp_with_result": done,
        "n_vasprun_xml": vasprun,
        "n_vasprun_xml_gz": vasprun_gz,
        "n_OUTCAR": outcar,
        "n_OUTCAR_gz": outcar_gz,
    }


def add_formula_unit_fields(record: dict, atoms_per_fu: float) -> None:
    natoms = record["natoms"]
    if not is_finite(natoms) or natoms <= 0:
        record["formula_units"] = math.nan
        record["energy_per_atom_eV"] = math.nan
        record["energy_per_fu_eV"] = math.nan
        record["volume_per_fu_A3"] = math.nan
        return
    nfu = natoms / atoms_per_fu
    record["formula_units"] = nfu
    record["energy_per_atom_eV"] = (
        record["energy_eV"] / natoms if is_finite(record["energy_eV"]) else math.nan
    )
    record["energy_per_fu_eV"] = (
        record["energy_eV"] / nfu if is_finite(record["energy_eV"]) else math.nan
    )
    record["volume_per_fu_A3"] = (
        record["volume_A3"] / nfu if is_finite(record["volume_A3"]) else math.nan
    )


def is_finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def summarize_volume(volume_dir: Path, args: argparse.Namespace) -> dict:
    parent_dir = find_parent_folder(volume_dir, args.parent_pattern)
    record = parse_calc_folder(parent_dir)
    record["volume_folder"] = volume_dir.name
    record["root"] = str(volume_dir)
    record["parent_source"] = record.pop("source_file")
    record["scale_factor"] = infer_scale_from_name(volume_dir.name)
    add_formula_unit_fields(record, args.atoms_per_fu)
    for artifact in PHONOPY_ARTIFACTS:
        path = volume_dir / artifact
        key = artifact.replace(".", "_")
        record[f"has_{key}"] = path.exists()
        record[f"{key}_MB"] = file_size_mb(path)
    record.update(count_disp_runs(volume_dir, args.disp_pattern))
    return record


def volume_dirs(root: Path, pattern: str) -> list[Path]:
    return sorted(path for path in root.glob(pattern) if path.is_dir())


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, root: Path, rows: list[dict], phonopy_module: str | None) -> None:
    valid = [
        row for row in rows if is_finite(row.get("energy_eV")) and is_finite(row.get("volume_A3"))
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("VASP phonopy QHA summary report\n")
        handle.write("=" * 40 + "\n")
        handle.write(f"Root: {root}\n")
        if phonopy_module:
            handle.write(f"Phonopy module hint: module load {phonopy_module}\n")
        handle.write(f"Volume folders scanned: {len(rows)}\n")
        handle.write(f"Valid energy/volume points: {len(valid)}\n\n")
        for row in rows:
            handle.write(
                f"{row['volume_folder']}: E={row['energy_eV']} eV, "
                f"V={row['volume_A3']} A^3, disp={row['n_disp_with_result']}/"
                f"{row['n_disp_dirs']}, parser={row['parser_used']}\n"
            )


def maybe_plot(path: Path, rows: list[dict]) -> bool:
    valid = [
        row for row in rows if is_finite(row.get("energy_eV")) and is_finite(row.get("volume_A3"))
    ]
    if not valid:
        return False
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    valid.sort(key=lambda row: row["volume_A3"])
    plt.figure(figsize=(6.8, 4.8))
    plt.plot([row["volume_A3"] for row in valid], [row["energy_eV"] for row in valid], marker="o")
    for row in valid:
        plt.annotate(
            row["volume_folder"],
            (row["volume_A3"], row["energy_eV"]),
            fontsize=8,
            xytext=(3, 3),
            textcoords="offset points",
        )
    plt.xlabel("Volume (A^3)")
    plt.ylabel("Energy (eV)")
    plt.title("QHA parent energies vs volume")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vasp-qha-summary",
        description="Summarize VASP/phonopy QHA volume folders and displacement coverage.",
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--volume-pattern", default="V*")
    parser.add_argument("--disp-pattern", default="disp-*")
    parser.add_argument("--parent-pattern", default="parent*")
    parser.add_argument("--atoms-per-fu", type=float, default=3.0)
    parser.add_argument("--phonopy-module", default=None)
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = args.root.resolve()
    outdir = args.outdir.resolve()
    if not root.is_dir():
        parser.error(f"root directory not found: {root}")
    if args.atoms_per_fu <= 0:
        parser.error("--atoms-per-fu must be positive")
    outdir.mkdir(parents=True, exist_ok=True)
    dirs = volume_dirs(root, args.volume_pattern)
    if not dirs:
        dirs = [root]
    rows = [summarize_volume(directory, args) for directory in dirs]
    rows.sort(
        key=lambda row: (
            row["scale_factor"] if is_finite(row["scale_factor"]) else math.inf,
            row["volume_folder"],
        )
    )

    csv_path = outdir / "qha_volume_summary.csv"
    report_path = outdir / "qha_summary_report.txt"
    plot_path = outdir / "qha_parent_energy_vs_volume.png"
    write_csv(csv_path, rows)
    write_report(report_path, root, rows, args.phonopy_module)
    print(csv_path)
    print(report_path)
    if not args.no_plot and maybe_plot(plot_path, rows):
        print(plot_path)


if __name__ == "__main__":
    main(sys.argv[1:])
