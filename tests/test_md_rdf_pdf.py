from pathlib import Path

from atomi.md.rdf_pdf import rdf_pdf_from_vasp_xdatcar


def write_vasp_pair(tmp_path: Path) -> tuple[Path, Path]:
    poscar = tmp_path / "POSCAR"
    xdatcar = tmp_path / "XDATCAR"
    poscar.write_text(
        """Toy UC
1.0
10.0 0.0 0.0
0.0 10.0 0.0
0.0 0.0 10.0
U C
1 1
Direct
0.0 0.0 0.0
0.25 0.0 0.0
""",
        encoding="utf-8",
    )
    xdatcar.write_text(
        """Toy UC
1.0
10.0 0.0 0.0
0.0 10.0 0.0
0.0 0.0 10.0
U C
1 1
Direct configuration=     1
0.0 0.0 0.0
0.25 0.0 0.0
Direct configuration=     2
0.0 0.0 0.0
0.30 0.0 0.0
""",
        encoding="utf-8",
    )
    return poscar, xdatcar


def test_vasp_xdatcar_rdf_pdf_outputs_element_swappable_columns(tmp_path: Path) -> None:
    poscar, xdatcar = write_vasp_pair(tmp_path)
    outdir = tmp_path / "rdf"

    meta = rdf_pdf_from_vasp_xdatcar(
        poscar=poscar,
        xdatcar=xdatcar,
        outdir=outdir,
        prefix="toy",
        species_order=["C", "U"],
        weights={"C": 6.0, "U": 92.0},
        rmax=5.0,
        dr=0.1,
    )

    partial_csv = outdir / "toy_partial_rdfs.csv"
    total_csv = outdir / "toy_total_pdf.csv"
    metadata_json = outdir / "toy_rdf_pdf_metadata.json"
    assert partial_csv.exists()
    assert total_csv.exists()
    assert metadata_json.exists()
    header = partial_csv.read_text(encoding="utf-8").splitlines()[0]
    assert "g_C_U" in header
    assert "g_total_weighted" in header
    assert meta["species_order"] == ["C", "U"]
    assert meta["n_selected_frames"] == 2
