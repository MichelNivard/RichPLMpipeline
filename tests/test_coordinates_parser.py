from proteinloss_pipeline.coordinates import parse_mmcif_chains


def test_minimal_experimental_mmcif_ca_parser(tmp_path):
    rows = []
    amino = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE"]
    for index, residue in enumerate(amino, start=1):
        rows.append(f"ATOM {index} C CA CA {residue} A A {index} {index} {index * 3.8:.2f} 0.0 0.0 1.0 . 1")
    cif = tmp_path / "test.cif"
    cif.write_text(
        "data_test\n_exptl.method 'X-RAY DIFFRACTION'\n_refine.ls_d_res_high 2.0\n"
        "loop_\n_atom_site.group_PDB\n_atom_site.id\n_atom_site.type_symbol\n"
        "_atom_site.label_atom_id\n_atom_site.auth_atom_id\n_atom_site.label_comp_id\n"
        "_atom_site.label_asym_id\n_atom_site.auth_asym_id\n_atom_site.label_seq_id\n"
        "_atom_site.auth_seq_id\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n_atom_site.occupancy\n_atom_site.label_alt_id\n"
        "_atom_site.pdbx_PDB_model_num\n" + "\n".join(rows) + "\n#\n",
        encoding="utf-8",
    )
    chains = parse_mmcif_chains(cif, pdb_id="1abc", min_len=8, max_len=16, methods={"X-RAY DIFFRACTION"})
    assert len(chains) == 1
    assert chains[0].sequence == "ACDEFGHI"
    assert chains[0].coords_ca.shape == (8, 3)
