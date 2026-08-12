#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bootstrap_public_sources.sh [--execute]

Prints the public-source bootstrap plan by default. With --execute, downloads
the modest validation inputs and the chosen UniProtKB FASTA files. AFDB is
never fetched implicitly: its Foldseek command is printed because it requires
roughly 151 GB RAM and substantial disk for the CA-backed database.

Required environment:
  PROTEINLOSS_SOURCE_ROOT   destination directory

Optional environment:
  UNIPROT_RELEASE_URL       immutable release directory (recommended)
  DOWNLOAD_PROTEINGYM      1 (default) or 0
  DOWNLOAD_UNIPROT         1 (default) or 0
EOF
}

execute=0
case "${1:-}" in
  --execute) execute=1 ;;
  -h|--help) usage; exit 0 ;;
  "") ;;
  *) usage >&2; exit 2 ;;
esac

source_root="${PROTEINLOSS_SOURCE_ROOT:-}"
if [[ -z "${source_root}" ]]; then
  echo "PROTEINLOSS_SOURCE_ROOT is required." >&2
  exit 2
fi

uniprot_url="${UNIPROT_RELEASE_URL:-https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete}"
proteingym_zip_url="https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip"
proteingym_meta_url="https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/144fe22b07dfaeec2b366f2346203a9838a55b4c/reference_files/DMS_substitutions.csv"

cat <<EOF
Destination: ${source_root}
UniProt source: ${uniprot_url}
ProteinGym: ${proteingym_zip_url}

Large manual steps:
  1. Download/pin UniClust30 and build its membership SQLite index.
  2. Install pinned MMseqs2 and Foldseek versions.
  3. Run: foldseek databases Alphafold/UniProt50 <PREFIX> <TMP>
  4. Obtain the project StructEncoder teacher checkpoint and verify SHA-256.
  5. Download/cache selected RCSB mmCIF and CASP15 targets.

See docs/DATA_SOURCES.md for exact contracts and conversion commands.
EOF

if [[ "${execute}" -eq 0 ]]; then
  echo "Dry run only; pass --execute to download public files."
  exit 0
fi

mkdir -p "${source_root}/uniprot" "${source_root}/proteingym"

if [[ "${DOWNLOAD_PROTEINGYM:-1}" == "1" ]]; then
  curl --fail --location --continue-at - --output "${source_root}/proteingym/DMS_ProteinGym_substitutions.zip" "${proteingym_zip_url}"
  curl --fail --location --output "${source_root}/proteingym/DMS_substitutions.csv" "${proteingym_meta_url}"
fi

if [[ "${DOWNLOAD_UNIPROT:-1}" == "1" ]]; then
  curl --fail --location --continue-at - --output "${source_root}/uniprot/uniprot_sprot.fasta.gz" "${uniprot_url}/uniprot_sprot.fasta.gz"
  curl --fail --location --continue-at - --output "${source_root}/uniprot/uniprot_trembl.fasta.gz" "${uniprot_url}/uniprot_trembl.fasta.gz"
  curl --fail --location --output "${source_root}/uniprot/RELEASE.metalink" "${uniprot_url}/RELEASE.metalink"
fi

(cd "${source_root}" && sha256sum uniprot/* proteingym/* > downloaded.sha256)
echo "Downloaded public files and wrote ${source_root}/downloaded.sha256"
