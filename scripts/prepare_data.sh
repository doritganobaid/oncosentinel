#!/usr/bin/env bash
set -euo pipefail

echo "FM-MAPB evaluates on public retrospective cohorts. Access points:"
echo "  TCGA Pan-Cancer        portal.gdc.cancer.gov         (GDC release 32.0, open)"
echo "  IMvigor210             ESMO Open final analysis      (open)"
echo "  OAK / POPLAR           published supplements         (open)"
echo "  UK Biobank Cardiac MRI ukbiobank.ac.uk               (restricted, application)"
echo "  CARDIOTOX/CARDIOCARE   zenodo.org/records/19727929   (CC-BY-NC)"
echo "  MIMIC-IV-ECG           physionet.org                 (restricted, CITI credential)"
echo
echo "Restricted cohorts require access agreements and are not redistributed here."
echo "Deterministic synthetic fixtures matched to reported cohort sizes are generated"
echo "on demand by oncosentinel.knowledge.cohort_synthesis; no download is needed to run the pipeline."
