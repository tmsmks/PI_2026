# Données externes — attribution

## `eaglei_maricopa_2022.csv`

Extrait (filtré sur le comté de **Maricopa, Arizona**, FIPS `04013`) du jeu de
données **EAGLE-I™** :

> The Environment for Analysis of Geo-Located Energy Information's Recorded
> Electricity Outages 2014–2023. Oak Ridge National Laboratory / U.S. Department
> of Energy, Office of Electricity.
> DOI : https://doi.org/10.6084/m9.figshare.24237376 — **Licence : CC BY 4.0**

Colonnes : `fips_code, county, state, customers_out, run_start_time` (pas de 15 min).
Fichier source 2022 complet ≈ 1,2 Go ; on ne conserve ici que les lignes Maricopa.

Régénération :
```bash
mkdir -p data/external
curl -sL "https://ndownloader.figshare.com/files/42547897" \
  | grep -E '^(fips_code|04013),' > data/external/eaglei_maricopa_2022.csv
```

Utilisé par `src/models/external_validation.py` pour la **validation externe**
(généralisation du signal météo Lacor → Phoenix). Voir README, section
« Validation EXTERNE sur un site réel indépendant ».
