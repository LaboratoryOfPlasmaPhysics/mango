# MANGO column dictionary — draft (2026-10-07)

Live server checked: `GET /api/v1/regions/{r}/columns` + `/data?limit≤200` samples (THA, THD, THE, C1, C3, DS1, MMS; 2001–2019), Arrow schema from `limit=5`.
Regions: **SW** = solar_wind, **MSH** = magnetosheath, **MSP** = magnetosphere. Column list in the brief matches the server exactly (SW 13, MSH 41, MSP 32).
Note: spacecraft partition key is `MMS`, not `MMS1`.

| name | regions | dtype | unit | frame | description | how computed / source | confidence |
|---|---|---|---|---|---|---|---|
| Time | all | timestamp[ns], tz-naive | — | UTC (assumed) | Sample time | 5 s grid (seconds ∈ {0,5,…,55}; Δt mode 5 s); 5 s mean resampling of mission data | VERIFIED grid/dtype; UTC and bin label (start vs centre) UNKNOWN |
| SC | all | string (hive key) | — | — | Spacecraft id: THA THB THC THD THE C1 C3 MMS DS1 | Hive partition dir `SC=<name>` | VERIFIED |
| Bx, By, Bz | all | float64 | nT | GSM | Local magnetic field | 5 s mean FGM, median filter k=3 (memory: per-mission make_dataset notebooks) | Unit/frame INFERRED (ranges consistent); processing from memory notes |
| Np | all | float64 | cm⁻³ | — | Local ion (proton) density | Ion moments (THEMIS ESA/MOM, Cluster HIA, MMS FPI via AMDA, DS1) | INFERRED (MSH 3–91, MSP 0.001–4) |
| Vx, Vy, Vz | all | float64 | km/s | GSM | Local ion bulk velocity | Ion moments; MMS/OMNI converted GSE→GSM (memory) | INFERRED (SW Vx −1070…−257) |
| Tp | all | float64 | K | — | Local ion temperature, (T∥+2T⊥)/3 | eV × 1.1605e4 (memory) | VERIFIED unit by magnitude (SW 1.9e5–3e7, MSP up to 7e7 K; eV impossible) |
| X_gsm, Y_gsm, Z_gsm | all | float64 | R_E | GSM | Spacecraft position | Mission orbit, 5 s mean | VERIFIED unit by magnitude |
| SW_pairing | MSH, MSP | bool | — | — | True if an upstream OMNI sample was associated to this row | Šafránková-type propagation, OMNI BSN_x → sat X with V_x (`spok.models.planetary.associate_SW_Safrankova`) | Method INFERRED; observed: False ⇒ Np_sw…Ma_sw null, but Bx/By/Bz_imf often **non-null** (MSP 100 %, MSH 43 %) |
| Bx_imf, By_imf, Bz_imf | MSH, MSP | float64 | nT | GSM | Paired upstream IMF | OMNI HRO 1-min (2-decimal values, repeat ~12 rows), time-lagged | Unit/source INFERRED; GSM INFERRED (OMNI BY/BZ_GSM) |
| Np_sw | MSH, MSP | float64 | cm⁻³ | — | Paired upstream proton density | OMNI | INFERRED |
| Vx_sw | MSH, MSP | float64 | km/s | GSM (=GSE for x) | Paired upstream flow, x comp. — **negative** (anti-sunward), not speed | OMNI (1-decimal native) | VERIFIED sign (−486…−262); ⚠ filter text "bulk speed" misleading |
| Vy_sw, Vz_sw | MSH, MSP | float64 | km/s | GSM | Paired upstream flow, y/z | OMNI GSE rotated to GSM (non-round values) | INFERRED |
| Tp_sw | MSH, MSP | float64 | K | — | Paired upstream proton temperature | OMNI (integer K) | VERIFIED unit by magnitude |
| Pd_sw | MSH, MSP | float64 | nPa | — | Paired upstream dynamic pressure | OMNI `Pressure` = 2e-6·Np·V² (includes ~4 % He++) | VERIFIED: Pd/(2e-6·Np_sw·\|V_sw\|²) = 1.000 ± 0.002 |
| Beta_sw | MSH, MSP | float64 | — | — | Paired upstream plasma beta | OMNI `Beta` = [(4.16e-5 T + 5.34) Np]/B² (incl. electrons/He) | VERIFIED vs OMNI formula (ratio 0.97 ± 0.06); not proton-only beta (ratio ~4–5) |
| Ma_sw | MSH, MSP | float64 | — | — | Paired upstream Alfvén Mach number | OMNI `Mach_num` | INFERRED: Ma / (V/V_A,p) = 1.07 ± 0.04 (OMNI mass loading) |
| tilt | MSP | float64 | rad | — | Dipole tilt, >0 near June solstice | `spok.get_tilt`: (23.4°cos(2π(doy−172)/365.25) + 11.2°cos(2π(UT−16.72)/24)) — analytic approximation, not IGRF/geopack | VERIFIED: matches to 1e-16 rad |
| R_mp | MSH, MSP | float64 | R_E | radial from Earth along sat. direction | Magnetopause distance in the spacecraft's direction for paired SW | ML magnetopause model (plan doc); R_mp/Shue98 = 0.98 ± 0.05 | INFERRED; null for some paired MSP rows (reason UNKNOWN) |
| R_bs | MSH | float64 | R_E | as R_mp | Bow-shock distance in the spacecraft's direction | ML bow-shock model; R_bs/R_mp 1.17–1.84 | INFERRED |
| R_norm | MSH | float64 | — | — | Fractional position MP(0) → BS(1) | **(\|r_gsm\| − R_mp)/(R_bs − R_mp)** | VERIFIED to 1e-15; this *is* the `d_msh` quantity |
| R_norm | MSP | float64 | — | — | Fractional position Earth(0) → MP(1) | **\|r_gsm\| / R_mp** | VERIFIED to 1e-16; this *is* the `d_msp` quantity |
| Norma_pos | MSH, MSP | bool | — | — | True if the row has a normalized position | Requires SW_pairing and R_norm inside a band: observed MSH ≈ [−0.09, 1.08], MSP ≈ [0.75, 1.10] | VERIFIED: *_norm null ⇔ False; band edges sample-limited, criterion UNKNOWN |
| X_gsm_norm, Y_gsm_norm, Z_gsm_norm | MSH, MSP | float64 | R_E | GSM directions | Position remapped between fixed average boundaries | Same direction as r_gsm (angle < 2e-6°), radius rescaled; MSP: \|r_norm\| = R_norm·R_mp,ref(θ); MSH: R_mp,ref + R_norm·(R_bs,ref − R_mp,ref). Reference ≈ paraboloids (MP nose ~10.4, BS nose ~13.6 R_E), from Formisano average (`some_code/models_magnetic_field.py:374`) | Direction VERIFIED; reference surface INFERRED (paraboloid fit rms 0.3–0.4 R_E, not exact); not SM despite notebook name |
| Bx_swi, By_swi, Bz_swi | MSH | float64 | nT | SWI | Local B rotated into the SWI frame | `spok.coordinates.swi_base`: X = −V_sw/\|V_sw\|; Z ∥ X × sign(Bx_imf)·B_imf; Y = Z × X ⇒ IMF in X–Y plane, B_imf,y sign = sign(Bx_imf) ⇒ quasi-parallel shock on +Y_swi | VERIFIED exact on 132k rows, all spacecraft (2026-10-08): B_swi = sgn(Bx_imf)·R·B. The earlier "THD/THE/DS1/C3 inconsistent" finding came from omitting the sgn factor. |
| Vx_swi, Vy_swi, Vz_swi | MSH | float64 | km/s | SWI | Local ion velocity in SWI, aberration-corrected | R_swi · (V − 29.8 ŷ km/s) (Earth orbital motion removed) | VERIFIED exact: R·(V − 29.8 km/s ŷ_GSM). |
| X_swi_norm, Y_swi_norm, Z_swi_norm | MSH | float64 | R_E | SWI | Normalized position in SWI | R_swi · (X,Y,Z)_gsm_norm | Direction exact on all rows; radius re-normalized between mean Shue98/Jelinek2012 boundaries at SWI angles and clipped to [0,1] (notebook cell 23): differs on ~9 % of rows, all near a boundary. |

## Side findings (for the caller)

1. **Dead filters are a catalog bug, not a data bug.** `R_norm` already holds exactly the `d_msh`/`d_msp` quantities. With names frozen, the fix is `RANGE_FILTERS["d_msh"/"d_msp"].column = "R_norm"`, `tilt.column = "tilt"` — the 2026-07-26 plan's "emit D_msh/D_msp/Tilt in data" would rename frozen columns. The plan's separate "R_norm = Formisano-frame radius" is actually what `|r_gsm_norm|` is.
2. **SWI columns are consistent** (re-checked 2026-10-08 against MSH_GSM_to_SWI.ipynb on kaa): see the three SWI rows above.
3. `vx_sw` filter description says "bulk speed" but the column is signed and negative.
4. IMF columns are non-null on `SW_pairing == False` rows (MSP: all such sample rows).
5. Sample selection hints: MSH Np ≥ 3.0, MSP Np < 4.0 (min/max over 3000 rows), so region post-filters may use density thresholds.
