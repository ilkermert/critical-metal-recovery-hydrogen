# When Does Critical Metal Recovery Matter for the Cost of Hydrogen from Electrolysers?

**Recycling Scenarios, Variance Decomposition and Iridium Material Flow in PEM and Alkaline Systems**

**Author:** İlker Mert*

This repository contains methodological documentation, representative code excerpts and the numerical input data for the research article (in preparation for *Renewable and Sustainable Energy Reviews*, special issue "Embedding Circular Practices into Renewable and Sustainable Energy Production and Management").

Repository: [repository URL]

## About the Study

The study asks when the recovery of iridium, platinum and nickel changes the levelised cost of hydrogen (LCOH) of PEM and alkaline electrolysers. Recovery scenarios are added to an LCOH model whose system CAPEX term is calibrated to the European Hydrogen Observatory (EHO) cost components. Uncertainty is propagated with a Latin hypercube Monte Carlo simulation of 65,536 draws per case. The variance is decomposed with block Sobol indices. The iridium material flow to 2050 is modelled separately. The literature context comes from a Web of Science screening (search of 21 November 2026, 1,038 records, 108 included).

The recovery credit is a gross credit. Process costs of recovery are not modelled. The LCOH result and the material-flow result use separate system boundaries. The 2050 flows are conditional scenarios, not forecasts. The iridium loading of 0.05 g/kW is a target, not an observed value. Calibration variants A and B are two calibration choices. Neither is a bound.

| Stage | Content |
|---|---|
| model | LCOH, recovery credit, calibration variants A and B, gross end-of-life break-even |
| sampling | Latin hypercube draws, triangular and uniform inputs, price paths (random walk and OU), Ir–Pt Gaussian copula |
| montecarlo | Mean, standard deviation and percentiles of LCOH with recovery off and on, five countries |
| sobol | Block Sobol indices, price-group shares, stability and bootstrap intervals |
| flow | Iridium demand, retirement, secondary supply and primary requirement to 2050 |
| screening | PRISMA flow and literature profile (keyword stages, 1,038 records) |
| validation | DOE capacity series and Ibrahim and Eggert (2026) reproduction |

## 1. Input Distributions and Sampling

Inputs are uniform or triangular (minimum, mode, maximum) as listed in the Parameters table below. Draws are Latin hypercube samples with a fixed seed. The shock quantiles ζ are shared between countries (common random numbers), so country differences do not come from sampling noise. Ir and Pt price shocks are independent in the base case. Dependence is examined in labelled scenarios with a Gaussian copula (ρ = 0.3, 0.6, 0.9). These ρ values are scenario parameters, not estimates.

$$z_{Pt} = \rho\,\Phi^{-1}(\zeta_{Ir}) + \sqrt{1-\rho^{2}}\,z_{Pt,0}$$

```python
def ppf(row, u):
    if row.dist == "U":
        return row.min + (row.max - row.min) * u
    cc = (row["mode"] - row["min"]) / (row["max"] - row["min"])
    return triang.ppf(u, cc, loc=row["min"], scale=row["max"] - row["min"])
```

## 2. LCOH Model and Recovery Credit

$$\mathrm{CRF}(r,N)=\frac{r(1+r)^{N}}{(1+r)^{N}-1}$$

$$\mathrm{LCOH}=\psi_j\,\mathrm{CRF}(r,N)\,K_j\,\nu\,\frac{\varepsilon_j}{t_{op}}+\mathrm{CRF}(r,N)\,\frac{\Delta M}{\chi}\,\frac{\varepsilon_j}{t_{op}}+(C_E+C_G+C_X)\frac{\varepsilon_j}{\varepsilon_{ref}}+C_O$$

$$\Delta M=\sum_m\left[I_m P_m f_m - I_{m,0}P_{m,0}\right],\qquad f_m = 1-C_{col}\,\alpha_m\,(1+r)^{-(N+\delta)}$$

Here, $K_j$ is the system CAPEX in EUR/kW, $\nu$ a CAPEX multiplier, $\varepsilon_j$ the specific electricity use in kWh/kg, $t_{op}$ the operating hours, $\chi$ the EUR–USD rate, $I_m$ the metal intensity in kg/kW, $P_m$ the metal price, $C_{col}$ the collection rate, $\alpha_m$ the recovery efficiency and $\delta$ the delay in years. The factor $\psi$ is applied only to the system CAPEX term. The metal term is not calibrated. With recovery off, $f_m=1$.

Calibration variant A fixes $\psi$ at the EHO central value. Variant B recomputes $\psi$ in every draw so that the system CAPEX component before $\nu$ equals the EHO CAPEX component $C_{CAP}$.

$$\psi_{A,j}=\frac{C_{CAP}}{K_j\,\mathrm{CRF}(r_{EHO},N_{EHO})\,\varepsilon_{c,j}/t_{op,c}},\qquad \psi_{B,j,d}=\frac{C_{CAP}}{K_j\,\mathrm{CRF}(r_d,N_d)\,\varepsilon_{j,d}/t_{op,d}}$$

The gross end-of-life break-even value is the metal value at which the recovery credit offsets a given LCOH change $G$.

$$G=\mathrm{LCOH}_{off}-\mathrm{LCOH}_{on},\qquad V_{eol}=\frac{G\,(t_{op}/\varepsilon_j)}{\mathrm{CRF}(r,N)\,(1+r)^{-(N+\delta)}}$$

```python
def _assemble(X, tech, country, dM, cal=None):
    eps = X["eps"] + (EPS_PEM_D if tech == "PEM" else 0.0)
    kg_per_kw = X["t_op"] / eps
    crf_i = crf(X["r"], X["N"])
    psi_i = 1.0 if cal is None else (cal[tech] if CAL_MODE == "A" else CAP_COMP / (K_TOT[tech] * crf_i / kg_per_kw))
    capex_term = psi_i * K_TOT[tech] * X["nu"] * crf_i / kg_per_kw + (dM / X["chi"]) * crf_i / kg_per_kw
    e = EHO.loc[country]
    return capex_term + (e.C_E + e.C_G + e.C_X) * (eps / EPS_REF) + e.C_O

# inside lcoh(): recovery credit factor per metal
f_m = 1.0 - X["C_col"] * X[REC_ALPHA[m]] * (1.0 + X["r"]) ** (-(X["N"] + X["delta"])) if rec else 1.0
dM = dM + i_t * price * f_m - i0 * P0[m]
```

## 3. Metal Price Paths

Metal prices follow a lognormal path with a drift and a horizon-specific volatility. The random walk (RW) and the Ornstein–Uhlenbeck (OU) process differ only in the variance term. OU is described as a deviation from the drift path. The mean-reversion speed κ is estimated from the IMF nickel series and is assumed equal for Ir and Pt.

$$\ln P_m=\ln P_{m,0}+T\ln(1+g_m)-\frac{\sigma_\tau^{2}}{2}+\sigma_\tau z$$

$$\sigma_\tau=\sigma_m\sqrt{\tau}\ \ \text{(RW)},\qquad \sigma_\tau=\sigma_m\sqrt{\frac{1-e^{-2\kappa\tau}}{2\kappa}}\ \ \text{(OU)}$$

```python
if sc["proc"] == "ou" and tau > 1.0:
    kap = X[f"kap_{m}"]
    sg = vol * np.sqrt((1.0 - np.exp(-2.0 * kap * tau)) / (2.0 * kap))
else:
    sg = vol * np.sqrt(tau)
price = np.exp(np.log(P0[m]) + T * np.log1p(g) - 0.5 * sg ** 2 + sg * z)
```

## 4. Sensitivity Analysis

The variance of LCOH is decomposed with Sobol indices. The estimators are Saltelli (2010) for first-order (closed) and Jansen (1999) for total-order indices. Scrambled Sobol quasi-Monte Carlo is used with a base sample of 2^13 and 200 bootstrap replicates. The bootstrap resamples row indices with replacement, shared across the A, B and hybrid matrices. Intervals are 95% percentile intervals. The sum of first-order indices may deviate slightly from 1 because of sampling error. Price-group shares use eight designs with base sample 2^16 and report the median.

$$S_i=\frac{\mathrm{Var}[E(Y\mid X_i)]}{\mathrm{Var}(Y)},\qquad S_{T,i}=\frac{E[\mathrm{Var}(Y\mid X_{\sim i})]}{\mathrm{Var}(Y)}$$

Block indices treat the Ir and Pt price shocks as one block, because they can be dependent.

$$S_B=\frac{\mathrm{Var}(E[Y\mid\zeta_{Ir},\zeta_{Pt}])}{\mathrm{Var}(Y)},\qquad S_{T,B}=\frac{E[\mathrm{Var}(Y\mid X_{\sim B})]}{\mathrm{Var}(Y)}$$

```python
M = qmc.Sobol(d=2 * k, scramble=True, seed=seed).random_base2(m)
A, B = M[:, :k], M[:, k:]
for i in range(k):
    ABi = A.copy(); ABi[:, i] = B[:, i]; yAB.append(ev(ABi))
var = np.var(np.concatenate([yA, yB]), ddof=1)
S1 = np.mean(yB * (yAB[i] - yA)) / var
ST = 0.5 * np.mean((yA - yAB[i]) ** 2) / var
```

## 5. Iridium Material Flow

Annual iridium demand follows the PEM capacity additions $Q_t$ (DOE series) and the loading $\omega_{Ir}$ in g/kW. Retirements follow a discrete triangular lifetime distribution over integer ages 10 to 20 (mode 15, weights 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1 divided by 36). The alternative lifetime is shifted by five years (ages 15 to 25, mode 20). Secondary supply is delayed by $\delta$ years and primary requirement is demand minus secondary supply.

$$D_t=\omega_{Ir}Q_t,\qquad R_t=\sum_{a=10}^{20}\pi_a Q_{t-a},\qquad Z_t=C_{col}\,\alpha_{Ir}\,\omega_{Ir}\,R_{t-\delta},\qquad B_t=\max(D_t-Z_t,0)$$

```python
lw = D["Ibrahim_S1_Lifetime"]
ages = lw["age"].astype(int).values
w = lw["raw_weight"].astype(float).values; w = w / w.sum()
Rt = lambda t: sum(pa * float(Q.get(t - a, 0.0)) for a, pa in zip(ages, w))
sec = C * al * om * Rt(2050 - d)
```

## Parameters

Uncertain inputs. U = uniform (min, max), T = triangular (min, mode, max).

| Parameter | Symbol | Unit | Dist. | Minimum / mode / maximum | Central |
|---|---|---|---|---|---|
| Discount rate | r | - | U | 0.06 / - / 0.10 | 0.06 |
| Economic lifetime | N | years | U | 15 / - / 25 | 25 |
| Operating hours | t_op | h/yr | U | 3500 / - / 4000 | 4000 |
| CAPEX multiplier | ν | - | U | 0.877 / - / 1 | 1 |
| Electricity use | ε | kWh/kg | T | 52.4 / 54.3 / 55.2 | 54.3 |
| Exchange rate | χ | USD/EUR | U | 1.02 / - / 1.18 | 1.10 |
| Ir intensity | I_Ir | kg/kW | T | 0.00034 / 0.00075 / 0.0025 | 0.00075 |
| Pt intensity | I_Pt | kg/kW | T | 0.000333 / 0.0005 / 0.001 | 0.0005 |
| Ni intensity | I_Ni | kg/kW | T | 0.8 / 0.8 / 2.78 | 0.8 |
| Ir volatility | σ_Ir | - | T | 0.075 / 0.35 / 0.536 | 0.35 |
| Pt volatility | σ_Pt | - | U | 0.10 / - / 0.20 | 0.155 |
| Ni volatility | σ_Ni | - | U | 0.241 / - / 0.314 | 0.29 |
| Ir drift | g_Ir | 1/yr | T | -0.02 / 0.03 / 0.0809 | 0.03 |
| Pt drift | g_Pt | 1/yr | U | -0.02 / - / 0.03 | 0 |
| Ni drift | g_Ni | 1/yr | T | -0.03 / 0 / 0.0333 | 0 |
| OU reversion speed (Ni, Ir, Pt) | κ | 1/yr | U | 0.082 / - / 0.702 | 0.16 |

Recovery parameters:

| Parameter | Symbol | Dist. | Minimum / mode / maximum | Central |
|---|---|---|---|---|
| Collection rate | C_col | U | 0.5 / - / 1 | 0.7 |
| Ir recovery efficiency | α_Ir | U | 0.4 / - / 0.97 | 0.5 |
| Pt recovery efficiency | α_Pt | T | 0.64 / 0.95 / 0.99 | 0.95 |
| Ni recovery efficiency | α_Ni | T | 0.57 / 0.92 / 0.97 | 0.92 |
| Delay | δ | U | 0 / - / 3 | 2 |

Named recovery scenarios:

| Scenario | C_col | α_Ir | α_Pt | α_Ni | δ (years) |
|---|---|---|---|---|---|
| Low | 0.5 | 0.4 | 0.64 | 0.57 | 3 |
| Central | 0.7 | 0.5 | 0.95 | 0.92 | 2 |
| High | 1 | 0.97 | 0.99 | 0.97 | 0 |

Fixed constants:

| Symbol | Meaning | Value |
|---|---|---|
| C_CAP | EHO CAPEX component (2023 and 2024 files) | 2.761182 EUR/kg |
| r_EHO, N_EHO | EHO calculator defaults | 0.06, 25 years |
| ε_ref | EHO reference electricity use | 54.3 kWh/kg |
| Δε_PEM | PEM electricity use above AEL | 0.9 kWh/kg |
| I_Ir,2035, I_Pt,2035 | Progressive intensities | 5.0e-05, 3.75e-05 kg/kW |
| Ni progressive factor | Ni intensity reduction | 0.5 |

Settings:

| Symbol | Meaning | Value |
|---|---|---|
| N | Latin hypercube draws per case | 65,536 |
| M | Sobol base sample | 2^13 |
| N_boot | Bootstrap replicates | 200 |
| ρ | Ir–Pt copula scenarios | 0.3, 0.6, 0.9 |
| s_0 | Random seed | 20261017 |
| s_PG | Price-group seeds | 20261017 + 1000 + r, r = 0 to 7 |

## Data Sources

- **EHO cost components and electrolyser CAPEX:** European Hydrogen Observatory, hydrogen production costs 2024, grid electrolysis (open data), and the EHO calculator manual.
- **Metal prices:** USGS (Ir 2020 to 2023, Pt and Ni 2021 to 2025), IMF nickel series. Iridium 2024 and 2025 values come from the author's code and are consistent with the USGS platinum-group metals summary.
- **Recovery and material parameters:** Ibrahim and Eggert (2026), Wortmann et al., Uekert et al., Hoppe and Minke, UNEP (2011), Eikeng et al., Kiemel et al., IRENA. Source pages are given per parameter in the article.
- **Capacity series:** US DOE.
- **Trade data:** UN Comtrade.
- **Literature:** Web of Science Core Collection, search of 21 November 2026. The search string is in Supplementary Note S2.

## Input Data

The numerical inputs of the study are shown below. The values are those used for the results of the article.

EHO cost components, 2024 (EUR/kg H2, grid electrolysis):

| Country | C_E | C_G | C_X | C_O |
|---|---|---|---|---|
| Spain | 1.0662 | 0.5409 | 0.4814 | 0.1874 |
| France | 1.1788 | 0.8493 | 0.3191 | 0.1874 |
| Belgium | 1.9500 | 0.5031 | 0.9683 | 0.1874 |
| Germany | 2.2505 | 2.1475 | 1.6390 | 0.1874 |
| Italy | 4.6158 | 0.7843 | 1.4226 | 0.1874 |

System CAPEX, 2024 (EUR/kW):

| Technology | Stack | Balance of plant | Other utilities | Other CAPEX | Total K |
|---|---|---|---|---|---|
| AEL | 323 | 693 | 693 | 601 | 2,310 |
| PEM | 563 | 647 | 693 | 601 | 2,504 |

Reference metal prices:

| Year | Ir (USD/oz) | Pt (USD/oz) | Ni (USD/t) |
|---|---|---|---|
| 2020 | 1,633.51 | - | - |
| 2021 | 5,158.40 | 1,094.31 | 18,476 |
| 2022 | 4,581.93 | 966.54 | 25,815 |
| 2023 | 4,672.78 | 973.00 | 21,495 |
| 2024 | 4,810.40 | 960.70 | 16,812 |
| 2025 | 4,400.00 | 1,200.00 | 15,000 |

Price scenarios:

| Scenario | Horizon τ (years) | Process | Intensity |
|---|---|---|---|
| 2025 | 1 | RW | constant |
| 2050_RW_C | 25 | RW | constant |
| 2050_RW_P | 25 | RW | progressive |
| 2050_OU_C | 25 | OU | constant |
| 2050_OU_P | 25 | OU | progressive |

## Software Environment

| Package | Version | Role |
|---|---|---|
| Python | 3.13 | Core implementation |
| NumPy | 2.5 | Array computation and random-number generation |
| SciPy | 1.18 | Distributions and scrambled Sobol sequences |
| pandas | 3.0 | Data handling |
| statsmodels | 0.15 | Regression and statistical tests |
| Matplotlib | 3.11 | Figure generation |
| openpyxl | 3.1 | Workbook input and output |

## How to Read the Results

Headline results of the paper (seed 20261017):

- 2025 PEM, variant A minus variant B: 1.035 EUR/kg.
- 2025 PEM recovery credit: 0.0282 EUR/kg. Central maximum credit: 0.0449 EUR/kg.
- 2050 PEM central credit (OU): 0.0620 EUR/kg. Alkaline maximum credit: 0.0039 EUR/kg.
- PEM recovery Sobol total-order index (B): 0.0024.
- 2050 primary iridium without recovery: 209.5 t/yr. High recovery: 101.2 t/yr. Alternative lifetime, High: 132.4 t/yr.
- 2025 PEM gross end-of-life break-even: 43.3 EUR/kW.

Results are conditional on the input ranges, which are scenario ranges and not forecasts. Prices are at the price level of the source publications, without inflation harmonisation. Title-and-abstract screening decisions were made by one evaluator. Raw Web of Science records are not redistributed (Clarivate terms of use). Third-party data keep their original terms.

## Citation

If you use this code or methodology, please cite:

Mert, İ. (2026). When Does Critical Metal Recovery Matter for the Cost of Hydrogen from Electrolysers? Recycling Scenarios, Variance Decomposition and Iridium Material Flow in PEM and Alkaline Systems. *Renewable and Sustainable Energy Reviews* (in preparation).

## License

Released under the MIT License (see LICENSE).

## Contact and Data Availability

The numerical input data of the study are given in the tables of this README, and representative code excerpts of the core methodological steps are given in the method sections above. The complete analysis script and the consolidated input dataset are not shared at this time because the same code base is being used in other ongoing studies. They are available from the corresponding author on reasonable request. Summary results of all analyses are provided in the Supplementary Material of the paper.

Maintainer: İlker Mert ([e-mail])

Note on scope. The code blocks above are representative excerpts of the core methodological steps, included for transparency. This repository does not allow all tables and figures of the paper to be reproduced by running a script.
