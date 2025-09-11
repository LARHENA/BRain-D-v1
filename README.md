# BRain-D-v1


The BRain-D (BRazilian Daily) dataset provides high-resolution gridded daily rainfall data for Brazil, spanning the period from 1961 to 2024. It is based on records from 18,370 rain gauges distributed across multiple national hydrometeorological networks. The dataset is presented on a regular grid with a spatial resolution of 0.1° × 0.1°, offering extensive spatial coverage and temporal continuity.

A key feature of BRain-D is its reliance on a systematic automatic quality control procedure (A-QCP), which annually classifies stations as High-Quality (HQ) or Low-Quality (LQ), ensuring that only reliable data are included in the gridding process. This approach enhances the accuracy and consistency of the final dataset.

Following quality control, the remaining HQ data were spatially interpolated to generate the gridded fields. The dataset was validated using cross-validation techniques and compared against an independent reference from the Brazilian National Institute of Meteorology (INMET), demonstrating strong agreement in both daily values and long-term climatological normals.

BRain-D is suitable for a wide range of climatological and hydrological applications, particularly those requiring reliable high-resolution daily precipitation data over Brazil.


This repository contains a complete data processing pipeline that:
- Cleans and filters raw rainfall data
- Performs outlier detection using adjacent day and neighboring analysis
- Calculates comprehensive quality metrics (P, Q1, Q2, Q3)
- Generates a final quality index for each rain gauge
- Produces publication-ready visualizations


Citations

Related Publication:
Vidal-Barbosa, J. L. (2024). Brazilian daily rainfall gridded data from a quality controlled dataset (p. 49) [Master’s Thesis, Universidade Federal da Paraíba]. UFPB Repositório Institucional. https://repositorio.ufpb.br/jspui/handle/123456789/33310

Data:
Vidal-Barbosa, J. L., Lemos, F., Freitas, E. da S., Coelho, V. H. R., Souza da Silva, G. N., Patriota, E. G., Claudino, C. M. de A., Meira, M. A., Fullhart, A., Bertrand, G., De Souza, S. A., Rampinelleg, C. G., Almeida, C. D. N., & Estévez Gualda, J. (2025). BRain-D: Gridded Daily Rainfall Dataset for Brazil (Versão 1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15468235



Metadata

Authors:

Vidal-Barbosa, José Lindemberg (Project leader)

Lemos, Filipe (Researcher)

Freitas, Emerson da Silva (Researcher)

Coelho, Victor Hugo R. (Supervisor)

Souza da Silva, Gerald Norbert (Researcher)

Patriota, Eduardo Gonçalves (Researcher)

Claudino, Cinthia Maria de Abreu (Researcher)

Meira, Marcela Antunes (Researcher)

Fullhart, Andrew (Project member)

Bertrand, Guillaume (Researcher)

De Souza, Saulo Aires (Project member)

Rampinelle, Cássio Guilherme (Project member)

Estévez Gualda, Javier (Project member)

Almeida, Cristiano Das Neves (Supervisor)


Publication Date: 2025

Location: Brazil

Institution: Universidade Federal da Paraíba (UFPB)
