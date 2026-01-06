# SWIVVEL: Score-Weighted Identification and Visualization of Vortex Evolution and Location

This repo contains the code for the vortex detection and tracking algorithm SWIVVEL, alongside a interactive visualization dashboard. The files include:

- `SWIVVEL.py`: a python file containing the core SWIVVEL pipeline
  
- `app.py`: a python file that creates the streamlit interactive dashboard that uses and visualizes the SWIVVEL data. To run this, use the command `python -m streamlit run app.py`

- `datasets`: the datasets used in this paper including testing and training sets, and ground truth sets for each. Raw wind vector data is from NOAA NCEI Blended Seawinds (NBS v2) dataset (https://coastwatch.noaa.gov/cwn/products/noaa-ncei-blended-seawinds-nbs-v2.html), while the 'ground truth' cyclone data was from the International Best Track Archive for Climate Stewardship (IBTrACS) dataset (https://ncics.org/ibtracs/)
