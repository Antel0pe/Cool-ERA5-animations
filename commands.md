cdo showname 2020_hourly_10mWind_2mTemp_sst.grib

cdo -f nc4 -z zip selname,2t 2020_hourly_10mWind_2mTemp_sst.grib t2m_2020.nc

curl -L --fail --progress-bar -C - \
  -o /home/dmmsp/Projects/rawEra5Data/era5_2020_250hPa_uv_hourly.grib \
  "https://object-store.os-api.cci2.ecmwf.int/cci2-prod-cache-1/2026-01-12/df058518f60c52d9d38acaa27db5bc81.grib"