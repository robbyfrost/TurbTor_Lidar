import numpy as np
from netCDF4 import Dataset
import datetime
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
fhalo   = "/Users/robbyfrost/Documents/MS_Project/data/20250425/"
dlid    = "dltruckdlcsmDL1.b1.20250425_reproc.cdf"
dhea    = "LIDR_TRUCK_20250425_MM.nc"
out_dir = fhalo + "cfradial/"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# ── READ DATA ─────────────────────────────────────────────────────────────────
f    = Dataset(fhalo + dlid)
vr   = f.variables['velocity'][:]
snum = f.variables['snum'][:]
az   = f.variables['azimuth'][:]
el   = f.variables['elevation'][:]
hour = f.variables['hour'][:]
r    = f.variables['range'][:]
time = f.variables['base_time'][0] + f.variables['time_offset'][:]
f.close()

# # midnight rollover
# day_offset   = np.zeros(len(time), dtype=np.float64)
# seconds_in_day = 86400.0
# rollover_indices = np.where(np.diff(hour) < -12)[0]
# for idx in rollover_indices:
#     day_offset[idx + 1:] += seconds_in_day
# time = time + day_offset

m         = Dataset(fhalo + dhea)
m_heading = m.variables['gps_dir'][:]
c_heading = m.variables['compass_dir'][:]
mtime     = m.variables['epochtime'][:]
lat_arr   = m.variables['lat'][:]
lon_arr   = m.variables['lon'][:]
m.close()

# ── MASK LONG RANGES ─────────────────────────────────────────────────────────
vr[:, np.where(r > 8)[0]] = np.nan

unique_snum = np.unique(snum)


# ── HELPER: split one scan into up to two sub-sweeps ─────────────────────────
def split_sweep(scan_az, scan_el, scan_hr, scan_vr, scan_time):
    """
    Replicates the direction-reversal logic and returns a list of
    dicts (one per sub-sweep) with keys: az, el, hr, vr, time.
    """
    az_1, el_1, hr_1, vr_1, t_1 = [], [], [], [], []
    az_2, el_2, hr_2, vr_2, t_2 = [], [], [], [], []

    sign   = 0
    switch = False

    for j in range(1, len(scan_az)):
        direction = np.sign(scan_az[j] - scan_az[j - 1])

        if j == 1:
            sign = np.copy(direction)
            az_1.append(scan_az[j - 1]); el_1.append(scan_el[j - 1])
            hr_1.append(scan_hr[j - 1]); vr_1.append(scan_vr[j - 1, :])
            t_1.append(scan_time[j - 1])
        else:
            if (sign == direction) and not switch:
                az_1.append(scan_az[j - 1]); el_1.append(scan_el[j - 1])
                hr_1.append(scan_hr[j - 1]); vr_1.append(scan_vr[j - 1, :])
                t_1.append(scan_time[j - 1])
            elif (sign == direction) and switch:
                az_2.append(scan_az[j - 1]); el_2.append(scan_el[j - 1])
                hr_2.append(scan_hr[j - 1]); vr_2.append(scan_vr[j - 1, :])
                t_2.append(scan_time[j - 1])
            else:
                if not switch:
                    switch = True
                    sign   = np.copy(direction) if abs(direction) > 0.1 else sign * -1
                    az_1.append(scan_az[j - 1]); el_1.append(scan_el[j - 1])
                    hr_1.append(scan_hr[j - 1]); vr_1.append(scan_vr[j - 1, :])
                    t_1.append(scan_time[j - 1])
                else:
                    if scan_az[j] - scan_az[j - 1] == 0:
                        continue
                    else:
                        break

    sweeps = []
    for az_s, el_s, hr_s, vr_s, t_s in [
        (az_1, el_1, hr_1, vr_1, t_1),
        (az_2, el_2, hr_2, vr_2, t_2),
    ]:
        if len(az_s) == 0:
            continue
        sweeps.append(dict(
            az   = np.array(az_s),
            el   = np.array(el_s),
            hr   = np.array(hr_s),
            vr   = np.array(vr_s),
            time = np.array(t_s),
        ))
    return sweeps


# ── HELPER: write one sweep to CfRadial netCDF ───────────────────────────────
def write_cfradial(sweep, r, lidar_lat, lidar_lon, lidar_alt,
                   sweep_number, out_dir):
    """
    Writes a single PPI sweep to a CfRadial-1.4 compliant netCDF file.
    sweep : dict with keys az, el, hr, vr, time
    """
    az_s   = sweep['az']
    el_s   = sweep['el']
    vr_s   = sweep['vr']
    time_s = sweep['time']

    nrays  = len(az_s)
    ngates = len(r)

    # epoch reference: seconds since 1970-01-01
    epoch_ref = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
    t0        = datetime.datetime.fromtimestamp(float(time_s[0]),
                                               tz=datetime.timezone.utc)
    t_str     = t0.strftime("%Y%m%d_%H%M%S")
    fname     = f"NSSLDWL_swp{sweep_number:04d}_{t_str}.nc"
    fpath     = os.path.join(out_dir, fname)

    with Dataset(fpath, "w", format="NETCDF4") as nc:

        # ── Global attributes (CfRadial 1.4) ──────────────────────────────
        nc.Conventions           = "CF/Radial instrument_parameters"
        nc.version               = "1.4"
        nc.title                 = "HALO Doppler lidar sweep"
        nc.institution           = ""
        nc.references            = ""
        nc.source                = "HALO StreamLine Doppler Wind Lidar"
        nc.history               = f"Created {datetime.datetime.utcnow().isoformat()}"
        nc.comment               = ""
        nc.instrument_name       = "HALO_DWL_TRUCK"
        nc.platform_type         = "moving"       # truck-mounted
        nc.primary_axis          = "axis_z"
        nc.time_coverage_start   = t0.isoformat()
        nc.time_coverage_end     = datetime.datetime.fromtimestamp(
                                       float(time_s[-1]),
                                       tz=datetime.timezone.utc).isoformat()
        nc.sweep_mode            = "azimuth_surveillance"  # PPI
        nc.scan_type             = "ppi"

        # ── Dimensions ────────────────────────────────────────────────────
        nc.createDimension("time",       nrays)
        nc.createDimension("range",      ngates)
        nc.createDimension("sweep",      1)
        nc.createDimension("string_length", 32)

        # ── Coordinate: time ──────────────────────────────────────────────
        vtime             = nc.createVariable("time", "f8", ("time",))
        vtime.standard_name = "time"
        vtime.units         = "seconds since 1970-01-01T00:00:00Z"
        vtime.calendar      = "gregorian"
        vtime[:]            = time_s.astype("f8")

        # ── Coordinate: range ─────────────────────────────────────────────
        vrange              = nc.createVariable("range", "f4", ("range",))
        vrange.standard_name = "projection_range_coordinate"
        vrange.long_name     = "range_to_measurement_volume"
        vrange.units         = "km"
        vrange.spacing_is_constant = "true"
        vrange.meters_to_center_of_first_gate = float(r[0]) * 1000.0
        vrange.meters_between_gates           = float(r[1] - r[0]) * 1000.0
        vrange[:]            = r.astype("f4")

        # ── Ray geometry ──────────────────────────────────────────────────
        vaz               = nc.createVariable("azimuth", "f4", ("time",))
        vaz.standard_name = "ray_azimuth_angle"
        vaz.units         = "degrees"
        vaz[:]            = az_s.astype("f4")

        vel               = nc.createVariable("elevation", "f4", ("time",))
        vel.standard_name = "ray_elevation_angle"
        vel.units         = "degrees"
        vel[:]            = el_s.astype("f4")

        # ── Sweep variables ───────────────────────────────────────────────
        vsn               = nc.createVariable("sweep_number", "i4", ("sweep",))
        vsn.units         = "count"
        vsn[:]            = [sweep_number]

        vsm               = nc.createVariable("sweep_mode", "S1",
                                              ("sweep", "string_length"))
        mode_str          = "azimuth_surveillance"
        arr               = np.array(list(mode_str.ljust(32)), dtype="S1")
        vsm[0, :]         = arr

        vfe               = nc.createVariable("fixed_angle", "f4", ("sweep",))
        vfe.units         = "degrees"
        vfe[:]            = [float(np.nanmean(el_s))]

        vsr               = nc.createVariable("sweep_start_ray_index", "i4", ("sweep",))
        vsr[:]            = [0]

        ver               = nc.createVariable("sweep_end_ray_index",   "i4", ("sweep",))
        ver[:]            = [nrays - 1]

        # ── Platform position ─────────────────────────────────────────────
        vlat              = nc.createVariable("latitude",  "f8", ())
        vlat.units        = "degrees_north"
        vlat[:]           = lidar_lat

        vlon              = nc.createVariable("longitude", "f8", ())
        vlon.units        = "degrees_east"
        vlon[:]           = lidar_lon

        valt              = nc.createVariable("altitude",  "f8", ())
        valt.units        = "meters"
        valt[:]           = lidar_alt          # set to 0.0 or truck height

        # ── Radial velocity field ─────────────────────────────────────────
        fill = np.float32(1e36)
        vvr               = nc.createVariable("vr",
                                              "f4", ("time", "range"),
                                              fill_value=fill)
        vvr.standard_name = "Radial Velocity"
        vvr.long_name     = "Radial velocity of scatterers away from instrument"
        vvr.units         = "m/s"
        vvr.coordinates   = "time range"
        data              = vr_s.astype("f4")
        data[np.isnan(data)] = fill
        vvr[:]            = data

    print(f"  Written: {fname}")
    return fpath


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
sweep_counter = 0

for i in unique_snum:
    foo        = np.where(i == snum)
    scan_vr    = vr[foo[0], :]
    scan_az    = az[foo]
    scan_el    = el[foo]
    scan_hr    = hour[foo]
    scan_time  = time[foo]

    sort_idx   = np.argsort(scan_hr)
    scan_az    = scan_az[sort_idx]
    scan_el    = scan_el[sort_idx]
    scan_hr    = scan_hr[sort_idx]
    scan_vr    = scan_vr[sort_idx, :]
    scan_time  = scan_time[sort_idx]

    # heading correction
    foo_m       = np.where((scan_time[0] <= mtime) & (scan_time[-1] >= mtime))
    compass_mean = np.nanmean(c_heading[foo_m])
    scan_lat    = np.nanmean(lat_arr[foo_m])
    scan_lon    = np.nanmean(lon_arr[foo_m])

    scan_az_corrected = (scan_az + compass_mean) % 360

    # split into sub-sweeps
    sweeps = split_sweep(scan_az_corrected, scan_el, scan_hr,
                         scan_vr, scan_time)

    for sw in sweeps:
        write_cfradial(
            sweep        = sw,
            r            = r,
            lidar_lat    = scan_lat,
            lidar_lon    = scan_lon,
            lidar_alt    = 0.0,        # adjust if you have altitude
            sweep_number = sweep_counter,
            out_dir      = out_dir,
        )
        sweep_counter += 1

print(f"\nDone. {sweep_counter} sweeps written to {out_dir}")