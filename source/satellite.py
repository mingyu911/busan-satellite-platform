import xarray as xr
import numpy as np
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
    
def _mask_fill_valid(arr, attrs):
    """FillValue 및 valid_min/max 기준으로 배열을 NaN 처리."""
    arr = arr.astype(float)
    fill = attrs.get('_FillValue')
    v_min = attrs.get('valid_min')
    v_max = attrs.get('valid_max')

    if fill is not None:
        arr[arr == fill] = np.nan
    if v_min is not None and v_max is not None:
        arr[(arr < v_min) | (arr > v_max)] = np.nan

    return arr

def SP5_TROPOMI_CH4():
    file_path = (
        os.path.join(ROOT_DIR, "data", "CH4", "SP5_TROPOMI", "20240313")
        + os.sep
        + "S5P_OFFL_L2__CH4____20240313T033717_20240313T051847_"
        + "33238_03_020600_20240314T194743.nc"
    )

    try:
        ds = xr.open_dataset(file_path, group="PRODUCT")

        ch4_var = ds["methane_mixing_ratio"]
        lat_var = ds["latitude"]
        lon_var = ds["longitude"]

        data = _mask_fill_valid(ch4_var.values, ch4_var.attrs)[0]
        lat  = _mask_fill_valid(lat_var.values, lat_var.attrs)[0]
        lon  = _mask_fill_valid(lon_var.values, lon_var.attrs)[0]

        return data, lat, lon

    except Exception as e:
        print(f"CH4 데이터 로드 실패: {e}")
        return None, None, None

def S2_MSI_Chl():
    file_path = (
        os.path.join(ROOT_DIR, "data", "Chl-a", "S2_MSI", "20250309")
        + os.sep
        + "S2C_MSI_2025_03_09_02_18_19_T52SDD_L2W.nc"
    )

    try:
        ds = xr.open_dataset(file_path)

        chl_var = ds["chl_oc3"]
        lat_var = ds["lat"]
        lon_var = ds["lon"]

        data = _mask_fill_valid(chl_var.values, chl_var.attrs)
        lat  = _mask_fill_valid(lat_var.values, lat_var.attrs)
        lon  = _mask_fill_valid(lon_var.values, lon_var.attrs)

        return data, lat, lon

    except Exception as e:
        print(f"Chl-a 데이터 로드 실패: {e}")
        return None, None, None

def GK2B_GOCI2_Chl():
    file_path = (
        os.path.join(ROOT_DIR, "data", "Chl-a", "GK2B_GOCI2", "20250309")
        + os.sep
        + "GK2B_GOCI2_L2_20250309_021530_LA_S007_Chl.nc"
    )

    try:
        dn = xr.open_dataset(file_path, group="navigation_data")
        dg = xr.open_dataset(file_path, group="geophysical_data")

        chl_var = dg["Chl"]
        lat_var = dn["latitude"]
        lon_var = dn["longitude"]

        data = _mask_fill_valid(chl_var.values, chl_var.attrs)
        lat  = _mask_fill_valid(lat_var.values, lat_var.attrs)
        lon  = _mask_fill_valid(lon_var.values, lon_var.attrs)

        return data, lat, lon

    except Exception as e:
        print(f"Chl-a 데이터 로드 실패: {e}")
        return None, None, None
    
def GK2B_GEMS_AOD():
    file_path = (
        os.path.join(ROOT_DIR, "data", "AOD", "GK2B_GEMS", "20250324")
        + os.sep
        + "GK2_GEMS_L2_20250324_0245_AERAOD_FC_DPRO_ORI.nc"
    )

    try:
        dd = xr.open_dataset(file_path, group="Data Fields")
        dg = xr.open_dataset(file_path, group="Geolocation Fields")

        aod_var = dd["FinalAerosolOpticalDepth"]
        lat_var = dg["Latitude"]
        lon_var = dg["Longitude"]

        # (nwavel=3, spatial=2048, image=695) → 파장 인덱스 1번 선택
        data = _mask_fill_valid(aod_var.values, aod_var.attrs)[1]
        lat  = _mask_fill_valid(lat_var.values, lat_var.attrs)
        lon  = _mask_fill_valid(lon_var.values, lon_var.attrs)

        # shape 확인 출력 (디버깅용)
        print(f"AOD data shape: {data.shape}, lat shape: {lat.shape}, lon shape: {lon.shape}")

        return data, lat, lon

    except Exception as e:
        print(f"AOD 데이터 로드 실패: {e}")
        return None, None, None
    
def GK2B_GEMS_NO2():
    file_path = (
        os.path.join(ROOT_DIR, "data", "NO2", "GK2B_GEMS", "20250331")
        + os.sep
        + "GK2_GEMS_L2_20250331_0045_NO2_FC-ETC_DPRO_ORI.nc"
    )

    try:
        dd = xr.open_dataset(file_path, group="Data Fields")
        dg = xr.open_dataset(file_path, group="Geolocation Fields")

        no2_var = dd["ColumnAmountNO2Trop"]
        lat_var = dg["Latitude"]
        lon_var = dg["Longitude"]

        # (nwavel=3, spatial=2048, image=695) → 파장 인덱스 1번 선택
        data = _mask_fill_valid(no2_var.values, no2_var.attrs) / 2.687e16  # 단위 변환 (molec/cm² → DU)
        lat  = _mask_fill_valid(lat_var.values, lat_var.attrs)
        lon  = _mask_fill_valid(lon_var.values, lon_var.attrs)

        # shape 확인 출력 (디버깅용)
        print(f"NO2 data shape: {data.shape}, lat shape: {lat.shape}, lon shape: {lon.shape}")

        return data, lat, lon

    except Exception as e:
        print(f"NO2 데이터 로드 실패: {e}")
        return None, None, None

def GK2B_GEMS_O3():
    file_path = (
        os.path.join(ROOT_DIR, "data", "NO2", "GK2B_GEMS", "20250331")
        + os.sep
        + "GK2_GEMS_L2_20250331_0045_NO2_FC-ETC_DPRO_ORI.nc"
    )

    try:
        dd = xr.open_dataset(file_path, group="Data Fields")
        dg = xr.open_dataset(file_path, group="Geolocation Fields")

        o3_var = dd["ColumnAmountO3"]
        lat_var = dg["Latitude"]
        lon_var = dg["Longitude"]

        data = _mask_fill_valid(o3_var.values, o3_var.attrs)
        lat  = _mask_fill_valid(lat_var.values, lat_var.attrs)
        lon  = _mask_fill_valid(lon_var.values, lon_var.attrs)

        # shape 확인 출력 (디버깅용)
        print(f"O3 data shape: {data.shape}, lat shape: {lat.shape}, lon shape: {lon.shape}")

        return data, lat, lon

    except Exception as e:
        print(f"NO2 데이터 로드 실패: {e}")
        return None, None, None