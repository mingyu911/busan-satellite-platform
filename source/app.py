import os
import matplotlib
matplotlib.use('Agg')
import io
import base64
import numpy as np
import geopandas as gpd
import folium
from folium import Element, plugins
from flask import Flask, render_template, request
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import branca

from satellite import SP5_TROPOMI_CH4
from satellite import S2_MSI_Chl
from satellite import GK2B_GOCI2_Chl
from satellite import GK2B_GEMS_AOD
from satellite import GK2B_GEMS_NO2
from satellite import GK2B_GEMS_O3

app = Flask(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))

# ── 상수 ────────────────────────────────────────────────────────────────────
BUSAN_CENTER   = [35.1796, 129.0756]
BUSAN_LAT_BOX  = (34.5, 35.8)
BUSAN_LON_BOX  = (128.5, 129.7)
SHP_PATH_L2 = os.path.join(ROOT_DIR, "shp", "sig_20230729", "sig.shp")
SHP_PATH_L3 = os.path.join(ROOT_DIR, "shp", "emd_20230729", "emd.shp")

CH4_VMIN, CH4_VMAX = 1900.0, 1950.0
CH4_COLORMAP       = mpl.colormaps.get_cmap('jet')
CH4_NORM           = mcolors.Normalize(vmin=CH4_VMIN, vmax=CH4_VMAX)

Chl_VMIN, Chl_VMAX = 0.0, 5.0
Chl_COLORMAP       = mpl.colormaps.get_cmap('jet')
Chl_NORM           = mcolors.Normalize(vmin=Chl_VMIN, vmax=Chl_VMAX)

AOD_VMIN, AOD_VMAX = 0.0, 1.5
AOD_COLORMAP       = mpl.colormaps.get_cmap('jet')
AOD_NORM           = mcolors.Normalize(vmin=AOD_VMIN, vmax=AOD_VMAX)

NO2_VMIN, NO2_VMAX = 0.0, 1.5
NO2_COLORMAP       = mpl.colormaps.get_cmap('jet')
NO2_NORM           = mcolors.Normalize(vmin=NO2_VMIN, vmax=NO2_VMAX)

O3_VMIN, O3_VMAX   = 300, 450
O3_COLORMAP        = mpl.colormaps.get_cmap('jet')
O3_NORM            = mcolors.Normalize(vmin=O3_VMIN, vmax=O3_VMAX)

COLORMAP_LEGEND_STYLE = """
<style>
    body.sat-mode svg.folium-colormap text { fill: white !important; }
    body.admin-mode svg.folium-colormap text { fill: black !important; }
</style>
"""

# ── 공통 이미지 생성 헬퍼 ─────────────────────────────────────────────────────
def _curvilinear_to_image(data, lat, lon, cmap, norm, lat_box, lon_box):
    """
    matplotlib pcolormesh로 원본 격자 그대로 렌더링 (리그리딩 없음)
    → PNG로 저장 후 ImageOverlay로 올림
    """
    # 부산 영역 crop
    region_mask = (
        (lat >= lat_box[0]) & (lat <= lat_box[1]) &
        (lon >= lon_box[0]) & (lon <= lon_box[1])
    )
    rows_idx = np.where(region_mask.any(axis=1))[0]
    cols_idx  = np.where(region_mask.any(axis=0))[0]

    if rows_idx.size == 0 or cols_idx.size == 0:
        print("부산 영역 데이터 없음")
        return None, None

    data = data[rows_idx[0]:rows_idx[-1]+1, cols_idx[0]:cols_idx[-1]+1]
    lat  =  lat[rows_idx[0]:rows_idx[-1]+1, cols_idx[0]:cols_idx[-1]+1]
    lon  =  lon[rows_idx[0]:rows_idx[-1]+1, cols_idx[0]:cols_idx[-1]+1]

    lat_min, lat_max = float(lat.min()), float(lat.max())
    lon_min, lon_max = float(lon.min()), float(lon.max())

    # pcolormesh: 리그리딩 없이 2D 격자 그대로 렌더링
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)  # 여백 제거

    ax.pcolormesh(
        lon, lat, data,
        cmap=cmap, norm=norm,
        shading='nearest',  # 보간 없이 원본 셀 그대로
    )

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect('equal')

    buffer = io.BytesIO()
    fig.savefig(buffer, format='PNG', transparent=True,
                bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    bounds = [[lat_min, lon_min], [lat_max, lon_max]]
    return f"data:image/png;base64,{b64}", bounds


def _add_image_overlay(m: folium.Map, data_url: str, bounds: list, opacity: float = 0.6):
    """base64 PNG를 folium ImageOverlay로 지도에 추가."""
    folium.raster_layers.ImageOverlay(
        image=data_url,
        bounds=bounds,
        opacity=opacity,
        origin='upper',
        cross_origin=False,
        zindex=1,
    ).add_to(m)


def _add_legend(m: folium.Map, cmap, vmin: float, vmax: float, caption: str):
    """색상바 레전드 추가."""
    color_steps = [mcolors.to_hex(cmap(v)) for v in np.linspace(0, 1, 9)]
    lgd = branca.colormap.StepColormap(
        colors=color_steps,
        vmin=vmin, vmax=vmax,
        index=np.linspace(vmin, vmax, 9),
        caption=caption,
    )
    m.add_child(lgd)
    m.get_root().header.add_child(Element(COLORMAP_LEGEND_STYLE))


# ── 지도 헬퍼 ────────────────────────────────────────────────────────────────
def _get_tile_config(layer_type: str) -> tuple[str, str]:
    if layer_type == "admin":
        return "OpenStreetMap", "© OpenStreetMap contributors"
    return (
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "Esri",
    )


def _build_base_map(layer_type: str, lat: float, lon: float, zoom: int) -> folium.Map:
    tiles, attr = _get_tile_config(layer_type)
    m = folium.Map(location=[lat, lon], zoom_start=zoom, tiles=tiles, attr=attr)

    plugins.LocateControl(
        location_options={'enableHighAccuracy': True},
        draw_circle=True,
        draw_marker=True,
        initial_zoom_level=15,
        stop_following=False,
        strings={"title": "내 위치 찾기", "popup": "현재 위치"},
    ).add_to(m)

    return m


def _add_busan_boundary(m: folium.Map, level: str):
    if str(level) == "2":
        shp_path = SHP_PATH_L2
        fields, aliases, filter_col = ['SIG_KOR_NM'], [''], 'SIG_CD'
    else:
        shp_path = SHP_PATH_L3
        fields, aliases, filter_col = ['EMD_KOR_NM'], [''], 'EMD_CD'

    try:
        gdf = gpd.read_file(shp_path, encoding='cp949')
        gdf[filter_col] = gdf[filter_col].astype(str)
        busan_gdf = gdf[gdf[filter_col].str.startswith('26')].copy()

        if busan_gdf.crs is None:
            busan_gdf.crs = "epsg:5179"
        busan_gdf = busan_gdf.to_crs(epsg=4326)

        folium.GeoJson(
            data=busan_gdf.to_json(),
            name=f'busan_lvl{level}',
            style_function=lambda _: {
                'fillColor': "#0000004C", 'color': "#ffffff",
                'weight': 1.5, 'fillOpacity': 1.0,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=fields, aliases=aliases, labels=False, sticky=True,
                style="background-color: #0b1629; color: #c8dff5; border: 1px solid #00b4ff;"
            ),
            highlight_function=lambda _: {'weight': 3, 'fillColor': "#FBFF0050", 'fillOpacity': 1.0},
        ).add_to(m)

    except Exception as e:
        print(f"!!! SHP 에러 발생: {str(e)}")


# ── 위성 데이터 오버레이 ──────────────────────────────────────────────────────
def _add_ch4_overlay(m: folium.Map, layer_type: str):
    data, lat, lon = SP5_TROPOMI_CH4()
    if data is None:
        print("CH4 데이터 없음")
        return

    data_url, bounds = _curvilinear_to_image(
        data, lat, lon,
        cmap=CH4_COLORMAP,
        norm=CH4_NORM,
        lat_box=BUSAN_LAT_BOX,
        lon_box=BUSAN_LON_BOX,
    )
    if data_url is None:
        return

    _add_image_overlay(m, data_url, bounds)
    _add_legend(m, CH4_COLORMAP, CH4_VMIN, CH4_VMAX, 'Methane Mixing Ratio (CH4) [ppb]')

def _add_chl_overlay(m: folium.Map, layer_type: str, source: str = "GOCI2"):
    if source == "MSI":
        data, lat, lon = S2_MSI_Chl()
    else:
        data, lat, lon = GK2B_GOCI2_Chl()

    if data is None:
        print(f"Chl-a 데이터 없음 ({source})")
        return

    data_url, bounds = _curvilinear_to_image(
        data, lat, lon,
        cmap=Chl_COLORMAP,
        norm=Chl_NORM,
        lat_box=BUSAN_LAT_BOX,
        lon_box=BUSAN_LON_BOX,
    )
    if data_url is None:
        return

    _add_image_overlay(m, data_url, bounds)
    _add_legend(m, Chl_COLORMAP, Chl_VMIN, Chl_VMAX, 'Chlorophyll-a Concentration [mg/m³]')

def _add_aod_overlay(m: folium.Map, layer_type: str, source: str = "GEMS"):
    if source == "GEMS":
        data, lat, lon = GK2B_GEMS_AOD()

    if data is None:
        print(f"AOD 데이터 없음 ({source})")
        return

    data_url, bounds = _curvilinear_to_image(
        data, lat, lon,
        cmap=AOD_COLORMAP,
        norm=AOD_NORM,
        lat_box=BUSAN_LAT_BOX,
        lon_box=BUSAN_LON_BOX,
    )
    if data_url is None:
        return

    _add_image_overlay(m, data_url, bounds)
    _add_legend(m, AOD_COLORMAP, AOD_VMIN, AOD_VMAX, 'Aerosol Optical Depth [unitless]')

def _add_no2_overlay(m: folium.Map, layer_type: str, source: str = "GEMS"):
    if source == "GEMS":
        data, lat, lon = GK2B_GEMS_NO2()

    if data is None:
        print(f"NO2 데이터 없음 ({source})")
        return

    data_url, bounds = _curvilinear_to_image(
        data, lat, lon,
        cmap=NO2_COLORMAP,
        norm=NO2_NORM,
        lat_box=BUSAN_LAT_BOX,
        lon_box=BUSAN_LON_BOX,
    )
    if data_url is None:
        return

    _add_image_overlay(m, data_url, bounds)
    _add_legend(m, NO2_COLORMAP, NO2_VMIN, NO2_VMAX, 'Nitrogen Dioxide Column Amount troposphere [DU]')

def _add_o3_overlay(m: folium.Map, layer_type: str, source: str = "GEMS"):
    if source == "GEMS":
        data, lat, lon = GK2B_GEMS_O3()

    if data is None:
        print(f"O3 데이터 없음 ({source})")
        return

    data_url, bounds = _curvilinear_to_image(
        data, lat, lon,
        cmap=O3_COLORMAP,
        norm=O3_NORM,
        lat_box=BUSAN_LAT_BOX,
        lon_box=BUSAN_LON_BOX,
    )
    if data_url is None:
        return

    _add_image_overlay(m, data_url, bounds)
    _add_legend(m, O3_COLORMAP, O3_VMIN, O3_VMAX, 'Ozone Column Amount [DU]')

def _add_click_event(m: folium.Map):
    m.get_root().html.add_child(Element("""
    <script>
    setTimeout(function(){
        var mapObj = Object.values(window).find(v => v instanceof L.Map);
        if (!mapObj) return;

        mapObj.eachLayer(function(layer){
            if (!layer.feature) return;
            layer.on('click', function(){
                var props = layer.feature.properties;
                var target = parent.document.getElementById("selected-region");
                if (target) target.innerHTML = props.NAME_2 + " " + props.NAME_3;
            });
        });
    }, 1000);
    </script>
    """))


# ── Flask 라우트 ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/map_view")
def map_view():
    layer_type    = request.args.get("type", "sat")
    sat_data      = request.args.get("sat_data")
    selected_date = request.args.get("date")
    selected_sat  = request.args.get("sat_name", "")
    level         = request.args.get("level", "2")

    m = _build_base_map(layer_type, BUSAN_CENTER[0], BUSAN_CENTER[1], 10)
    _add_busan_boundary(m, level)

    if sat_data == "ch4" and selected_date == "2024-03-13" and "TROPOMI" in selected_sat:
        _add_ch4_overlay(m, layer_type)
    if sat_data == "chl" and selected_date == "2025-03-09" and "MSI" in selected_sat:
        _add_chl_overlay(m, layer_type, source="MSI")
    if sat_data == "chl" and selected_date == "2025-03-09" and "GOCI" in selected_sat:
        _add_chl_overlay(m, layer_type, source="GOCI2")
    if sat_data == "aod" and selected_date == "2025-03-24" and "GEMS" in selected_sat:
        _add_aod_overlay(m, layer_type, source="GEMS")
    if sat_data == "no2" and selected_date == "2025-03-31" and "GEMS" in selected_sat:
        _add_no2_overlay(m, layer_type, source="GEMS")
    if sat_data == "o3" and selected_date == "2025-03-31" and "GEMS" in selected_sat:
        _add_o3_overlay(m, layer_type, source="GEMS")

    _add_click_event(m)

    m.get_root().width  = '100%'
    m.get_root().height = '100%'
    return m._repr_html_()


if __name__ == "__main__":
    app.run(debug=True)