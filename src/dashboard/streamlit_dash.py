import streamlit as st
import skmob
import pandas as pd
import geopandas as gpd
from streamlit_folium import folium_static

def load_hist():
    pass

def load_modal_split():
    pass
    
@st.cache
def load_city_regions():
    wroc_map = gpd.read_file('EtapII-REJONY_wroclaw.shp')
    city_regions = wroc_map[['NUMBER', 'geometry']]
    city_regions.columns = ['REGION', 'geometry']
    return city_regions.to_crs(epsg=4326)

def plot_flows(flow_matrix, city_regions, min_flow):
    fdf_from = skmob.FlowDataFrame(
    data=flow_matrix,
    origin='start_region',
    destination='dest_region',
    flow='mean',
    tile_id='REGION',
    tessellation=city_regions
    )
    m = fdf_from.plot_tessellation(tiles='OpenStreetMap')
    plt = fdf_from.plot_flows(m, flow_color='red', tiles='OpenStreetMap', min_flow=min_flow)
    folium_static(plt)


def load_flows():
    scenario = st.sidebar.selectbox('Scenario',['real', 'public transport+', 'public transport++', 'cars-', 'cars--'])
    scenario_dict = {'real' : "0_0_0"} #TODO: rest of dict #, 'public transport+', 'public transport++', 'cars-', 'cars--'}
    flows_mode = st.sidebar.selectbox('Flows',['All', 'Age + sex', 'Transport mode'])
    flows_mode_dict = {'All' : 'all', 'Age + sex': 'asc', 'Transport mode': 'tm'}
    flow_matrix = pd.read_pickle('fm_'+ flows_mode_dict[flows_mode] + "_sceanario_" + scenario_dict[scenario] + '.pkl')
    if flows_mode == 'Age + sex':
        sex = st.sidebar.selectbox('Sex',['Male', 'Female'])
        if sex == 'Female':
            age = st.sidebar.selectbox('Age',['0-5', '6-15', '16-19', '20-24', '25-44', '45-60', '61-x'])
            sex = 'K'
        else:
            age = st.sidebar.selectbox('Age',['0-5', '6-15', '16-19', '20-24', '25-44', '45-65', '66-x'])
            sex = 'M'
        age_sex = age + "_" + sex
        flow_matrix = flow_matrix.reset_index()
        flow_matrix=flow_matrix[flow_matrix['age_sex'] == age_sex]
    elif flows_mode == 'Transport mode':
        transport_mode_dict = {'car': 0, 'public': 1, 'bicycle': 3, 'pedestrian': 2}
        transport_mode = st.sidebar.selectbox('Transport mode',['car', 'public', 'bicycle', 'pedestrian'])
        flow_matrix = flow_matrix.reset_index()
        flow_matrix = flow_matrix[flow_matrix['transport_mode']==transport_mode_dict[transport_mode]]

    min_flow = 50
    min_flow = st.sidebar.slider('min flow', 50, 100)
    submmit = st.sidebar.button("Check result")
    if submmit: 
        city_regions = load_city_regions()
        plot_flows(flow_matrix, city_regions, min_flow)



    




selected = st.sidebar.selectbox('Results',['Histograms', 'Modal split', 'Flows'])
if selected == 'Histograms':
    load_hist()
elif selected == 'Modal split':
    load_modal_split()
elif selected == 'Flows':
    load_flows()



