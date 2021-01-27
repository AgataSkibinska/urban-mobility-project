import streamlit as st
import skmob
import pandas as pd
import geopandas as gpd
from streamlit_folium import folium_static
import os
import plotly.express as px

JSON_ASC_DIR = './age_sex/'
JSON_MS_DIR = "./modal_split/"
JSON_FM_DIR = './flow_matrix/'
st.set_page_config(layout="wide")


def get_all_simulation_filenames(directory):
    result = []
    for filename in os.listdir(directory):
        if filename[:3] == 'asc':
            result.append(filename)
    return result


def get_simulation_dict(filepath_list):
    return {file[13:-5]: file for file in filepath_list}


def plot_transportation_by_agesex(display_name, selected_age_sex):
    file_path = JSON_ASC_DIR + 'asc_scenario_' + display_name + ".json"
    df = pd.read_json(file_path)

    chosen_column = 'modal_split'
    chosen_age = selected_age_sex

    data = df.loc[chosen_column]
    result = pd.DataFrame()

    for row in data.iteritems():
        id, obj = row
        temp_df = pd.DataFrame(obj)
        temp_df['age_sex'] = id
        result = pd.concat([result, temp_df])

    df_m = result.loc['mean']
    df_m = df_m.transpose()
    headers = df_m.iloc[-1]
    index = df_m.index[:-1]
    new_df = pd.DataFrame(df_m.values[:-1], columns=headers, index=index)
    new_df = (100. * new_df / new_df.sum()).astype(float).round(1)
    labels = {'index': "Transport mode", chosen_age: "Mean of percentage of transport mode usage"}
    fig = px.bar(data_frame=new_df, y=chosen_age, x=new_df.index, title=chosen_age,
                 labels=labels, range_y=[0, 50]
                 )
    fig.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig)
    # st.dataframe(pd.DataFrame(df.loc[chosen_column][chosen_age]).transpose(), width=500)


def plot_transportation_no_filter(display_name):
    file_path = JSON_ASC_DIR + 'asc_scenario_' + display_name + ".json"
    df = pd.read_json(file_path)

    chosen_column = 'modal_split'
    data = df.loc[chosen_column]
    result = pd.DataFrame()
    for row in data.iteritems():
        id, obj = row
        temp_df = pd.DataFrame(obj)
        temp_df['age_sex'] = id
        result = pd.concat([result, temp_df])

    df_m = result.loc['mean']
    df_m = df_m.iloc[:, :-1]

    new_df = (df_m.sum()).astype(float)

    new_df = (100. * new_df / new_df.sum()).astype(float).round(1)
    fig = px.bar(data_frame=pd.DataFrame(new_df, columns=['prc']), y='prc', title="All age/sex groups",
                 labels=dict(index="Transportation", prc="Percentage of sum of transport usage"), range_y=[0, 50])
    fig.update_yaxes(tickprefix="%")
    st.plotly_chart(fig)


def load_comparison():
    col1, col2 = st.beta_columns(2)
    all_simulations_names = get_all_simulation_filenames(JSON_ASC_DIR)
    display_names_dict = get_simulation_dict(all_simulations_names)

    selected_filtered = st.sidebar.select_slider('Age/sex filter', options=['Filter', 'No filter'])
    if selected_filtered == 'Filter':
        sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
        if sex == 'Female':
            age = st.sidebar.selectbox('Age', ['6-15', '16-19', '20-24', '25-44', '45-60', '61-x'])
            sex = 'K'
        else:
            age = st.sidebar.selectbox('Age', ['6-15', '16-19', '20-24', '25-44', '45-65', '66-x'])
            sex = 'M'
        selected_age_sex = age + "_" + sex
        with col1:
            selected_1 = st.selectbox('Select scenario # 1', options=list(display_names_dict.keys()))
            plot_transportation_by_agesex(selected_1, selected_age_sex)

        with col2:
            selected_2 = st.selectbox('Select scenario # 2', options=list(display_names_dict.keys()))
            plot_transportation_by_agesex(selected_2, selected_age_sex)
    elif selected_filtered == 'No filter':
        with col1:
            selected_1 = st.selectbox('Select scenario # 1', options=list(display_names_dict.keys()))
            plot_transportation_no_filter(selected_1)

        with col2:
            selected_2 = st.selectbox('Select scenario # 2', options=list(display_names_dict.keys()))
            plot_transportation_no_filter(selected_2)


def load_city_regions():
    wroc_map = gpd.read_file('EtapII-REJONY_wroclaw.shp')
    city_regions = wroc_map[['NUMBER', 'geometry']]
    city_regions.columns = ['REGION', 'geometry']
    return city_regions.to_crs(epsg=4326)


def plot_flows(flow_matrix, city_regions, min_flow):
    fdf = skmob.FlowDataFrame(
        data=flow_matrix,
        origin='start_region',
        destination='dest_region',
        flow='mean',
        tile_id='REGION',
        tessellation=city_regions
    )
    m = fdf.plot_tessellation(tiles='OpenStreetMap')
    plt = fdf.plot_flows(m, flow_color='red', tiles='OpenStreetMap', min_flow=min_flow, flow_weight=0.35, zoom=1000)
    folium_static(plt, width=1000, height=600)


def load_flows():
    scenario = st.sidebar.selectbox('Scenario', ['0_0_0', '0_0_3', '15_15_0', '15_15_15', '0_0_15', '3_3_0', '3_3_3'])
    flows_mode = st.sidebar.selectbox('Flows', ['All', 'Age + sex', 'Transport mode'])
    flows_mode_dict = {'All': 'all', 'Age + sex': 'asc', 'Transport mode': 'tm'}
    flow_matrix = pd.read_pickle(JSON_FM_DIR + 'fm_' + flows_mode_dict[flows_mode] + "_sceanario_" + scenario + '.pkl')
    if flows_mode == 'Age + sex':
        sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
        if sex == 'Female':
            age = st.sidebar.selectbox('Age', ['6-15', '16-19', '20-24', '25-44', '45-60', '61-x'])
            sex = 'K'
        else:
            age = st.sidebar.selectbox('Age', ['6-15', '16-19', '20-24', '25-44', '45-65', '66-x'])
            sex = 'M'
        age_sex = age + "_" + sex
        flow_matrix = flow_matrix.reset_index()
        flow_matrix = flow_matrix[flow_matrix['age_sex'] == age_sex]
    elif flows_mode == 'Transport mode':
        transport_mode_dict = {'car': 0, 'public': 1, 'bicycle': 3, 'pedestrian': 2}
        transport_mode = st.sidebar.selectbox('Transport mode', ['car', 'public', 'bicycle', 'pedestrian'])
        flow_matrix = flow_matrix.reset_index()
        flow_matrix = flow_matrix[flow_matrix['transport_mode'] == transport_mode_dict[transport_mode]]

    min_flow = 20
    min_flow = st.sidebar.slider('min flow', 20, 100)
    submmit = st.sidebar.button("Check result")
    if submmit:
        city_regions = load_city_regions()
        plot_flows(flow_matrix, city_regions, min_flow)


def load_info():
    st.balloons()
    st.title('Social-Explained Urban Mobility Model')
    st.header('Background')
    st.markdown("Current approaches for urban traffic modelling consider only number of objects (mainly cars) that "
                "moved from one area to another and don't answer questions: who, how, and why move? Using social "
                "related data to get knowledge about transport decisions motivations could be a significant factor "
                "to improve traffic modelling. The research concerns the city of Wrocław.")
    st.header('Scenarios description')
    st.markdown(
        'x1_x2_x3 \n'
        '- x1 - percentage of the population experiencing an improvement in the comfort of public transport \n'
        '- x2 - percentage of the population experiencing an improvement in the punctuality of public transport \n'
        '- x3 - percentage of the population that will reduce household car ownership by one \n'
    )
    st.header('Authors')
    st.markdown(
        '- Marcel Cielinski \n'
        '- Weronika Pawlak \n'
        '- Kornel Romański \n'
        '- Agata Skibińska \n'
    )


# ENTRYPOINT
selected = st.sidebar.selectbox('Mode', options=['Info', 'Scenario comparison', 'Flows'], index=0)
if selected == 'Scenario comparison':
    load_comparison()
elif selected == 'Flows':
    load_flows()
elif selected == "Info":
    load_info()
