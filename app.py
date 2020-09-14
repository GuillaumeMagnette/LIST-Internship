from utils_sift import *
import datetime
import os
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from zipfile import ZipFile
import io
import base64
import json




external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

""" Folder creations """

#img_path = 'baseDataset'
img_path = 'newDataset'
aug_path = 'augDataset'
#new_aug_path = 'newAugDataset'
selected_area = ''

cropped_path = 'cropDataset'
if (not os.path.exists(cropped_path)):
    os.makedirs(cropped_path)


new_cropped_path = 'newCropDataset'
nbr_newts = 6

""" Here is the layout of the app """

app.layout = html.Div([

    html.Div(
        className="app-header",
        children=[
            html.Div('Newt Re-identification', className="app-header--title")
        ]
    ),
    html.Hr(),
    html.Div([
        html.H3('You can add here folders containing images of newts'),
        html.H4("Filenames must begin by the area where the newts were photo-captured, followed by an underscore '_'"),
        html.H5("Ex : Bascha_P01_T01_K07_M_20190421025319-180.jpg   This newt will be saved in the area : 'Bascha'")
    ]),
    html.Div([
    dcc.Upload(
    id='upload_prediction',
    children=html.Div([
        'Drag and Drop or ',
        html.A('Select Folder'),
        ' (*.zip)'
    ]),
    style={
        'width': '100%',
        'height': '60px',
        'lineHeight': '60px',
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center',
        'margin': '10px'
    },
    accept=".zip",
    multiple=False
),
html.Div([
    dbc.Spinner(html.Div(id="loading-output")),
    dcc.Markdown(id = 'output_uploaded'),
]),

html.Hr(),

html.Div([
    html.H4("You can see here the list of the different areas detected in the filenames"),
    html.H5("You can select areas to see the number of different newts photo-captured. Delete an area by clicking the button below")
]),

    html.Div(
    id='dropdown_areas_parent',
    children=[
        dcc.Dropdown(
            id='dropdown_areas',
            options=[{'label': label, 'value': label} for label in os.listdir(cropped_path)],
            #value=os.listdir(cropped_path)[0]
        )
    ]
    ),


 
]),

html.Div(
    [
    
    html.Button('Delete Area', id='submit-del', n_clicks=0),

    dcc.Markdown(id='nbr_newts'),
    dcc.Markdown(id='message_del'),

    ]
),

html.Hr(),

html.Div([
    html.H4("You can select here areas geographically near one another"),
    html.H5("This will create a new 'meta-area', and compare every newt of present in the areas selected")
]),


html.Div(
    id='dropdown_areas_parent_multi',
    children=[
        dcc.Dropdown(
            id='dropdown_areas_multi',
            options=[{'label': label, 'value': label} for label in os.listdir(cropped_path)],
            #value=os.listdir(cropped_path)[0]
            multi=True,
        )
    ]
),
html.Button('Associate selected areas', id='submit-val', n_clicks=0),

html.Div([
    dbc.Spinner(html.Div(id="loading-regroup")),
    dcc.Markdown(id = 'nbr_newts_grouped'),
]),


])


@app.callback([Output('output_uploaded', 'children'),
               Output("loading-output", "children")],
              [Input('upload_prediction', 'contents')])
def update_output(zip_file):

    """ This method handles the zip files dropped by the user, the processing and the reidentification """

    if zip_file is None:
        raise dash.exceptions.PreventUpdate()
    

    content_type, content_string = zip_file.split(',')
        # Decode the base64 string
    content_decoded = base64.b64decode(content_string)
        # Use BytesIO to handle the decoded content
    zip_str = io.BytesIO(content_decoded)
        # Now you can use ZipFile to take the BytesIO output
    zip_obj = ZipFile(zip_str, 'r')
    if (os.path.exists(img_path)):
        shutil.rmtree(img_path)
    zip_obj.extractall(img_path)
    
    #if(not os.path.exists(cropped_path)):
    #    os.makedirs(cropped_path)

    create_area(img_path,cropped_path)

    input_shape = (75,30,3)

    preprocess_newts(img_path,aug_path,new_cropped_path,input_shape)

    nbr_newts = []
    
    for area in os.listdir(new_cropped_path):
        nbr_newts.append(regroupSameNewts(cropped_path + '/' + area, new_cropped_path +'/' + area))
    
    shutil.rmtree(img_path)
    shutil.rmtree(aug_path)
    shutil.rmtree(new_cropped_path)
    
    list_area = os.listdir(cropped_path)
    results = "There are at the moment :"
    for area in list_area:
        area_path = cropped_path + '/' + area
        for nbr_newt in os.listdir(area_path):
            results += f" {nbr_newt} in the area called {area}"

  
    return f"There are at the moment {nbr_newts} different newts in the database", f"Re-identification completed"
    

@app.callback(Output('message_del', 'children'),
              [Input('submit-del', 'n_clicks')],
             [State('dropdown_areas', 'value')])
def deleteArea(n_clicks, value):
    """ This method enables the user to delete an unwanted area """

    if not n_clicks or not value:
        raise dash.exceptions.PreventUpdate()

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if ('submit-del' in changed_id):
        area_path = cropped_path + '/' + value
        shutil.rmtree(area_path)

        return f"Area called {value} has been succesfully deleted"
    

    
@app.callback([Output('nbr_newts_grouped', 'children'),
               Output("loading-regroup", "children")],
              [Input('submit-val', 'n_clicks')],
             [State('dropdown_areas_multi', 'value')])
def compareSimilarAreas(n_clicks, value):
  """ This method determines whether some newts in an area have been also detected in a near area """

  if not n_clicks or not value:
      raise dash.exceptions.PreventUpdate()
  changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

  if ('submit-val' in changed_id):

    new_cropped_path = f"{cropped_path}/"

    for area in value:
        new_cropped_path += f"{area}_"

    new_cropped_path = new_cropped_path[:-1]
    if(not os.path.exists(new_cropped_path)):
        os.makedirs(new_cropped_path)

        areas = os.listdir(cropped_path)

        for area in areas:
            if (area in value):
                area_path = cropped_path + '/' + area
                for newts_name in os.listdir(area_path):
                    newts_path = area_path + '/' + newts_name
                    print(newts_path)
                    print(new_cropped_path)
                    if (not os.path.exists(new_cropped_path + '/' + newts_name)):
                        shutil.copytree(newts_path,new_cropped_path + '/' + newts_name)
                
                    
    
        nbr_newts = regroupSimilarAreas(new_cropped_path)

    else:
        nbr_newts = len(os.listdir(new_cropped_path))
  return f"Right now, there are : {nbr_newts} different newts in this area that were 'photo-captured'", f"Regrouping completed"


@app.callback(Output('nbr_newts', 'children'),
             [Input('dropdown_areas', 'value')])
def select_area(value):
    """ This method returns how many newts there are in the selected area """

    if value is None:
        raise dash.exceptions.PreventUpdate()
    final_path = cropped_path + '/' + str(value)
    print(f"area selected : {value}")
    message = f"Right now, there are : {len(os.listdir(final_path))} different newts in {value} that were 'photo-captured'"
    return message

@app.callback(
    Output("dropdown_areas", "options"),
    [Input('dropdown_areas_parent', 'n_clicks')],
    #Input('area_created', 'children')],
)
def update_options(n_clicks):
    """ This method updates the dropdown containing the different names of areas, as new ones are added """

    if not n_clicks:
        raise dash.exceptions.PreventUpdate()
    options = [{'label': label, 'value': label} for label in os.listdir(cropped_path)]
    return options

@app.callback(
    Output("dropdown_areas_multi", "options"),
    [Input('dropdown_areas_parent_multi', 'n_clicks')],
    #Input('area_created', 'children')],
)
def update_options(n_clicks):
    """ This method updates the dropdown containing the different names of areas, as new 'meta-areas' are added """
    if not n_clicks:
        raise dash.exceptions.PreventUpdate()
    
    options = [{'label': label, 'value': label} for label in os.listdir(cropped_path)]
    return options


if __name__ == '__main__':
    app.run_server(debug=True)