from utils_sift import *
import datetime
import os
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html

from zipfile import ZipFile
import io
import base64
import json

#Unet = tf.keras.models.load_model('my_model_Unet')

#print(new_model)

#file_path = 'images/Bascha_P01_T01_K04_F_Adult_4240_20190330204648.jpg'
#input_shape = (75,30,3)
#image = preprocess_images(file_path)
#image = np.resize(image, (128,128,3))
#plt.imsave('images/augNewt.jpg',image/255)
#image_extracted = np.reshape(image_extracted, (128,128,3))
#image_extracted = extract_image_unet('images/augNewt.jpg', Unet)

#plt.imsave('images/augNewtExtracted.jpg',image_extracted)
#plt.imshow(image)
#plt.show()


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)



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



app.layout = html.Div([

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
    

    dcc.Markdown(id='nbr_newts'),


    dcc.Upload(
    id='upload_prediction',
    children=html.Div([
        'Drag and Drop or ',
        html.A('Select Files'),
        ' New Predictions (*.zip)'
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

dcc.Markdown(id = 'output_uploaded'), 

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
html.Button('Associate the areas', id='submit-val', n_clicks=0),
dcc.Markdown(id = 'nbr_newts_grouped'),

])


@app.callback(Output('output_uploaded', 'children'),
              [Input('upload_prediction', 'contents')])
              #Input('area_selected', 'children')])
              #[State('upload_prediction', 'filename'),
              # State('upload_prediction', 'last_modified')])
def update_output(zip_file):
    #for content, name, date in zip(list_of_contents, list_of_names, list_of_dates):
        # the content needs to be split. It contains the type and the real content
    #print(type(zip_file))
    #final_path = json.loads(final_path)
    #print(final_path)
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

    markdown = ''' 
    ### newts processed '''
    return f"There are at the moment {nbr_newts} different newts in the database"
    #return markdown
    
@app.callback(Output('nbr_newts_grouped', 'children'),
              [Input('submit-val', 'n_clicks')],
             [State('dropdown_areas_multi', 'value')])
def compareSimilarAreas(n_clicks, value):
    
  if not n_clicks:
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
                    shutil.copytree(newts_path,new_cropped_path + '/' + newts_name)
    
        nbr_newts = regroupSimilarAreas(new_cropped_path)

    else:
        nbr_newts = len(os.listdir(new_cropped_path))
  return f"Right now, there are : {nbr_newts} different newts in this area that were 'photo-captured'"


@app.callback(Output('nbr_newts', 'children'),
             [Input('dropdown_areas', 'value')])
def select_area(value):
    if value is None:
        raise dash.exceptions.PreventUpdate()
    final_path = cropped_path + '/' + str(value)
    print(f"area selected : {value}")

    return f"Right now, there are : {len(os.listdir(final_path))} different newts in this area that were 'photo-captured'"

@app.callback(
    Output("dropdown_areas", "options"),
    [Input('dropdown_areas_parent', 'n_clicks')],
    #Input('area_created', 'children')],
)
def update_options(n_clicks):
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
    if not n_clicks:
        raise dash.exceptions.PreventUpdate()
    
    options = [{'label': label, 'value': label} for label in os.listdir(cropped_path)]
    return options

#@app.callback(Output("dropdown_areas_multi", "value"),
#              Input("submit_val", "n_clicks"))
#def submit_area_grouped(n_clicks):
#    if not n_clicks:
#        raise dash.exceptions.PreventUpdate()

#    return


    #Output('message_updated')
    #Input('upload_prediction', 'contents')
#def update_area()

if __name__ == '__main__':
    app.run_server(debug=True)