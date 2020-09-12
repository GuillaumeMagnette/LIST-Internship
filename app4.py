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
    #html.Label('Dropdown'),
    #dcc.Dropdown(
    #    id='dropdown_areas',
    #    options=[{'label': label, 'value': label} for label in os.listdir(cropped_path)],
    #    value=os.listdir(cropped_path)[0]
    #),
    #hmtl.Div(dcc.Markdown(f""))
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
    
    #dcc.Dropdown(
    #    id='dropdown_areas_updated',
    html.Div(id='area_selected', style={'display': 'none'}),

    html.Div(dcc.Input(id='input-box', type='text')),
    html.Button('new_area', id='button', n_clicks=0),
    #html.Div(id='output-container-button'),
    dcc.Markdown(id = "return_message"),
    html.Div(id='area_created', style={'display': 'none'}),

    #dcc.Markdown(f"the selected path is {final_path}"),
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
dcc.Markdown(f"There are at the moment {nbr_newts} different newts in the database"),
dcc.Markdown(id = 'output_uploaded'), 
])


@app.callback(Output('output_uploaded', 'children'),
              [Input('upload_prediction', 'contents'),
              Input('area_selected', 'children')])
              #[State('upload_prediction', 'filename'),
              # State('upload_prediction', 'last_modified')])
def update_output(zip_file, final_path):
    #for content, name, date in zip(list_of_contents, list_of_names, list_of_dates):
        # the content needs to be split. It contains the type and the real content
    #print(type(zip_file))
    final_path = json.loads(final_path)
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
        
    input_shape = (75,30,3)

    preprocess_newts(img_path,aug_path,new_cropped_path,input_shape)

    if(not os.path.exists(cropped_path)):
        os.makedirs(cropped_path)
    

    nbr_newts = regroupSameNewts(final_path, new_cropped_path)
    

    markdown = ''' 
    ### newts processed '''
    return f"There are at the moment {nbr_newts} different newts in the database"
    #return markdown
    



@app.callback([Output('area_created', 'children'),
             Output('return_message', 'children')],
             [Input('button', 'n_clicks')],
             [State('input-box', 'value')])
def ask_new_area(n_clicks,value):
    if value is None:
        raise dash.exceptions.PreventUpdate()
    final_path = cropped_path + '/' + str(value)
    if (not os.path.exists(final_path)):
        os.makedirs(final_path)
        message = f"new area created : {value}"

    else:
        message = f"area already created, selecting : {value}"

    return json.dumps(final_path), message   
    

@app.callback(Output('area_selected', 'children'),
             [Input('dropdown_areas', 'value')])
def select_area(value):
    if value is None:
        raise dash.exceptions.PreventUpdate()
    final_path = cropped_path + '/' + str(value)
    print(f"area selected : {value}")

    return json.dumps(final_path)

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

    #Output('message_updated')
    #Input('upload_prediction', 'contents')
#def update_area()
if __name__ == '__main__':
    app.run_server(debug=True)