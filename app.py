from utils import *
import datetime
import os
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html

from zipfile import ZipFile
import io
import base64

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


#encoded_image = base64.b64encode(open('images/augNewtExtracted.jpg', 'rb').read()).decode('ascii')
#encoded_image = base64.b64encode(image).decode('ascii')
img_path = "images"
aug_path = "augmented_images"
final_path = "extracted_images"

app.layout = html.Div([
    
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
])


@app.callback(Output('output_uploaded', 'children'),
              [Input('upload_prediction', 'contents')])
              #[State('upload_prediction', 'filename'),
              # State('upload_prediction', 'last_modified')])
def update_output(zip_file):
    #for content, name, date in zip(list_of_contents, list_of_names, list_of_dates):
        # the content needs to be split. It contains the type and the real content
    print(type(zip_file))
    if zip_file is None:
        raise dash.exceptions.PreventUpdate()
    
    content_type, content_string = zip_file.split(',')
        # Decode the base64 string
    content_decoded = base64.b64decode(content_string)
        # Use BytesIO to handle the decoded content
    zip_str = io.BytesIO(content_decoded)
        # Now you can use ZipFile to take the BytesIO output
    zip_obj = ZipFile(zip_str, 'r')

    zip_obj.extractall(img_path)
        
    #input_shape = (75,30,3)

    #preprocess_newts(img_path,aug_path,final_path,input_shape)

    #dataset = import_images(final_path,input_shape)

    #model = train_triplet_loss(128, input_shape, dataset)

    markdown = ''' 
    ### newts processed '''
    return markdown
    


if __name__ == '__main__':
    app.run_server(debug=True)