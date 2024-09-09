# %%
# An attempt to create a reactive app to plot the latent space
from dash import Dash, html, dcc, Output, Input, no_update
import plotly.express as px
import pandas as pd
import numpy as np
from models.dataset_static import ZarrCellDataset
from torchvision.transforms import v2
from models.static_utils import read_config
import numpy as np
from sklearn.decomposition import PCA
import base64
import io
from PIL import Image

# %% Load the dataset
# Define the metadata keys
metadata_keys = ['gene', 'barcode', 'stage']
images_keys = ['cell_image']
crop_size = 96
normalizations = v2.Compose([v2.CenterCrop(crop_size)])
yaml_file_path = "/mnt/efs/dlmbl/G-et/yaml/dataset_info_20240901_155625.yaml"
csv_file = f"/mnt/efs/dlmbl/G-et/csv/dataset_split_benchmark.csv"
label_type = 'gene'
balance_classes = True

latents_file = '/mnt/efs/dlmbl/G-et/example_latents/val_3_latent_vectors.csv'

df = pd.read_csv(latents_file)
ordered_df = df.sort_values(by=["gene", "barcode", "stage", "cell_idx",])
labels = ordered_df[label_type].tolist()
class_names = ordered_df[label_type].sort_values().unique().tolist()
num_classes = len(class_names)

data_df = pd.read_csv(csv_file)
data_df = data_df[data_df['split'] == 'val'].reset_index()

# %% Run pca
pca = PCA(n_components=2)
latent_columns = [c for c in ordered_df.columns if 'latent' in c]

data = pca.fit_transform(ordered_df[latent_columns])

metadata_options = ['gene', 'barcode', 'stage'] 

# %% Load the training dataset
# Create the dataset
dataset_mean, dataset_std = read_config(yaml_file_path)
dataset = ZarrCellDataset(
    parent_dir = '/mnt/efs/dlmbl/S-md/',
    csv_file = csv_file, 
    split='val',
    channels=[0, 1, 2, 3], 
    mask='min', 
    normalizations=normalizations,
    interpolations=None, 
    mean=dataset_mean, 
    std=dataset_std
)

def renorm(image):
    """
    Turns 4 channel, channel-first tensor from the dataset into a single channel, channel-last numpy array
    """
    im = image.cpu().numpy() 
    return (im - im.min()) / (im.max() - im.min())

def encode_image(image_array):
    """
    Encodes a numpy array as a base64 string
    """
    image = Image.fromarray((renorm(image_array)*255).clip(0, 255).astype(np.uint8))  # Normalize from [-1, 1] to [0, 255]
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


df = pd.DataFrame(data, columns=['pc0', 'pc1'])
for opt in metadata_options:
    df[opt] = ordered_df[opt].tolist()

app = Dash()
app.layout = html.Div([
    html.H1(children='Latent Space Visualization', style={'textAlign': 'center'}),
    html.Div([
        # html.Div([
        #     dcc.Dropdown(
        #         id='channel-dropdown',
        #         options=[{'label': f'Channel {i}', 'value': i} for i in range(4)],
        #         value=0,
        #         clearable=False,
        #         style={'width': '150px'}
        #     ),
        #     dcc.Graph(
        #         id='image',
        #         figure=px.imshow(
        #             renorm(dataset[0]["cell_image"][0]),
        #             color_continuous_scale='gray'
        #         ).update_layout(coloraxis_showscale=False, xaxis_visible=False, yaxis_visible=False),
        #         style={'width': '400px', 'height': '400px'}
        #     ),
        # ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),
        html.Div([
                dcc.Dropdown(
                    id='color-dropdown',
                    options=[{'label': f'{label}', 'value': label} for label in metadata_options],
                    value="gene",
                    clearable=False,
                    style={'width': '150px'}
                ),
                dcc.Graph(
                    id='latent-space',
                    figure=px.scatter(df, x='pc0', y='pc1', color='gene'),
                    style={'width': '1200px', 'height': '600px'}
                ), 
                dcc.Tooltip(id='latent-space-tooltip')
            ]
        )
    ], style={'display': 'flex', 'flexDirection': 'row'})
])

# @app.callback(
#     Output('image', 'figure'),
#     [Input('latent-space', 'clickData'), Input('channel-dropdown', 'value')]
# )
# def update_image(clickData, channel_index):
#     if clickData is None:
#         return no_update
#         # return px.imshow(
#         #     turn_into_rgb(dataset[0]["cell_image"], channel_index),
#         #     color_continuous_scale='gray'
#         # ).update_layout(coloraxis_showscale=False, xaxis_visible=False, yaxis_visible=False)

#     print(clickData) 
#     point_index = clickData['points'][0]['pointIndex']
#     # Get the sample in the ordered_df
#     row = ordered_df.iloc[point_index]
#     # Find the corresponding row in the data_df
#     other_row = data_df[
#         (data_df['gene'] == row['gene']) &
#         (data_df['barcode'] == row['barcode']) &
#         (data_df['stage'] == row['stage']) &
#         (data_df['cell_idx'] == row['cell_idx'])
#     ]
#     point_index = other_row.index[0]
#     return px.imshow(
#         renorm(dataset[point_index]["cell_image"][channel_index]),
#         color_continuous_scale='gray'
#     ).update_layout(coloraxis_showscale=False, xaxis_visible=False, yaxis_visible=False)

@app.callback(
    Output("latent-space-tooltip", "show"),
    Output("latent-space-tooltip", "bbox"),
    Output("latent-space-tooltip", "children"),
    Input("latent-space", "hoverData"),
)
def display_hover(hoverData):
    if hoverData is None or not hoverData["points"]:
        return False, no_update, no_update

    # demo only shows the first point, but other points may also be available
    hover_data = hoverData["points"][0]
    bbox = hover_data["bbox"]
    point_index = hover_data["pointNumber"]

    # point_index = clickData['points'][0]['pointIndex']
    # Get the sample in the ordered_df
    row = ordered_df.iloc[point_index]
    # Find the corresponding row in the data_df
    other_row = data_df[
        (data_df['gene'] == row['gene']) &
        (data_df['barcode'] == row['barcode']) &
        (data_df['stage'] == row['stage']) &
        (data_df['cell_idx'] == row['cell_idx'])
    ]
    
    channel_index = 0
    point_index = other_row.index[0]

    images = [dataset[point_index]["cell_image"][i] for i in range(4)]
    encoded_images = [encode_image(image) for image in images]

    children = [
        html.Div([
            html.Div([
                html.Img(src=f'data:image/png;base64,{encoded_images[i]}', style={'width': '100px', 'height': '100px'}),
                html.P(f'Channel {i}')
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}) for i in range(4)
        ], style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'center'}),
        html.P(f'Gene: {row["gene"]}'),
        html.P(f'Barcode: {row["barcode"]}'),
        html.P(f'Stage: {row["stage"]}'),
        html.P(f'Cell Index: {row["cell_idx"]}')
    ]

    return True, bbox, children

# Callback to change what we color the latent space points by
@app.callback(
    Output('latent-space', 'figure'),
    [Input('color-dropdown', 'value')]
)
def update_latent_space(value):
    return px.scatter(df, x='pc0', y='pc1', color=value)



if __name__ == '__main__':
    app.run(debug=True)

# %%
