# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 16:21:53 2022

@author: shijun
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Equation of ring cyclide
# see https://en.wikipedia.org/wiki/Dupin_cyclide
import numpy as np
import matplotlib
from numpy import pi, sin, cos, mgrid

# cmap from matplot to plotly
def cmap_m2p(name='coolwarm', n=300, vmin=0, vmax=1):
    cmap = matplotlib.colormaps.get_cmap(name)
    pos = np.linspace(0, 1, n)
    values = np.linspace(vmin, vmax, n)
    colorscale = []
    for i in range(n):
        rgb = np.array(cmap(values[i])[:3]) * 255
        rgb = rgb.astype(np.uint8)
        colorscale.append([pos[i], f'rgb({rgb[0]},{rgb[1]},{rgb[2]})'])
    return colorscale





# data1
# data2

def plot(data1,data2,PN_save):
    # a, b, d = 1.32, 1., 0.8
    # c = a**2 - b**2
    # u, v = np.mgrid[0:2*np.pi:100j, 0:2*np.pi:100j]
    # x = (d * (c - a * np.cos(u) * np.cos(v)) + b**2 * np.cos(u)) / (a - c * np.cos(u) * np.cos(v))
    # y = b * np.sin(u) * (a - d*np.cos(v)) / (a - c * np.cos(u) * np.cos(v))
    # z = b * np.sin(v) * (c*np.cos(u) - d) / (a - c * np.cos(u) * np.cos(v))
    
    fig = make_subplots(rows=1, cols=2,
                    specs=[[{'is_3d': True}, {'is_3d': True} ]],
                    # subplot_titles=['FCNN', 'MMNN'],
                    subplot_titles=['NN1', 'NN2'],
                    horizontal_spacing=0.01,
                    vertical_spacing=0.3
                    )
    
    
    mycmap = cmap_m2p("coolwarm", n=250)
    
    surfaces=[]
    
    # mykargs=dict(opacity=1,
    #          colorscale=mycmap,
    #          cmin=-1, # set min value for the colorbar
    #          cmax=1
    #          )
    
    mykargs=dict(opacity=1,
         colorscale=mycmap,
         # cmin=-0.6, # set min value for the colorbar
         # cmax=0.6
         )
    
    
    
    def myTruncate(xyz0,n=0):
        xyz1=[]
        for xi in xyz0:
            if n>0.5:
                xi_new=xi[n:-n,n:-n]
            else:
                xi_new=xi
            xyz1.append(xi_new)
        return(xyz1)
    
    
    xyz=myTruncate(data1)
        
    
    
    
    s1=go.Surface(x=xyz[0], y=xyz[1], z=xyz[2],
              # colorbar_x=1.03,
               **mykargs,
               colorbar=dict(
                   # title="wb", 
                             x=0.475)
              ) 
    surfaces.append(s1)
    # s1.colorbar.title="zsj"
    
    # cr=s1.colorbar
    # cr.lenmode='fraction'
    # cr.len=0.28
    # cr.thickness=25
    # cr.x=0.167
    # cr.y=-0.1
    # cr.orientation='h'
    ############
    # fig.update_traces(colorbar=dict(orientation='v',x=0.5,y=0,len=0.3))
    
    xyz=myTruncate(data2)
    s2=go.Surface(x=xyz[0], y=xyz[1], z=xyz[2], 
               # showscale=False,
               **mykargs,
               colorbar=dict(
                   # title="ac",
                   x=1.05
                             )
              )
    surfaces.append(s2)
    # cr=s2.colorbar
    # cr.lenmode='fraction'
    # cr.len=0.3
    # cr.thickness=25
    # cr.x=0.17+0.33
    # cr.y=-0.1
    # cr.orientation='h'
    
    # xyz=myTruncate(data2)
    # s3=go.Surface(x=xyz[0], y=xyz[1], z=xyz[2]-myTruncate(data1)[2], 
    #               # colorscale=mycmap,
    #                showscale=False,
    #                **mykargs
    #               )
    # surfaces.append(s3)
    # cr=s3.colorbar
    # cr.lenmode='fraction'
    # cr.len=0.3
    # cr.thickness=25
    # cr.x=0.173+0.66
    # cr.y=-0.1
    # cr.orientation='h'
    
    # fig.add_trace(go.Surface(x=x, y=y, z=z0,colorscale=mycmap), 1, 1)
    # fig.add_trace(go.Surface(x=x, y=y, z=z0, colorscale=[[0,"red"],[1,'black']]), 1, 2)
    # fig.add_trace(go.Surface(x=x, y=y, z=z0-z), 1, 3)
    
    for k,s in enumerate(surfaces):
        fig.add_trace(s, 1, k+1)
    
    
    
    ###########
    fig.update_annotations(font=dict(family="Helvetica", size=22))
    # fig.layout.annotations[0].update(y=0.8)
    
    
    ########### scenes
    fig.update_scenes(aspectratio=dict(x=0.75,y=0.75,z=0.485),
                  # camera_eye=dict(x=1.25,y=1.25,z=0.6),
                  camera_eye=dict(x=1.25,y=-1.25,z=0.696),
                  )
    
    ### axis x, y, z
    ## ticks
    # fig.update_scenes(xaxis=dict(
    #                           # color='black' # line font tick
    #                           # linecolor='black', linewidth=1
    #                           tickfont=dict(size=10,color='red'),
    #                           tickprefix='zsj',ticksuffix='dw',
    #                           tickvals=[0,0.5,1],tickmode='array',ticktext=['zsj','dw']
    #                              )
    #                   )
    
    # fig.update_scenes(xaxis=dict(range=[-0.1,1.1],
    #                       # title=dict(text="x",font=dict(color='blue',size=15))
    #                              ),
    #                   yaxis=dict(range=[-0.1,1.1],
    #                      # title=dict(text="x",font=dict(color='blue',size=15))
    #                              ),
    #                     # zaxis=dict(range=[-0.2,0.2],
    #                     #     # title=dict(text="x",font=dict(color='blue',size=15))
    #                     #           )                    
    #                     )
    
    # zerror=data2[2]-data1[2]
    # # zmin=np.floor(zerror.min()*30)/30
    # # zmax=np.floor(zerror.max()*30)/30+0.033
    # zmin=zerror.min()
    # zmax=zerror.max()
    # fig.update_scenes( zaxis=dict(range=[zmin,zmax],
    #                         # title=dict(text="x",font=dict(color='blue',size=15))
    #                               )  ,
    #                     row=1, col=3
    #                     )
    
    
    #################################
    ##### update layout
    # fig title
    fig.update_layout(title_font=dict(size=50,color='red'))
    # fig.update_layout(title_pad=dict(l=-0,b=0,r=0,t=0),title_text='zsj',
    #                   title_xanchor='center',
    #                   title_x=0.5 # middle,
    #                   )
    
    
    
    # legend (not colorbar)
    fig.update_layout(showlegend=False,
                  legend_x=0.5)
    
    # r=69*1.7
    r=data1[0].max()*1.15
    myrange=[-1.01*r, 1.01*r]
    # Update layout for ticks
    fig.update_layout(
    scene=dict(
        xaxis=dict(
            title="x",
            range=myrange
            # tickvals=[-20, -10, 0, 10, 20],  # Custom tick positions
            # ticktext=["-20", "-10", "Zero", "10", "20"],  # Custom tick labels
        ),
        yaxis=dict(
            title="y",
            range=myrange
            # tickangle=45,  # Rotate tick labels by 45 degrees
            # dtick=10       # Set tick spacing to 10
        ),
        zaxis=dict(
            title="z",
            # tickvals=[0, 10, 20, 30],  # Custom tick positions
            # ticktext=["Low", "Medium", "High", "Max"]  # Custom labels
        )
    ),
    
    scene2=dict(
        xaxis=dict(
            title="x",
            range=myrange
            # tickvals=[-20, -10, 0, 10, 20],  # Custom tick positions
            # ticktext=["-20", "-10", "Zero", "10", "20"],  # Custom tick labels
        ),
        yaxis=dict(
            title="y",
            range=myrange
            # tickangle=45,  # Rotate tick labels by 45 degrees
            # dtick=10       # Set tick spacing to 10
        ),
        zaxis=dict(
            title="z",
            # tickvals=[0, 10, 20, 30],  # Custom tick positions
            # ticktext=["Low", "Medium", "High", "Max"]  # Custom labels
        )
    )
    )
    
    # margin
    fig.update_layout(margin=dict(b=80,l=80,r=80,t=100))
    
    # size
    # fig.update_layout(width=500,height=500)
    
    
    #  the global font
    # fig.update_layout(font=dict(color="blue",size=50))
    
    
    # modebar
    fig.update_layout(modebar=dict(activecolor='#EF7C00',
                               add=[
                                    "v1hovermode",
                                    "toggleSpikelines",
                                    ],
                               color="#003D7C",
                               # bgcolor="#00AAAA",
                               orientation="h",
                               )
                  )
    
    
    
    
    
    
    # fig.write_html("demo.html",include_plotlyjs='cdn')
    # config={'modeBarButtonsToAdd':['drawline',
    #                                         'drawopenpath',
    #                                         'drawclosedpath',
    #                                         'drawcircle',
    #                                         'drawrect',
    #                                         'eraseshape'
    #                                        ]}
    # config = dict({'scrollZoom': True})
    
    # fig.update_layout(
    #     dragmode='drawopenpath',
    #     newshape_line_color='cyan',
    #     title_text='Draw a path to separate versicolor and virginica',
    # )
    
    
    config = {'displaylogo': False,
          'toImageButtonOptions': {
            'format': 'png', # one of png, svg, jpeg, webp
            'filename': 'custom_image',
            # 'height': 500,
            # 'width': 3700,
            'scale': 5 # Multiply title/legend/axis/canvas sizes by this factor
          },
          # 'modeBarButtonsToAdd':['drawline',
          #                               'drawopenpath',
          #                               'drawclosedpath',
          #                               'drawcircle',
          #                               'drawrect',
          #                               'eraseshape'
          #                              ]
          }
    fig.write_html(PN_save,config=config,include_plotlyjs='cdn')
    fig.show(config=config)
    # print("zz")
    
    # # Save the plot as a PDF
    # fig.write_image("surface_plot.png", width=80, height=60, scale=1)
    
    
    # # Show the plot
    # fig.show()
    
    # print("ss")

if __name__ == "__main__":
    
    nn_type, W_idx="FCNN", 1
    
    nn_type, W_idx="MMNN", 1 
    
    # nn_type, W_idx="MMNN", 2
    

    act_idx=1 # 1 for sin
    # act_idx=2 # 2 for SinTU_0
    
    PN_save=f"Landscape{nn_type}{W_idx}Act{act_idx}"
    PN=f"{PN_save}.npz"
    with np.load(PN) as a:
        x=a["X"]
        y=a["Y"]
        z=a["Z"]
    data1=(x,y,z)
    plot(data1, data1, f"{PN_save}.html")
    
    
