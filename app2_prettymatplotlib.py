import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.ticker as mtick
import seaborn as sns
import squarify
import pandas as pd
# from sklearn import preprocessing

def app():

    st.title('Pretty Matplotlib')
    st.write('Generate clean matplotlib code automatically. Just use this template forever')

    df = pd.read_csv('datasets/app2/kaggle_survey_2020_responses.csv')
    colors = [
        "#0a3d62", # main
        "#008294",
        "#4b4b4c",
        "#676767",
        "#808080",
        "#989898",
        "#fbfbfb", # background
    ]

    st.subheader('Color Palette')
    f, axes = plt.subplots(1, len(colors), figsize=(10,1), dpi=600)
    for i, ax in enumerate(axes.ravel()):
        ax.add_patch(Rectangle(xy = (0,0),width=20, height=18, facecolor=colors[i]))
        ax.set_axis_off()
    st.pyplot(f)
    st.write('|'.join(colors))

    
    chart_type = ['Bar Chart', 'Heatmap']
    chart_select = st.selectbox('Chart Type', chart_type)


    # Hyperparameters
    st.sidebar.subheader('General Chart Properties')
    w = st.sidebar.slider('Width', 1, 30, 12, 1)
    h = st.sidebar.slider('Height', 1, 30, 6, 1)
    background_color = st.sidebar.color_picker('Background Color', colors[-1])
    fontsize = st.sidebar.slider('Font Size', 10, 100, 20, 1)
    fontfamily = st.sidebar.selectbox('Font Family', ['serif','sans-serif','monospace','cursive','fantasy'])
    title_x_offset = st.sidebar.slider('Title Offset from Left', -5.0, 20.0, -1.5, 0.25)


    # Bar charts
    if chart_select == 'Bar Chart':
        st.header('Bar Chart')

        st.sidebar.header('Bar Chart Properties')

        x_lab = st.sidebar.text_input('X Label', 'Coding Experience')
        y_lab = st.sidebar.text_input('Y Label','Number of Kagglers')
        t = st.sidebar.text_input('Title','Kagglers Coding Experience')
        subt = st.sidebar.text_input('Subtitle','Most Kagglers have between 1-5 years of coding experience')
        bar_color = st.sidebar.color_picker('Bar color', colors[1])
        
        

        st.sidebar.subheader('Additional for Clearner Bar chart')
        spacing = st.sidebar.slider('Annotation Spacing', 1, 100, 10)

        data = df['Q6'].loc[1:].value_counts().reset_index()
        data.columns = ['x','y']

        fig,ax = plt.subplots(1,1, figsize=(w, h)) # create figure
        ax.bar(data['x'], data['y'], color=bar_color, zorder=3)
        background_color = background_color
        fig.patch.set_facecolor(background_color)
        ax.set_ylim(0, int(data.y.max()/100)*100)
        ax.set_facecolor(background_color)
        ax.grid(color='black', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
        ax.text(title_x_offset, data['y'].max()*1.125, t, fontsize=fontsize, fontweight='bold', fontfamily=fontfamily)
        ax.text(title_x_offset, data['y'].max()*1.05, subt, fontsize=int(np.ceil(fontsize*0.75)), fontweight='light', fontfamily=fontfamily)
        ax.set_xlabel(x_lab, fontsize=int(np.ceil(fontsize*0.75)), fontfamily=fontfamily)
        ax.set_ylabel(y_lab, fontsize=int(np.ceil(fontsize*0.75)), fontfamily=fontfamily)
        ax.set_xticklabels(data['x'], fontsize=int(np.ceil(fontsize*0.6)), fontfamily=fontfamily)
        ax.set_yticklabels(ax.get_yticks().tolist(), fontsize=int(np.ceil(fontsize * 0.6)), fontfamily=fontfamily)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
        st.pyplot(fig)

        st.markdown('''
        ### Code
        The code assumes that the data is in `data` object with two columns `x` and `y`
        ```python
        
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(1,1, figsize=({w}, {h}) # create figure
        
        # bar plot
        ax.bar(data['x'], data['y'], color='{bar_color}', zorder=3)
        
        # design elements
        background_color = '{c}'
        fontsize = {fontsize}
        fig.patch.set_facecolor(background_color)
        ax.set_ylim(0, int(data.y.max()/100)*100)
        ax.set_facecolor(background_color)
        ax.grid(color='black', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
        ax.text({title_x_offset}, data['y'].max()*1.125, '{t}', fontsize={fontsize}, fontweight='bold', fontfamily='{fontfamily}')
        ax.text({title_x_offset}, data['y'].max()*1.05, '{subt}', fontsize=int(np.ceil(fontsize*0.75)), fontweight='light', fontfamily='{fontfamily}')
        ax.set_xlabel('{x_lab}', fontsize=int(np.ceil(fontsize*0.75)), fontfamily='{fontfamily}')
        ax.set_ylabel('{y_lab}', fontsize=int(np.ceil(fontsize*0.75)), fontfamily='{fontfamily}')
        ax.set_xticklabels(data['x'], fontsize=int(np.ceil(fontsize*0.6)), fontfamily='{fontfamily}')
        ax.set_yticklabels(ax.get_yticks().tolist(), fontsize=int(np.ceil(fontsize * 0.6)), fontfamily='{fontfamily}')
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
        plt.show()
        ```
        '''.format(w=w, h=h, bar_color=bar_color, fontsize=fontsize, fontfamily=fontfamily, c=colors[-1],
                   x_lab=x_lab, y_lab=y_lab, t=t, subt=subt, title_x_offset=title_x_offset
                   ))

        st.header('Cleaner Bar Chart')

        fig,ax = plt.subplots(1,1, figsize=(w, h)) # create figure
        ax.bar(data['x'], data['y'], color=bar_color, zorder=3)
        background_color = background_color
        fig.patch.set_facecolor(background_color)
        ax.set_ylim(0, int(data.y.max()/100)*105)
        ax.set_facecolor(background_color)
        # ax.grid(color='black', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
        ax.text(title_x_offset, data['y'].max()*1.175, t, fontsize=fontsize, fontweight='bold', fontfamily=fontfamily)
        ax.text(title_x_offset, data['y'].max()*1.10, subt, fontsize=int(np.ceil(fontsize*0.75)), fontweight='light', fontfamily=fontfamily)
        ax.set_xlabel(x_lab, fontsize=int(np.ceil(fontsize*0.75)), fontfamily=fontfamily)
        ax.set_xticklabels(data['x'], fontsize=int(np.ceil(fontsize*0.6)), fontfamily=fontfamily)
        ax.yaxis.set_ticks([])
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)


        for rect in ax.patches:
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2
            space = spacing * -1 if y_value < 0 else spacing
            va = 'top' if y_value < 0 else 'bottom'
            label = int(y_value)
            ax.annotate(label,  (x_value, y_value), xytext=(0, space), textcoords="offset points", ha='center',va=va, fontfamily=fontfamily, fontsize=int(np.ceil(fontsize*0.6)))

        st.pyplot(fig)

        st.markdown('''
            ### Code
            The code assumes that the data is in `data` object with two columns `x` and `y`
            ```python
    
            import matplotlib.pyplot as plt
            fig,ax = plt.subplots(1,1, figsize=({w}, {h}) # create figure
    
            # bar plot
            ax.bar(data['x'], data['y'], color='{bar_color}', zorder=3)
    
            # design elements
            background_color = '{c}'
            fontsize = {fontsize}
            spacing = {spacing}
            fig.patch.set_facecolor(background_color)
            ax.set_ylim(0, int(data.y.max()/100)*100)
            ax.set_facecolor(background_color)
            ax.text({title_x_offset}, data['y'].max()*1.175, '{t}', fontsize={fontsize}, fontweight='bold', fontfamily='{fontfamily}')
            ax.text({title_x_offset}, data['y'].max()*1.1, '{subt}', fontsize=int(np.ceil(fontsize*0.75)), fontweight='light', fontfamily='{fontfamily}')
            ax.set_xlabel('{x_lab}', fontsize=int(np.ceil(fontsize*0.75)), fontfamily='{fontfamily}')
            ax.set_xticklabels(data['x'], fontsize=int(np.ceil(fontsize*0.6)), fontfamily='{fontfamily}')
            ax.yaxis.set_ticks([])
            
            # spines
            for s in ["top", "right", "left"]:
                ax.spines[s].set_visible(False)
            
            # annotations    
            for rect in ax.patches:
                y_value = rect.get_height()
                x_value = rect.get_x() + rect.get_width() / 2
                space = spacing*-1 if y_value < 0 else spacing
                va = 'top' if y_value < 0 else 'bottom'
                label = int(y_value)
                ax.annotate(label, (x_value, y_value), xytext=(0, space), textcoords="offset points", ha='center', va=va, fontfamily='{fontfamily}', fontsize=int(np.ceil(fontsize*0.6)))
            plt.show()
            ```
            '''.format(w=w, h=h, bar_color=bar_color, fontsize=fontsize, fontfamily=fontfamily, c=colors[-1],
                       x_lab=x_lab, y_lab=y_lab, t=t, subt=subt, title_x_offset=title_x_offset, spacing=spacing))

    if chart_select == 'Heatmap':
        st.header('Heatmap')

        st.sidebar.subheader('Heatmap Properties')
        accent_color = st.sidebar.color_picker('Accent Color', colors[1])
        t = st.sidebar.text_input('Title','Correlation Matrix')
        subt = st.sidebar.text_input('Subtitle','Correlation between variables of car crashes dataset')

        data = sns.load_dataset('car_crashes')
        

        f,ax = plt.subplots(1,1, figsize=(w,h), facecolor=background_color)
        ax.set_facecolor(background_color)
        ax.text(title_x_offset, -1, t, fontsize=fontsize, fontweight='bold', fontfamily=fontfamily)
        ax.text(title_x_offset, -.5, subt, fontsize=int(np.ceil(fontsize*0.75)), fontweight='light', fontfamily=fontfamily)
        colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", [accent_color, colors[-1], accent_color])
        sns.heatmap(data.corr(), ax=ax, annot=True, vmin=-1, vmax=1, square=True, cbar=False, cmap=colormap, fmt='.0g')
        ax.set_xticklabels(ax.get_xticklabels(), fontfamily=fontfamily, fontsize=fontsize/2.5, rotation=90, ha='center')
        ax.set_yticklabels(ax.get_yticklabels(), fontfamily=fontfamily, fontsize=fontsize/2.5, rotation=0, va='center')
        st.pyplot(f)

        st.markdown('''
            ### Code
            The code assumes that the data is in `data` object with two columns `x` and `y`
            ```python
    
            import matplotlib.pyplot as plt
            import seaborn as sns

            data = sns.load_dataset('car_crashes')
            fig,ax = plt.subplots(1,1, figsize=({w}, {h}) # create figure
        
            # heatmap
            background_color = '{background_color}'
            f,ax = plt.subplots(1,1, figsize=(w,h), facecolor=background_color)
            ax.set_facecolor(background_color)
            ax.text({title_x_offset}, -1, '{t}', fontsize=fontsize, fontweight='bold', fontfamily='{fontfamily}')
            ax.text({title_x_offset}, -.5, '{subt}', fontsize=int(np.ceil(fontsize*0.75)), fontweight='light', fontfamily='{fontfamily}')
            colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['{accent_color}', '{background_color}', '{accent_color}'])
            sns.heatmap(data.corr(), ax=ax, annot=True, vmin=-1, vmax=1, square=True, cbar=False, cmap=colormap, fmt='.0g')
            ax.set_xticklabels(ax.get_xticklabels(), fontfamily=fontfamily, fontsize=fontsize/2.5, rotation=90, ha='center')
            ax.set_yticklabels(ax.get_yticklabels(), fontfamily=fontfamily, fontsize=fontsize/2.5, rotation=0, va='center')
            plt.show()
            ```
            '''.format(w=w, h=h, accent_color=accent_color, fontsize=fontsize, fontfamily=fontfamily, c=colors[-1],
                       t=t, subt=subt, title_x_offset=title_x_offset, background_color=background_color))

