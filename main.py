import streamlit as st
import app1_logisticregressionwithnn
from libs.streamlithelper import show_logo


# Custom CSS
st.markdown("""
<style>
.reportview-container .main .block-container{max-width: 60%;}
</style>
""", unsafe_allow_html=True,)

PAGES = {
    "Logistic Regression with Neural Network Mindset": app1_logisticregressionwithnn,
}

c1, c2, c3 = st.beta_columns((3, 1, 3))
c2.image(show_logo(), width=150)
# st.markdown('---')
st.title('Go Break Some Eggs!')
st.subheader('Select Playground')
selection = st.selectbox("", list(PAGES.keys()))
st.markdown('---')
# Logo
page = PAGES[selection]
page.app()

# headings
