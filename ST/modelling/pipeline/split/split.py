import streamlit as st

# Debug: st.write("This is: modelling/pipeline/split/split.py")
# Debug: st.write("\n\n Prompt the user to select a dataset split.")


if 'split_val' not in st.session_state:
    st.session_state.split_val = None
    
value = st.text_input("Enter a value for the test/train split fraction:", value=0.8)


st.session_state.split_val = value
text = f"Split value set to {value}"
st.success(text)
