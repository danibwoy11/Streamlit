import streamlit as st
from PIL import Image



def home_page():
    
    
    st.title('Telco Churn Classification App')

    st.markdown('''
    This app uses machine learning to classify whether a customer is likely to churn
                ''')
    st.subheader('Instructions')
    st.markdown('''
        - upload a csv file
        -select the features for the classification
        - Choose a machine learning model form the dropdown
        -Click on 'Classify' to get the predicted results
        -The app gives you a report on the performance of the model
        - Expect it to give metrics like the f1 score, recall, prcision and accuracy        
                ''')
    
    st.header('App Features')
    st.markdown('''
                - **Data View**: Access the customer data.
                - **Predict View**: Shows the various models and predictons you will make
                - **Dashboard**: Shows data visualizaitons for insights
    ''')
    
    st.subheader('User Benefits')
    st.markdown('''
            - **Data Driven Decisions**: You make an informed decision backed by data
            - **Access Machine**: utilize machine learning
    '''
    )

    st.write('#### How to run the application ')
    with st.container(border= True):
        st.code('''
            # Activate the virtual environment
                env/scripts/activate

                #Run the app
                streamlit run p.py
        ''')

    st.video('https://www.youtube.com/watch?v=HfN67NC1AVw&t=52s', autoplay= True, muted=True)
    
    #adding the clickable link
    st.markdown('[Watch a demo](https://www.youtube.com/watch?v=i1xqq8IB8ZA)' )

    # st.markdown('[a demo](r:C:\Users\Alucard\Videos\Captures\Sekiro 2024-10-16 22-48-29.mp4)')

    

    st.divider()
    st.write('+++'*15)

    st.write('Need Help? ')
    st.write('Contact me on:')

    #add an image/ way 1
    
    st.markdown('[Visit LinkedIn Profile](www.linkedin.com/in/daniel-nortey-954952328)')


if __name__ == "__main__":
    home_page()