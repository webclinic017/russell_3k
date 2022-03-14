import streamlit as st


class Credentials(object):

    def __init__(self, today_stamp):
        self.today_stamp = today_stamp
        self.authUser = False


    def check_password(self):
        try:
            access_dict = st.secrets["passwords"]
            nameUser = list(access_dict.keys())[0]
            passWord = list(access_dict.values())[0]
            if nameUser == "admin" and passWord == "lord_gordon":
                self.authUser = True
                return self.authUser
        except:
            pass


        def password_entered():
            # Checks whether a password entered by the user is correct.
            st.session_state.username = list(st.secrets["passwords"].keys())[0]
            st.session_state.password = list(st.secrets["passwords"].values())[0]
            condition_A = st.session_state["username"] in st.secrets["passwords"]
            condition_B = st.session_state["password"] == st.secrets["passwords"][st.session_state["username"]]

            if condition_A == True and condition_B ==True:
                st.session_state["password_correct"] = True
                # don't store username + password
                del st.session_state["password"]
                del st.session_state["username"]
            else:
                st.session_state["password_correct"] = False


        if "password_correct" not in st.session_state:
            st.title(f" ~ 4M ~  \n\- Today Date: [{self.today_stamp}]\n")
            st.write(f"{'__'*25} \n {'__'*25}")

            with st.form('login_credentials'):
                nameUser = st.text_input('Username', key="username")
                wordPass = st.text_input('Password', key="password")
                st.form_submit_button('Login', help="click to submit login credentials", on_click=password_entered)
                self.authUser = False
                return self.authUser


        elif not st.session_state["password_correct"]:
            st.title(f" ~ 4M ~  \n\ - Today Date: [{self.today_stamp}]\n")
            st.write(f"{'__'*25} \n {'__'*25}")
            
            with st.form('login_credentials'):
                nameUser = st.text_input('Username', key='username')
                wordPass = st.text_input('Password', key='password')
                st.form_submit_button('Login', help="click to submit login credentials", on_click=password_entered)
                st.error("😕 User not known or password incorrect")
                self.authUser = False
                return self.authUser

        else:
            self.authUser = True
            return self.authUser
