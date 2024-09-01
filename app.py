# IMPORT NECESSARY LIBRARIES
import os
import re
import streamlit as st
import pandas as pd
from textwrap import dedent
from crewai import Agent, Task, Crew
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_MODEL'] = 'gpt-4o-mini-2024-07-18'

# CSS TO HIDE THE STREAMLIT MENU
hide_menu = """<style> p {font-weight: 600;} div.block-container{padding-top:3rem;padding-bottom:3rem;} header{ visibility: hidden; } footer{ visibility: hidden; } </style> """

# CONFIGURE STREAMLIT PAGE SETTINGS
st.set_page_config(page_title="j33ni.ai",layout="wide",initial_sidebar_state="auto",page_icon='🔧')
#st.markdown(hide_menu, unsafe_allow_html=True) # HIDE THE STREAMLIT MAIN MENU

# CREWAI: AGENT
prompt_engineer_agent = Agent(
    role = "prompt engineer specialist",
    goal = "create a detailed prompts for the {ai_model} using the {method} prompting technique to execute {task}, following the best prompt engineering strategies for that model",
    backstory = "You are working in a team that creates prompts for the {ai_model} using the {method} prompting technique to execute {task}.Once you create a prompt, you will check with the prompt verification specialist. He will review the prompt.", allow_delegation = False, verbose = False)

prompt_verification_agent = Agent(
    role = "prompt verification specialist",
    goal = "create a prompts to execute {task} for the {ai_model} using the {method} prompting technique. verify whether the prompt technique is used, if any relevant roles are added, and if any example included.",
    backstory = "You are working in a team that creates prompts for the {ai_model} using the {method} prompting technique to execute {task}.Once the prompt engineer specialist creates a prompt, you need to verify whether the {method} prompting technique is used, if any relevant roles are added, and if an example is included. you meticulously review each prompt based on original prompt paper and provide your feedback. you will share the correction with the prompt engineer specialist to implement those changes.", allow_delegation = False, verbose = False)

# CREWAI: TASK
prompt_preparation = Task(
    description = dedent("""\
        conduct thorough research on the task, break it to actionable steps, and craft a prompt template tailored for the specified taskusing the indicated promptingmethod.
        task description: {task}
        prompting method: {method}
        language model: {ai_model}
        example: {example}
        output format: {output_format}

        when employing the {method} prompting approach, construct a prompt that effectively directs the {ai_model} to execute the specified {task}. Ensure the prompt contains these essential components:
        1. **Context:** Briefly describe the situation or challenge that the AI needs to address.
        2. **Roles:** Define any specific roles or personas the AI should adopt to effectively perform the task.
        3. **Instructions:** Specify the precise steps or actions the AI must undertake to complete the task successfully.
        4. **Constraints:** List any restrictions or rules that the AI must adhere to while performing the task.
        5. **Examples:** Provide concrete examples or scenarios that illustrate the task, helping to guide the AI’s responses and ensuring alignment with the desired outcomes.
        6. **Output Format:** Think step by step and Clearly present the output in a format that will help the user achieve the task with minimal effort.
        7. **Evaluation Criteria:** Establish metrics or criteria to assess the AI’s performance, ensuring the quality and relevance of its output.
        
        IMPORTANT: MAKE SURE YOU ONLY OUTPUT THE PROMPT DELIMTED BY <PROMPT></PROMPT> Tag"""
    ),
    expected_output = "a well structured instructional prompt",
    async_execution = True,
    agent = prompt_engineer_agent,
    verbose = True
)

prompt_verification = Task(
    description = dedent("""\
         Cross-verify the prompt created by the Prompt Engineer Specialist and transform it into a well-detailed instruction to guide a robot to execute the task {task}."""
    ),
    expected_output = "a well structured final prompt for others to use un executing the task {task} ",
    async_execution = False,
    agent = prompt_engineer_agent,
    verbose = False
)

# CREWAI: CREW
crew = Crew(
    agents = [prompt_engineer_agent, prompt_verification_agent],
    tasks = [prompt_preparation],
    verbose = False, memory = False, async_tasks_limit = 2
)

task = ""
method = "Tree of Thoughts"
ai_model = "chatgpt"
example = ""
output_format = "text"


# SIDEBAR CONTENT
def home_sidebar_content():
    with st.sidebar:
        openapi_ai_key = st.text_input('Add your OpenAI API key', 'sk-xxxxxxxxxxxxxxxxxxxxxxxx')
        if st.button("Submit"):
            return openapi_ai_key
    return None

# MAIN PAGE CONTENT
def home_main_page_content():
    context_text = ''
    content = ''

    ## SECTION1
    col_ctnr_3_a, col_ctnr_3_b = st.columns([3,3],gap="medium")

    ## SECTION1 - LEFT
    with col_ctnr_3_a:
        st.write(':violet[**Enter your task:**]')
        task = st.text_area("**Please provide detailed information about the task you need help with:**", value="", height=20,help="Describe the task in detail to help us understand your needs better.")
        st.write(':violet[**Additional Context:**]')
        context_text = st.text_area("**Need to provide more details? Simply type them into the text box below. If you have a document that adds context, you can easily upload it using the file uploader just beneath the text box.**", height=250)
        uploaded_file = st.file_uploader("Upload your context file here. We support CSV and TXT files.")
        req_btn_res = st.button("Submit",use_container_width=True)

    ## SECTION1 - LEFT
    with col_ctnr_3_b:
        st.write(':violet[**Sample Input:**] \n 1. Find the hardcoded value in the provided PySpark code. \n 2. Convert SQL code to PySpark code. \n 3. Write unit test cases for the uploaded code. \n 4. Create a user story for the specified task. \n 5. List tasks based on the uploaded architecture diagram. \n 6. Analyze performance bottlenecks in the existing codebase. \n 7. Suggest optimizations for database queries based on the schema. \n 8. Generate API documentation from the source code comments. \n 9. Prepare a migration strategy for upgrading from Python 2 to Python 3. \n 10. Design a security audit checklist for a web application.')
    
    st.markdown("---")

    ## SECTION 2
    if req_btn_res:
        prompt = crew.kickoff(inputs={"task":task, "method":method, "ai_model":ai_model, "example":example, "output_format":output_format})
        pattern = r"<PROMPT>(.*?)</PROMPT>" 
        match = re.search(pattern, prompt.raw, re.DOTALL)
        if match: 
            extracted_text = match.group(1).strip() 

        PROMPT_TEMPLATE = """
        {prompt_text}
        <actual_problem>
        {question}
        </actual_problem>
        """

        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE, input_variables=["prompt_text", "question"]
        )

        if uploaded_file is not None:
            file_type = uploaded_file.name.split('.')[-1]
            if file_type.lower() == 'csv':
                df = pd.read_csv(uploaded_file)
                content = df.to_string()
            elif file_type.lower() == 'txt':
                content = str(uploaded_file.read(), 'utf-8')
            else:
                st.error("Unsupported file format!")

        llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
        rag_chain = (
            prompt
            | llm
            | StrOutputParser()
        )

        merged_context_text = context_text + content
        res = rag_chain.invoke({"prompt_text": extracted_text , "question":merged_context_text})

        output_prompt = f"""

{extracted_text}

**Actual_problem:**
{merged_context_text}"""

        col_ctnr_4_a, col_ctnr_4_b = st.columns([2,4],gap="medium")
        with col_ctnr_4_a:
            st.write(":violet[**Here's the detailed prompt generated for your task:**]")
            st.code(output_prompt)
               
        with col_ctnr_4_b:
            st.write(':violet[**Final Output:**]')
            st.write(res)

        # with st.expander("Here's the detailed prompt generated for your task:"):
        #     st.write(f"{extracted_text}\n<actual_problem>\n{merged_context_text}</actual_problem>")

        # st.subheader('Final Output:')
        # st.write("")
        # st.write(res)

        st.markdown("---")
        
        
    st.markdown("---")

# MAIN FUNCTION
if __name__ == "__main__":

    # SIDE BAR
    # open_ai_key = home_sidebar_content()
    st.title(':gray[Effortlessly  execute your tasks with ] :blue[  **Jeeni.ai**  ]')
    st.markdown("---")
    st.write('')
    # MAIN PAGE
    home_main_page_content()