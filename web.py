import streamlit as st
import subprocess
import os


PYTHON_EXE_PATH = r'D:\Program\Anaconda\envs\py312\python.exe'

config_file = st.text_input('config_file', 'configs/single/train-template.toml')
ext_args = st.text_input('ext_args', '--help') # arg1;arg2;arg3;...

stdout_container = st.empty()
stderr_container = st.empty()

def main(config_file: str, ext_args: str):
    st.write(f"Executing with config_file: {config_file} and ext_args: {ext_args}")
    
    ext_args = [arg.strip() for arg in ext_args.split(';')]
    process = subprocess.Popen(
        [PYTHON_EXE_PATH, "main.py", "-c", config_file, *ext_args], 
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    with stdout_container.container(), stderr_container.container():
        for line in iter(process.stdout.readline, ''):
            st.write(f'<div style="border:1px solid black;padding:10px;">{line.strip()}</div>', unsafe_allow_html=True)
        for line in iter(process.stderr.readline, ''):
            st.write(f'<div style="border:1px solid red;padding:10px;">{line.strip()}</div>', unsafe_allow_html=True)

def main_pipeline(config_file: str):
    st.write(f"Running main_pipeline with config_file: {config_file} and ext_args: {ext_args}")

    process = subprocess.Popen(
        [PYTHON_EXE_PATH, "main_pipeline.py", "-c", config_file], 
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    with stdout_container.container(), stderr_container.container():
        for line in iter(process.stdout.readline, ''):
            st.write(f'<div style="border:1px solid black;padding:10px;">{line.strip()}</div>', unsafe_allow_html=True)
        for line in iter(process.stderr.readline, ''):
            st.write(f'<div style="border:1px solid red;padding:10px;">{line.strip()}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    if st.button('Run'):
        main(config_file, ext_args)

    if st.button('Run Main Pipeline'):
        main_pipeline(config_file)
