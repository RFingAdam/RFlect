from calculations import angles_match, calculate_passive_variables, calculate_active_variables
from plotting import plot_passive_3d_component, plot_2d_passive_data, plot_active_2d_data, plot_active_3d_data
from file_utils import read_active_file, read_passive_file

from tkinter import simpledialog, filedialog, Tk
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT

import os
import base64
import requests
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from the openai.env file
load_dotenv('openapi.env')

# Retrieve the OpenAI API key
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY2")
    )

# Helper function to encode an image as a base64 string for OpenAI API
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
class RFAnalyzer:
    def __init__(self):
        self.messages = []

    '''def send_message(self, message):
        self.messages.append({"role": "user", "content": message})  # Append the user message to the messages list
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # The available models are listed at https://beta.openai.com/docs/api-reference/models/list
            messages=self.messages,  # The messages exchanged so far
            max_tokens=1500,  # 1500 is the default, but can be adjusted to generate longer or shorter completions
            temperature=0.7,  # 0.7 is the default, but can be adjusted to increase or decrease creativity
            top_p=1,  # 1 is the default, but can be adjusted to increase or decrease randomness
            n=1,  # 1 is the default, but can be adjusted to generate multiple completions
            stop=None,  # Stop the completion when receiving a stop sequence
        )
        
        reply = response.choices[0].message.content  # Extract the completion from the API response
        self.messages.append({"role": "assistant", "content": reply})  # Append the system reply to the messages list
        return reply'''
        
    def analyze_image(self, image_path):
        base64_image = encode_image(image_path)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY2')}"
        }
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": "You are an expert RF Engineer analyzing antenna measurements including, gain, efficiency, directivity, 2D/3D patterns, and/or S-Parameters. Please provide a short, but detailed analysis of the images calling out and min/max/averages present. Based on the text file name or frequency range given, one should be able to assume the band of operation. ie if a scan is 2300-2600MHz, this is likely BLE or 2.4GHz Wifi, and the band is about 2.4-2.48GHz for any notable informations. There is no need to provide an analysis of data outside the band."
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                    }
                ]
                }
            ],
            "max_tokens": 300,
            "temperature": 0.0,
            "top_p": 0.1,
            }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        result = response.json()
        reply = result['choices'][0]['message']['content']
        self.messages.append({"role": "assistant", "content": reply}) 
        print(response.json())
        return reply
        
def generate_report(doc_title, images, save_path, analyzer, logo_path=None):
    print("Starting Report Generation...")
    # Create a new Document
    doc = Document()
    
    # Add logo to header if provided
    if logo_path and os.path.exists(logo_path):
        header = doc.sections[0].header
        paragraph = header.paragraphs[0]
        run = paragraph.add_run()
        run.add_picture(logo_path, width=Inches(1))
        paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER

    # Add title to the document
    doc.add_heading(doc_title, 0)

    # Add images with captions and analysis to the document
    for img_path in images:
        if img_path is not None and os.path.exists(img_path):
            doc.add_picture(img_path, width=Inches(6))
            analysis = analyzer.analyze_image(img_path)
            doc.add_paragraph(analysis)
            doc.add_page_break()
        else:
            print(f"Warning: Image path {img_path} does not exist or is invalid.")
    
    # Save the document
    doc.save(os.path.join(save_path, f"{doc_title}.docx"))
    print("Report Generation Complete!")


def save_to_results_folder(selected_frequency, freq_list, scan_type, hpol_path, vpol_path, active_path, cable_loss, datasheet_plots, word=False, logo_path=None):
    # Initialize the GUI
    root = Tk()
    root.withdraw()  # Hide the main window

    # Prompt the user for the project name
    project_name = simpledialog.askstring("Input", "Enter Project Name:")
    
    # Prompt the user to select a directory
    directory = filedialog.askdirectory(title="Select Directory to Save Project")
    
    # Check if user provided a project name and directory
    if not project_name or not directory:
        print("Project name or directory not provided. Exiting...")
        return
    
    # Create the directory structure
    project_path = os.path.join(directory, project_name)
    two_d_data_path = os.path.join(project_path, "2D Plots")
    if word:
        report_path = os.path.join(project_path, "Report")
        os.makedirs(report_path, exist_ok=True)
    if scan_type == "active":
        three_d_data_path = os.path.join(project_path, "3D Plots")
        os.makedirs(three_d_data_path, exist_ok=True)
    elif scan_type == "passive":
        user_selected_frequency_folder_name = f"3D Plots at {selected_frequency} MHz"
        user_selected_frequency_path = os.path.join(project_path, user_selected_frequency_folder_name)
        os.makedirs(user_selected_frequency_path, exist_ok=True)
    os.makedirs(two_d_data_path, exist_ok=True)
          
    # Call the modified plotting functions with save_path argument to save the plots
    if scan_type == "active":
        # TODO Perform active calculations and plotting method calls
        # Assuming active data has been pre-processed similarly to how passive data is handled
        data = read_active_file(active_path)
        
        # Unpack the data
        (frequency, start_phi, start_theta, stop_phi, stop_theta, inc_phi, inc_theta,
        calc_trp, theta_angles_deg, phi_angles_deg, h_power_dBm, v_power_dBm) = (
            data["Frequency"], data["Start Phi"], data["Start Theta"], data["Stop Phi"], 
            data["Stop Theta"], data["Inc Phi"], data["Inc Theta"], data["Calculated TRP(dBm)"],
            data["Theta_Angles_Deg"], data["Phi_Angles_Deg"], data["H_Power_dBm"], data["V_Power_dBm"]
        )

        # Calculate active variables
        active_variables = calculate_active_variables(start_phi, stop_phi, start_theta, stop_theta, inc_phi, inc_theta, h_power_dBm, v_power_dBm)

        # Unpack calculated active variables
        (data_points, theta_angles_deg, phi_angles_deg, theta_angles_rad, phi_angles_rad, total_power_dBm_2d,
        total_power_dBm_min, total_power_dBm_nom, h_power_dBm_2d, h_power_dBm_min, v_power_dBm_2d,
        v_power_dBm_min, h_power_dBm_nom, v_power_dBm_nom, TRP_dBm, h_TRP_dBm, v_TRP_dBm) = active_variables
        
        # Plot and save the 2D and 3D data (instead of displaying)
        print("Saving 2D Active Plots...")
        plot_active_2d_data(data_points, theta_angles_rad, phi_angles_rad, total_power_dBm_2d, frequency, save_path=two_d_data_path)
        
        print("Saving 3D Active Plots...")
        plot_active_3d_data(theta_angles_deg, phi_angles_deg, total_power_dBm_2d, frequency, power_type='total', interpolate=True, save_path=three_d_data_path)
        plot_active_3d_data(theta_angles_deg, phi_angles_deg, h_power_dBm_2d, frequency, power_type='hpol', interpolate=True, save_path=three_d_data_path)
        plot_active_3d_data(theta_angles_deg, phi_angles_deg, v_power_dBm_2d, frequency, power_type='vpol', interpolate=True, save_path=three_d_data_path)
    
    elif scan_type == "passive":
        # After reading & parsing, hpol_data and vpol_data will be lists of dictionaries. 
        # Each dictionary will represent a frequency point and will contain:
        #'frequency': The frequency value
        #'cal_factor': Calibration factor for that frequency
        #'data': A list of tuples, where each tuple contains (theta, phi, mag, phase).

        parsed_hpol_data, start_phi_h, stop_phi_h, inc_phi_h, start_theta_h, stop_theta_h, inc_theta_h = read_passive_file(hpol_path)
        hpol_data = parsed_hpol_data

        parsed_vpol_data, start_phi_v, stop_phi_v, inc_phi_v, start_theta_v, stop_theta_v, inc_theta_v = read_passive_file(vpol_path)
        vpol_data = parsed_vpol_data
        
        # Check to see if selected files have mismatched frequency or angle data
        angles_match(start_phi_h, stop_phi_h, inc_phi_h, start_theta_h, stop_theta_h, inc_theta_h,
                        start_phi_v, stop_phi_v, inc_phi_v, start_theta_v, stop_theta_v, inc_theta_v)

        # Call Methods to Calculate Required Variables and Set up variables for plotting 
        passive_variables = calculate_passive_variables(hpol_data, vpol_data, cable_loss, start_phi_h, stop_phi_h, inc_phi_h, start_theta_h, stop_theta_h, inc_theta_h, freq_list, selected_frequency)
        theta_angles_deg, phi_angles_deg, v_gain_dB, h_gain_dB, Total_Gain_dB = passive_variables
        
        # Call Method to Plot Passive Data
        print("Plotting 2D Passive Data...")
        plot_2d_passive_data(theta_angles_deg, phi_angles_deg, v_gain_dB, h_gain_dB, Total_Gain_dB, freq_list, selected_frequency, datasheet_plots, save_path=two_d_data_path)
        
        print("Plotting 3D Passive Data...")
        plot_passive_3d_component(theta_angles_deg, phi_angles_deg, v_gain_dB, h_gain_dB, Total_Gain_dB, freq_list, float(selected_frequency), gain_type="total", save_path=user_selected_frequency_path)
        plot_passive_3d_component(theta_angles_deg, phi_angles_deg, v_gain_dB, h_gain_dB, Total_Gain_dB, freq_list, float(selected_frequency), gain_type="hpol", save_path=user_selected_frequency_path)
        plot_passive_3d_component(theta_angles_deg, phi_angles_deg, v_gain_dB, h_gain_dB, Total_Gain_dB, freq_list, float(selected_frequency), gain_type="vpol", save_path=user_selected_frequency_path)
    
    print(f"Data saved to {project_path}")
# Changed word routine to be called from separate method to select a root directory for the report to include all test frequencies, not just the one being processed at the time
"""
    if word:
        # Collect the paths of the generated images
        for filename in os.listdir(two_d_data_path):
            if filename.endswith(".png"):
                image_paths.append(os.path.join(two_d_data_path, filename))
        
        # 3D plots
        for gain_type in ["total", "hpol", "vpol"]:
            plot_path = plot_passive_3d_component(theta_angles_deg, phi_angles_deg, v_gain_dB, h_gain_dB, Total_Gain_dB, freq_list, float(selected_frequency), gain_type=gain_type, save_path=two_d_data_path)
            image_paths.append(plot_path)
    
        # Initialize RFAnalyzer
        analyzer = RFAnalyzer()
        
        # Generate the report
        generate_report(project_name, image_paths, report_path, analyzer, logo_path)
        print(f"Report saved to {report_path}")
        """