import numpy as np

#Auto Determine Polarization for HPOL & VPOL Files
def determine_polarization(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
        if "Horizontal Polarization" in content:
            return "HPol"
        else:
            return "VPol"
        
#Verify angle data and frequencies are not mismatched      
def angles_match(start_phi_h, stop_phi_h, inc_phi_h, start_theta_h, stop_theta_h, inc_theta_h,
                            start_phi_v, stop_phi_v, inc_phi_v, start_theta_v, stop_theta_v, inc_theta_v):

    return (start_phi_h == start_phi_v and stop_phi_h == stop_phi_v and inc_phi_h == inc_phi_v and
            start_theta_h == start_theta_v and stop_theta_h == stop_theta_v and inc_theta_h == inc_theta_v)

#Extract Frequency points for selection in the drop-down menu      
def extract_passive_frequencies(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines()

    # Extracting frequencies
    frequencies = [float(line.split("=")[1].split()[0]) for line in content if "Test Frequency" in line]

    return frequencies

#Calculate Total Gain Vector and add cable loss etc - Use Phase for future implementation?
def calculate_passive_variables(hpol_data, vpol_data, cable_loss, start_phi, stop_phi, inc_phi, start_theta, stop_theta, inc_theta, freq_list, selected_frequency):
    theta_points = int((stop_theta - start_theta) / inc_theta + 1)
    phi_points = int((stop_phi - start_phi) / inc_phi + 1)
    data_points = theta_points * phi_points

    theta_angles_deg = np.zeros((data_points, len(freq_list)))
    phi_angles_deg = np.zeros((data_points, len(freq_list)))
    v_gain_dB = np.zeros((data_points, len(freq_list)))
    h_gain_dB = np.zeros((data_points, len(freq_list)))
    v_phase = np.zeros((data_points, len(freq_list)))
    h_phase = np.zeros((data_points, len(freq_list)))

    for m, (hpol_entry, vpol_entry) in enumerate(zip(hpol_data, vpol_data)):
        for n, (theta_h, phi_h, mag_h, phase_h, theta_v, phi_v, mag_v, phase_v) in enumerate(zip(hpol_entry['theta'], hpol_entry['phi'], hpol_entry['mag'], hpol_entry['phase'], vpol_entry['theta'], vpol_entry['phi'], vpol_entry['mag'], vpol_entry['phase'])):
            v_gain = mag_v
            h_gain = mag_h
            v_ph = phase_v
            h_ph = phase_h

            theta_angles_deg[n, m] = theta_h
            phi_angles_deg[n, m] = phi_h
            v_gain_dB[n, m] = v_gain
            h_gain_dB[n, m] = h_gain
            v_phase[n, m] = v_ph
            h_phase[n, m] = h_ph

    cable_loss_matrix = np.ones((phi_points * theta_points, len(freq_list))) * cable_loss
    v_gain_dB += cable_loss_matrix
    h_gain_dB += cable_loss_matrix

    Total_Gain_dB = 10 * np.log10(10**(v_gain_dB/10) + 10**(h_gain_dB/10))
    
   
    return theta_angles_deg, phi_angles_deg, v_gain_dB, h_gain_dB, Total_Gain_dB
